#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import collections
import itertools
import random

import numpy as np
from common_python.utils.common_func import Frame, create_cls

from agent_diy.conf.conf import Config


def _lineup_iterator_shuffle_cycle(camps):
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


def lineup_iterator_roundrobin_camp_heroes(camp_heroes=None):
    if not camp_heroes:
        raise Exception("camp_heroes is empty")
    camps = []
    for lineups in itertools.product(camp_heroes, camp_heroes):
        camps.append(list(lineups))
    return _lineup_iterator_shuffle_cycle(camps)


ObsData = create_cls("ObsData", feature=None, legal_action=None, logit_bias=None, lstm_cell=None, lstm_hidden=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    d_prob=None,
    value=None,
    lstm_cell=None,
    lstm_hidden=None,
)

SampleData = create_cls("SampleData", sample=sum([shape[0] for shape in Config.data_shapes]))

NONE_ACTION = [0, 15, 15, 15, 15, 0]


def sample_process(collector):
    return collector.sample_process()


def build_frame(agent, observation):
    obs_data, act_data = agent.obs_data, agent.act_data
    frame_state = observation["frame_state"]
    frame_no = frame_state["frame_no"]
    is_train = False
    for hero in frame_state.get("hero_states", []):
        if hero.get("camp") == agent.hero_camp:
            is_train = hero.get("hp", 0) > 0
            break

    feature_vec = np.array(obs_data.feature if obs_data.feature is not None else observation["feature"])
    reward = observation["reward"]["reward_sum"]
    sub_action_mask = observation["sub_action_mask"]
    prob, value = act_data.prob, act_data.value
    action = _normalize_action(act_data.action)
    act_data.action = action
    lstm_cell, lstm_hidden = act_data.lstm_cell, act_data.lstm_hidden
    legal_action = _update_legal_action(obs_data.legal_action, action)
    sub_action = _get_sub_action_mask(sub_action_mask, action)

    return Frame(
        frame_no=frame_no,
        feature=feature_vec.reshape([-1]),
        legal_action=legal_action.reshape([-1]),
        action=action,
        reward=reward,
        reward_sum=0,
        value=value.flatten()[0],
        next_value=0,
        advantage=0,
        prob=prob,
        sub_action=sub_action,
        lstm_info=np.concatenate([lstm_cell.flatten(), lstm_hidden.flatten()]).reshape([-1]),
        is_train=False if action[0] < 0 else is_train,
    )


def _update_legal_action(original_la, action):
    target_size = Config.LABEL_SIZE_LIST[-1]
    top_size = Config.LABEL_SIZE_LIST[0]
    original_la = np.array(original_la, dtype=np.float32)
    expected_size = sum(Config.LEGAL_ACTION_SIZE_LIST)
    if original_la.size != expected_size:
        fixed = np.ones(expected_size, dtype=np.float32)
        return _update_legal_action(fixed, action)
    fix_part = original_la[: -target_size * top_size]
    target_la = original_la[-target_size * top_size :]
    target_la = target_la.reshape([top_size, target_size])[action[0]]
    return np.concatenate([fix_part, target_la], axis=0)


def _normalize_action(action):
    if action is None:
        return list(NONE_ACTION)
    if isinstance(action, np.ndarray):
        action = action.tolist()
    if isinstance(action, tuple):
        action = list(action)
    while isinstance(action, list) and len(action) == 1 and isinstance(action[0], (list, tuple, np.ndarray)):
        action = action[0].tolist() if isinstance(action[0], np.ndarray) else list(action[0])
    if not isinstance(action, list) or len(action) != len(Config.LABEL_SIZE_LIST):
        return list(NONE_ACTION)
    fixed = []
    for idx, value in enumerate(action):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = NONE_ACTION[idx]
        fixed.append(max(0, min(value, Config.LABEL_SIZE_LIST[idx] - 1)))
    return fixed


def _get_sub_action_mask(sub_action_mask, action):
    default = [1, 1, 1, 1, 1, 1]
    if isinstance(sub_action_mask, dict):
        mask = sub_action_mask.get(str(action[0]), sub_action_mask.get(action[0], default))
    else:
        try:
            mask = sub_action_mask[action[0]]
        except (TypeError, IndexError):
            mask = default
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    if isinstance(mask, tuple):
        mask = list(mask)
    if not isinstance(mask, list) or len(mask) != len(Config.LABEL_SIZE_LIST):
        return default
    return [1 if int(value) > 0 else 0 for value in mask]


class FrameCollector:
    def __init__(self, num_agents):
        self._data_shapes = Config.data_shapes
        self._LSTM_FRAME = Config.LSTM_TIME_STEPS
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA
        self.reset(num_agents)

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]

    def save_frame(self, rl_data_info, agent_id):
        reward = self._clip_reward(rl_data_info.reward)
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = rl_data_info.value
            last_rl_data_info.reward = reward

        rl_data_info.reward = 0
        self.rl_data_map[agent_id][rl_data_info.frame_no] = rl_data_info

    def save_last_frame(self, reward, agent_id):
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = 0
            last_rl_data_info.reward = reward

    def sample_process(self):
        self._calc_reward()
        self._format_data()
        return self.m_replay_buffer

    def _calc_reward(self):
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae = 0.0
            for key in reversed_keys:
                rl_info = self.rl_data_map[i][key]
                delta = -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                gae = gae * self.gamma * self.lamda + delta
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value

    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)])
        idx, s_idx = 0, 0
        sample[-sample_lstm.shape[0] :] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx : s_idx + split_shape[0]] = sample_batch[:, idx : idx + one_shape].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]
        return sample.astype(np.float32)

    def _format_data(self):
        sample_one_size = np.sum(self._data_shapes[:-2]) // self._LSTM_FRAME
        sample_lstm_size = np.sum(self._data_shapes[-2:])
        sample_batch = np.zeros([self._LSTM_FRAME, sample_one_size])

        for i in range(self.num_agents):
            sample_lstm = np.zeros([sample_lstm_size])
            cnt = 0
            for key in self.rl_data_map[i]:
                rl_info = self.rl_data_map[i][key]
                idx = 0

                dlen = rl_info.feature.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.feature
                idx += dlen

                dlen = rl_info.legal_action.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.legal_action
                idx += dlen

                sample_batch[cnt, idx] = rl_info.reward_sum
                idx += 1
                sample_batch[cnt, idx] = rl_info.advantage
                idx += 1

                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.action
                idx += dlen

                for p in rl_info.prob:
                    dlen = len(p)
                    sample_batch[cnt, idx : idx + dlen] = p
                    idx += dlen

                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.sub_action
                idx += dlen

                sample_batch[cnt, idx] = rl_info.is_train
                idx += 1

                assert idx == sample_one_size, "Sample check failed, {}/{}".format(idx, sample_one_size)

                cnt += 1
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    sample_array = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                    self.m_replay_buffer[i].append(SampleData(sample=sample_array))
                    sample_lstm = rl_info.lstm_info

    def _clip_reward(self, reward, max=100, min=-100):
        if reward > max:
            return max
        if reward < min:
            return min
        return reward

    def __len__(self):
        return max([len(agent_samples) for agent_samples in self.rl_data_map])
