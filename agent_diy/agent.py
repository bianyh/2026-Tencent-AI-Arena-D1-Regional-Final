#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import random

import numpy as np
import torch
from kaiwudrl.interface.agent import BaseAgent
from torch.optim.lr_scheduler import LambdaLR

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Args, Config
from agent_diy.feature.action_control import ActionController
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.obs_builder import ObsBuilder
from agent_diy.feature.reward_process import GameRewardManager
from agent_diy.feature.state_info import Info
from agent_diy.model.model import Model
from agent_diy.utils.hero_batch import HeroBatchRearrange

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


SUMMONER_SKILL_IDS = Args.SUMMONER_SKILL_IDS


class Agent(BaseAgent):
    _init_config_counter = 0
    _summoner_skill_counters = {}

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None

        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.info = Info()
        self.obs_builder = ObsBuilder(logger=logger)
        self.action_controller = ActionController()
        self.hero_rearrange = HeroBatchRearrange()

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)
        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        if self.lr <= 0:
            return 1.0
        if step > self.target_step:
            return self.target_lr / self.lr
        return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        my_heroes = config_data.get("my_heroes", [])
        opponent_heroes = config_data.get("opponent_heroes", [])
        Agent._init_config_counter += 1
        select_skills = {}
        for idx, hero_id in enumerate(my_heroes):
            key = (hero_id, tuple(opponent_heroes), idx)
            counter = Agent._summoner_skill_counters.get(key, 0)
            select_skills[hero_id] = SUMMONER_SKILL_IDS[counter % len(SUMMONER_SKILL_IDS)]
            Agent._summoner_skill_counters[key] = counter + 1
        return select_skills

    def reset(self, observation):
        self.hero_camp = observation["camp"]
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id)
        self.info.reset()
        self.obs_builder.reset()

    def _hero_index_from_obs_data(self, obs_data):
        return int(round(float(obs_data.feature[0])))

    def _model_inference(self, list_obs_data):
        hero_idx = [self._hero_index_from_obs_data(obs_data) for obs_data in list_obs_data]
        self.hero_rearrange.update(hero_idx)
        list_obs_data_reordered = self.hero_rearrange.forward(list_obs_data)
        hero_split_info = (self.hero_rearrange.active_head_indices, self.hero_rearrange.split_nums)

        feature = [obs_data.feature for obs_data in list_obs_data_reordered]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data_reordered]
        logit_bias = [obs_data.logit_bias for obs_data in list_obs_data_reordered]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data_reordered]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data_reordered]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32).to(self.device) for nparr in input_list]
        for idx, data in enumerate(torch_inputs):
            torch_inputs[idx] = data.reshape(-1).float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)
        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state, hero_split_info]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = [output.detach().cpu().numpy() for output in output_list]
        logits, value, new_lstm_cell, new_lstm_hidden = np_output[:4]
        logits = logits + np.array(logit_bias, dtype=np.float32)
        new_lstm_cell = new_lstm_cell.squeeze(axis=0)
        new_lstm_hidden = new_lstm_hidden.squeeze(axis=0)

        list_act_data = []
        for idx in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[idx], legal_action[idx])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value[idx : idx + 1],
                    lstm_cell=new_lstm_cell[idx],
                    lstm_hidden=new_lstm_hidden[idx],
                )
            )
        return self.hero_rearrange.inverse(list_act_data)

    def predict(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, True)

    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, False)

    def observation_process(self, observation):
        self.info.update(observation)
        feature = self.obs_builder.build_observation(self.info)
        legal_action = self.action_controller.refine_legal_action(observation["legal_action"], self.info)
        logit_bias = self.action_controller.build_logit_bias(self.info)
        return ObsData(
            feature=feature,
            legal_action=legal_action,
            logit_bias=logit_bias,
            lstm_cell=self.lstm_cell,
            lstm_hidden=self.lstm_hidden,
        )

    def action_process(self, observation, act_data, is_stochastic):
        action = act_data.action if is_stochastic else act_data.d_action
        action = self._normalize_env_action(action)
        if not is_stochastic:
            action = self.action_controller.fallback_action(action, self.info)
            act_data.d_action = action
        self.action_controller.record_executed_action(action, self.info)
        if self.reward_manager is not None:
            self.reward_manager.set_last_action(action)
        return action

    def _normalize_env_action(self, action):
        if action is None:
            return [0, 15, 15, 15, 15, 0]
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, tuple):
            action = list(action)
        while isinstance(action, list) and len(action) == 1 and isinstance(action[0], (list, tuple, np.ndarray)):
            action = action[0].tolist() if isinstance(action[0], np.ndarray) else list(action[0])
        if not isinstance(action, list) or len(action) != len(self.label_size_list):
            return [0, 15, 15, 15, 15, 0]
        fixed = []
        for idx, value in enumerate(action):
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = 0
            fixed.append(max(0, min(value, self.label_size_list[idx] - 1)))
        return fixed

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.cur_model_name = model_file_path
        self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits, legal_action):
        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])

        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            action_list.append(self._legal_sample(probs, use_max=False))
            d_action_list.append(self._legal_sample(probs, use_max=True))

        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [self.legal_action_size[0], self.legal_action_size[-1] // self.legal_action_size[0]],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]].reshape([self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        action_list.append(self._legal_sample(probs, use_max=False))

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]].reshape([self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)
        d_action_list.append(self._legal_sample(probs, use_max=True))
        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        input_hidden = np.nan_to_num(np.array(input_hidden, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        legal_action = np.array(legal_action, dtype=np.float64)
        if legal_action.sum() <= 0:
            legal_action = np.ones_like(legal_action)
        else:
            legal_action = (legal_action > 0).astype(np.float64)
        lsm_const_w, lsm_const_e = 1e20, 1e-5
        tmp = input_hidden - lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -lsm_const_w, 1)
        tmp = (np.exp(tmp) + lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0) * legal_action
        prob_sum = probs.sum(dtype=np.float64)
        if prob_sum <= 0:
            probs = legal_action / legal_action.sum(dtype=np.float64)
        else:
            probs = probs / prob_sum
        return probs.astype(np.float64)

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        probs = np.nan_to_num(np.array(probs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.maximum(probs, 0.0)
        total = probs.sum(dtype=np.float64)
        if total <= 0:
            probs = np.ones_like(probs, dtype=np.float64) / len(probs)
        else:
            probs = probs / total
        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))
