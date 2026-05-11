#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

from agent_diy.conf.conf import Args, Config, DimConfig


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.dim_public = Config.DIM_PUBLIC
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.var_beta = self.m_var_beta
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.single_hero_feature_dim = int(DimConfig.DIM_OF_HERO_EMY[0])
        self.single_soldier_feature_dim = int(DimConfig.DIM_OF_SOLDIER_1_4[0])
        self.single_organ_feature_dim = int(DimConfig.DIM_OF_ORGAN_1[0])
        self.single_river_crab_feature_dim = int(DimConfig.DIM_OF_RIVER_CRAB[0])
        self.single_bullet_feature_dim = int(DimConfig.DIM_OF_BULLET_1_9[0])

        self.all_hero_feature_dim = int(np.sum(DimConfig.DIM_OF_HERO_FRD)) + int(np.sum(DimConfig.DIM_OF_HERO_EMY))
        self.all_soldier_feature_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4)) + int(
            np.sum(DimConfig.DIM_OF_SOLDIER_5_8)
        )
        self.all_organ_feature_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1)) + int(np.sum(DimConfig.DIM_OF_ORGAN_2))
        self.all_bullet_feature_dim = int(np.sum(DimConfig.DIM_OF_BULLET_1_9)) + int(np.sum(DimConfig.DIM_OF_BULLET_10))

        self.position_delta_dim = 64 - Args.DIM_DISTANCE
        self.position_mlp = MLP([Args.DIM_DISTANCE, 128, 64], "position_mlp")
        self.unit_delta_dim = 64 + 32 - Args.DIM_UNIT
        self.unit_no_pos_mlp = MLP([Args.DIM_UNIT - Args.DIM_DISTANCE, 64, 32], "unit_no_pos_mlp")

        fc_hero_dim_list = [self.single_hero_feature_dim + self.unit_delta_dim, 512, 256, 128]
        self.hero_mlp = MLP(fc_hero_dim_list[:-1], "hero_mlp", non_linearity_last=True)
        self.hero_frd_fc = make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1])
        self.hero_emy_fc = make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1])

        fc_soldier_dim_list = [self.single_soldier_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.soldier_mlp = MLP(fc_soldier_dim_list[:-1], "soldier_mlp", non_linearity_last=True)
        self.soldier_frd_fc = make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1])
        self.soldier_emy_fc = make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1])

        fc_river_crab_list = [self.single_river_crab_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.river_crab_mlp = MLP(fc_river_crab_list, "river_crab_mlp")

        fc_organ_dim_list = [self.single_organ_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.organ_mlp = MLP(fc_organ_dim_list[:-1], "organ_mlp", non_linearity_last=True)
        self.organ_frd_fc = make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1])
        self.organ_emy_fc = make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1])

        fc_bullet_list = [self.single_bullet_feature_dim + self.position_delta_dim, 64, 64, 32]
        self.bullet_mlp = MLP(fc_bullet_list[:-1], "bullet_mlp", non_linearity_last=True)
        self.bullet_hero_fc = make_fc_layer(fc_bullet_list[-2], fc_bullet_list[-1])
        self.bullet_organ_fc = make_fc_layer(fc_bullet_list[-2], fc_bullet_list[-1])

        concat_dim = 128 * 2 + 32 * 2 + 32 + 32 * 2 + 32 * 2
        self.concat_mlp = MLP([concat_dim, self.lstm_unit_size], "concat_mlp", non_linearity_last=True)
        self.concate_mlp_other = MLP([concat_dim, 512, self.lstm_unit_size], "concat_other_mlp")
        self.lstm_and_linear_mlp = MLP([self.lstm_unit_size * 2, self.dim_public], "lstm_and_linear_mlp", non_linearity_last=True)

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT,
            bidirectional=False,
        )

        self.label_mlps = nn.ModuleList(
            [
                ModuleDict(
                    {
                        f"hero_label{label_index}_mlp": MLP(
                            [self.dim_public, self.label_size_list[label_index]],
                            f"hero{head_idx}_label{label_index}_mlp",
                        )
                        for label_index in range(len(self.label_size_list) - 1)
                    }
                )
                for head_idx in range(Args.HERO_HEAD_NUM)
            ]
        )
        self.lstm_tar_embed_mlps = nn.ModuleList(
            [make_fc_layer(self.dim_public, self.target_embed_dim) for _ in range(Args.HERO_HEAD_NUM)]
        )
        self.target_embed_mlps = nn.ModuleList(
            [make_fc_layer(32, self.target_embed_dim, use_bias=False) for _ in range(Args.HERO_HEAD_NUM)]
        )
        self.value_mlps = nn.ModuleList([MLP([self.dim_public, 64, 1], f"hero{idx}_value_mlp") for idx in range(Args.HERO_HEAD_NUM)])

    def process_sub_feature(self, x, mlp, is_unit):
        parts = [x]
        dim_suffix = Args.DIM_DISTANCE
        if is_unit:
            unit_no_pos = self.unit_no_pos_mlp(x[..., -Args.DIM_UNIT : -Args.DIM_DISTANCE])
            parts.append(unit_no_pos)
            dim_suffix = Args.DIM_UNIT
        pos = self.position_mlp(x[..., -Args.DIM_DISTANCE :])
        parts.append(pos)
        parts[0] = x[..., :-dim_suffix]
        return mlp(torch.cat(parts, dim=-1))

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init, hero_split_info = data_list
        active_head_indices, hero_split_nums = hero_split_info

        feature_vec_split_list = feature_vec.split(
            [
                self.all_hero_feature_dim,
                self.all_soldier_feature_dim,
                self.single_river_crab_feature_dim,
                self.all_organ_feature_dim,
                self.all_bullet_feature_dim,
            ],
            dim=1,
        )
        hero_vec_list = feature_vec_split_list[0].split(
            [int(np.sum(DimConfig.DIM_OF_HERO_FRD)), int(np.sum(DimConfig.DIM_OF_HERO_EMY))], dim=1
        )
        soldier_vec_list = feature_vec_split_list[1].split(
            [int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4)), int(np.sum(DimConfig.DIM_OF_SOLDIER_5_8))], dim=1
        )
        river_crab_tensor = feature_vec_split_list[2]
        organ_vec_list = feature_vec_split_list[3].split(
            [int(np.sum(DimConfig.DIM_OF_ORGAN_1)), int(np.sum(DimConfig.DIM_OF_ORGAN_2))], dim=1
        )
        bullet_vec_list = feature_vec_split_list[4].split(
            [int(np.sum(DimConfig.DIM_OF_BULLET_1_9)), int(np.sum(DimConfig.DIM_OF_BULLET_10))], dim=1
        )

        hero_frd = hero_vec_list[0].split(DimConfig.DIM_OF_HERO_FRD, dim=1)
        hero_emy = hero_vec_list[1].split(DimConfig.DIM_OF_HERO_EMY, dim=1)
        soldier_frd = soldier_vec_list[0].split(DimConfig.DIM_OF_SOLDIER_1_4, dim=1)
        soldier_emy = soldier_vec_list[1].split(DimConfig.DIM_OF_SOLDIER_5_8, dim=1)
        organ_frd = organ_vec_list[0].split(DimConfig.DIM_OF_ORGAN_1, dim=1)
        organ_emy = organ_vec_list[1].split(DimConfig.DIM_OF_ORGAN_2, dim=1)
        bullet_hero = bullet_vec_list[0].split(DimConfig.DIM_OF_BULLET_1_9, dim=1)
        bullet_organ = bullet_vec_list[1].split(DimConfig.DIM_OF_BULLET_10, dim=1)

        tar_embed_list = []

        hero_emy_results = []
        for hero in hero_emy:
            out = self.hero_emy_fc(self.process_sub_feature(hero, self.hero_mlp, True))
            _, target_embed = out.split([96, 32], dim=1)
            tar_embed_list.append(target_embed)
            hero_emy_results.append(out)
        hero_emy_concat = torch.cat(hero_emy_results, dim=1)

        hero_frd_results = []
        for hero in hero_frd:
            out = self.hero_frd_fc(self.process_sub_feature(hero, self.hero_mlp, True))
            _, target_embed = out.split([96, 32], dim=1)
            tar_embed_list.append(target_embed)
            hero_frd_results.append(out)
        hero_frd_concat = torch.cat(hero_frd_results, dim=1)

        soldier_frd_results = []
        for soldier in soldier_frd:
            soldier_frd_results.append(self.soldier_frd_fc(self.process_sub_feature(soldier, self.soldier_mlp, True)))
        soldier_frd_concat = torch.cat(soldier_frd_results, dim=1).reshape(-1, Args.SOLDIER_MAX_NUM, 32).max(dim=1)[0]

        soldier_emy_results = []
        for soldier in soldier_emy:
            out = self.soldier_emy_fc(self.process_sub_feature(soldier, self.soldier_mlp, True))
            soldier_emy_results.append(out)
            tar_embed_list.append(out)
        soldier_emy_concat = torch.cat(soldier_emy_results, dim=1).reshape(-1, Args.SOLDIER_MAX_NUM, 32).max(dim=1)[0]

        river_crab_result = self.process_sub_feature(river_crab_tensor, self.river_crab_mlp, True)

        organ_frd_results = []
        for organ in organ_frd:
            organ_frd_results.append(self.organ_frd_fc(self.process_sub_feature(organ, self.organ_mlp, True)))
        organ_frd_concat = torch.cat(organ_frd_results, dim=1)

        organ_emy_results = []
        for organ in organ_emy:
            out = self.organ_emy_fc(self.process_sub_feature(organ, self.organ_mlp, True))
            organ_emy_results.append(out)
            tar_embed_list.append(out)
        organ_emy_concat = torch.cat(organ_emy_results, dim=1)

        bullet_hero_results = []
        for bullet in bullet_hero:
            bullet_hero_results.append(self.bullet_hero_fc(self.process_sub_feature(bullet, self.bullet_mlp, False)))
        bullet_hero_concat = torch.cat(bullet_hero_results, dim=1).reshape(-1, Args.BULLET_MAX_NUM - 1, 32).max(dim=1)[0]

        bullet_organ_results = []
        for bullet in bullet_organ:
            bullet_organ_results.append(self.bullet_organ_fc(self.process_sub_feature(bullet, self.bullet_mlp, False)))
        bullet_organ_concat = torch.cat(bullet_organ_results, dim=1)

        target_pad = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_list.insert(0, target_pad)
        tar_embed_list.append(river_crab_result)
        tar_embedding = torch.stack(tar_embed_list, dim=1)

        concat_result = torch.cat(
            [
                hero_frd_concat,
                hero_emy_concat,
                soldier_frd_concat,
                soldier_emy_concat,
                river_crab_result,
                organ_frd_concat,
                organ_emy_concat,
                bullet_hero_concat,
                bullet_organ_concat,
            ],
            dim=1,
        )

        fc_public_result = self.concat_mlp(concat_result)
        reshape_fc_public_result = fc_public_result.reshape(-1, self.lstm_time_steps, self.lstm_unit_size)
        lstm_initial_state_in = [lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0)]
        lstm_outputs, state = self.lstm(reshape_fc_public_result, lstm_initial_state_in)
        lstm_hidden_output, lstm_cell_output = state[0], state[1]
        lstm_outputs = lstm_outputs.reshape(-1, self.lstm_unit_size)

        public_mlp_result = self.concate_mlp_other(concat_result)
        public_result = self.lstm_and_linear_mlp(torch.cat([lstm_outputs, public_mlp_result], dim=-1))

        split_frame_nums = [num * self.lstm_time_steps for num in hero_split_nums]
        public_by_head = public_result.split(split_frame_nums, dim=0)
        target_by_head = tar_embedding.split(split_frame_nums, dim=0)

        result_list = [[] for _ in range(len(self.label_size_list) + 1)]
        for local_idx, head_idx in enumerate(active_head_indices):
            label_mlp = self.label_mlps[head_idx]
            head_public = public_by_head[local_idx]
            head_target = target_by_head[local_idx]
            for label_index in range(len(self.label_size_list) - 1):
                result_list[label_index].append(label_mlp[f"hero_label{label_index}_mlp"](head_public))

            lstm_target = self.lstm_tar_embed_mlps[head_idx](head_public).reshape(-1, self.target_embed_dim, 1)
            query = self.target_embed_mlps[head_idx](head_target)
            query = nn.functional.softmax(query, dim=-1)
            target_logits = torch.matmul(query, lstm_target).reshape(-1, self.label_size_list[-1])
            result_list[-2].append(target_logits)
            result_list[-1].append(self.value_mlps[head_idx](head_public))

        for idx in range(len(result_list)):
            result_list[idx] = torch.cat(result_list[idx], dim=0)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value, lstm_cell_output, lstm_hidden_output]
        return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1, self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index]
            )

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = [label.squeeze(dim=1) for label in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]
        frame_is_train = usq_is_train.squeeze(dim=1)
        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - value_result_squeezed), dim=0)

        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0, device=value_result.device)
        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                continue
            boundary = torch.pow(torch.tensor(10.0, device=value_result.device), torch.tensor(20.0, device=value_result.device))
            one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
            legal_mask = legal_action_flag_list[task_index].to(value_result.device)
            label_logits = label_result[task_index]
            legal_action_flag_list_max_mask = (1 - legal_mask) * boundary
            label_logits_subtract_max = torch.clamp(
                label_logits - torch.max(label_logits - legal_action_flag_list_max_mask, dim=1, keepdim=True).values,
                -boundary,
                1,
            )
            label_exp_logits = legal_mask * torch.exp(label_logits_subtract_max) + self.min_policy
            label_probability = label_exp_logits / label_exp_logits.sum(1, keepdim=True)
            label_probability_list.append(label_probability)

            policy_p = (one_hot_actions.to(value_result.device) * label_probability).sum(1)
            policy_log_p = torch.log(policy_p + epsilon)
            old_policy_p = (
                one_hot_actions.to(value_result.device) * old_label_probability_list[task_index].to(value_result.device) + epsilon
            ).sum(1)
            old_policy_log_p = torch.log(old_policy_p)
            ratio = torch.exp(policy_log_p - old_policy_log_p).clamp(0.0, 3.0)
            surr1 = ratio * advantage.to(value_result.device)
            surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage.to(value_result.device)
            weight = weight_list[task_index].float().to(value_result.device) * frame_is_train.to(value_result.device)
            temp_policy_loss = -torch.sum(torch.minimum(surr1, surr2) * weight) / torch.maximum(
                torch.sum(weight), torch.tensor(1.0, device=value_result.device)
            )
            self.policy_cost = self.policy_cost + temp_policy_loss

        entropy_loss_list = []
        prob_idx = 0
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                prob = label_probability_list[prob_idx]
                legal_mask = legal_action_flag_list[task_index].to(value_result.device)
                temp_entropy_loss = -torch.sum(prob * legal_mask * torch.log(prob + epsilon), dim=1)
                weight = weight_list[task_index].float().to(value_result.device) * frame_is_train.to(value_result.device)
                temp_entropy_loss = -torch.sum(temp_entropy_loss * weight) / torch.maximum(
                    torch.sum(weight), torch.tensor(1.0, device=value_result.device)
                )
                entropy_loss_list.append(temp_entropy_loss)
                prob_idx += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0, device=value_result.device))

        self.entropy_cost = torch.tensor(0.0, device=value_result.device)
        for entropy in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features, out_features, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(self, fc_feat_dim_list: List[int], name: str, non_linearity=nn.ReLU, non_linearity_last=False):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for idx in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[idx], fc_feat_dim_list[idx + 1])
            self.fc_layers.add_module(f"{name}_fc{idx + 1}", fc_layer)
            if idx + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{idx + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
