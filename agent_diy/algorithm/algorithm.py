#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import time

import numpy as np
import torch

from agent_diy.conf.conf import Config
from agent_diy.utils.hero_batch import HeroBatchRearrange


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0
        self.logger = logger
        self.monitor = monitor
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.last_report_monitor_time = 0
        self.hero_rearrange = HeroBatchRearrange()

    def _get_sample_array(self, sample_data):
        return sample_data.sample

    def _get_hero_idx(self, sample_data):
        sample = self._get_sample_array(sample_data)
        value = sample[0]
        if hasattr(value, "item"):
            value = value.item()
        return int(round(float(value)))

    def learn(self, list_sample_data):
        if not list_sample_data:
            return

        hero_idx = [self._get_hero_idx(sample) for sample in list_sample_data]
        self.hero_rearrange.update(hero_idx)
        list_sample_data = self.hero_rearrange.forward(list_sample_data)
        hero_split_info = (self.hero_rearrange.active_head_indices, self.hero_rearrange.split_nums)

        _input_datas = torch.stack([self._get_sample_array(sample) for sample in list_sample_data]).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for idx, data in enumerate(data_list):
            data_list[idx] = data.reshape(-1).float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, _legal_action = seri_vec.split(
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state, hero_split_info]

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1
        self.scheduler.step(self.train_step)

        info_values = []
        for info in info_list:
            if isinstance(info, list):
                info_values.append([item.item() for item in info])
            else:
                info_values.append(info.item())

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = info_values
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now
