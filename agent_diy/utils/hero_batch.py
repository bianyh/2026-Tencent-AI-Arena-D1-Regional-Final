#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from agent_diy.conf.conf import Args


class HeroBatchRearrange:
    def __init__(self):
        self.order = []
        self.inverse_order = []
        self.active_head_indices = []
        self.split_nums = []

    def update(self, hero_indices):
        groups = []
        for head_idx in range(Args.HERO_HEAD_NUM):
            indices = [idx for idx, hero_idx in enumerate(hero_indices) if int(hero_idx) == head_idx]
            if indices:
                groups.append((head_idx, indices))

        self.active_head_indices = [head_idx for head_idx, _ in groups]
        self.split_nums = [len(indices) for _, indices in groups]
        self.order = [idx for _, indices in groups for idx in indices]
        self.inverse_order = [0] * len(self.order)
        for new_idx, old_idx in enumerate(self.order):
            self.inverse_order[old_idx] = new_idx

    def forward(self, values):
        return [values[idx] for idx in self.order]

    def inverse(self, values):
        return [values[self.inverse_order[idx]] for idx in range(len(values))]
