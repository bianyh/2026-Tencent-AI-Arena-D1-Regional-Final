#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_diy.conf.conf import GameConfig


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    TOWER_SUB_TYPE = 21

    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        # Get both defense towers
        # 获取双方防御塔
        main_tower, enemy_tower = None, None
        enemy_soldiers = []
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == self.TOWER_SUB_TYPE:
                    main_tower = organ
            else:
                if organ_subtype == self.TOWER_SUB_TYPE:
                    enemy_tower = organ
                else:
                    enemy_soldiers.append(organ)

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Tower health points
            # 塔血量
            if reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = self._safe_rate(main_tower, "hp", "max_hp")
            elif reward_name == "hero_hp_point":
                reward_struct.cur_frame_value = self._safe_rate(main_hero, "hp", "max_hp")
            elif reward_name == "money":
                reward_struct.cur_frame_value = self._safe_value(main_hero, "money_cnt", self._safe_value(main_hero, "money"))
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self._safe_value(main_hero, "exp")
            elif reward_name == "kill":
                reward_struct.cur_frame_value = self._safe_value(main_hero, "kill_cnt")
            elif reward_name == "death":
                reward_struct.cur_frame_value = self._safe_value(main_hero, "dead_cnt")
            elif reward_name == "hurt_to_hero":
                reward_struct.cur_frame_value = self._safe_value(main_hero, "total_hurt_to_hero")
            elif reward_name == "hurt_by_hero":
                reward_struct.cur_frame_value = self._safe_value(main_hero, "total_be_hurt_by_hero")
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = self._estimate_last_hit_value(frame_data, camp, enemy_soldiers)
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "unsafe_forward":
                reward_struct.cur_frame_value = self.calculate_unsafe_forward(main_hero, main_tower, enemy_tower, enemy_hero)

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["location"]["x"],
            main_hero["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if main_hero["hp"] / main_hero["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    def calculate_unsafe_forward(self, main_hero, main_tower, enemy_tower, enemy_hero):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0
        hp_rate = self._safe_rate(main_hero, "hp", "max_hp")
        if hp_rate >= 0.35:
            return 0
        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        total_dist = max(math.dist(main_tower_pos, enemy_tower_pos), 1.0)
        progress = 1.0 - min(math.dist(hero_pos, enemy_tower_pos) / total_dist, 1.0)
        enemy_near = 0
        if enemy_hero is not None and enemy_hero["hp"] > 0:
            enemy_pos = (enemy_hero["location"]["x"], enemy_hero["location"]["z"])
            enemy_near = 1 if math.dist(hero_pos, enemy_pos) < 8000 else 0
        return progress * (1.0 + enemy_near)

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "unsafe_forward":
                reward_struct.value = -self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name in ["death", "hurt_by_hero"]:
                reward_struct.value = -(
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_main_calc_frame_map[reward_name].last_frame_value
                )
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum

    def _safe_rate(self, obj, value_key, max_key):
        if obj is None:
            return 0.0
        max_value = obj.get(max_key, 0)
        if max_value <= 0:
            return 0.0
        return obj.get(value_key, 0) / max_value

    def _safe_value(self, obj, key, default=0.0):
        if obj is None:
            return default
        return obj.get(key, default)

    def _estimate_last_hit_value(self, frame_data, camp, enemy_soldiers):
        frame_action = frame_data.get("frame_action", {})
        dead_actions = frame_action.get("dead_action", []) if isinstance(frame_action, dict) else []
        if dead_actions:
            last_hit_count = 0
            for dead_action in dead_actions:
                death = dead_action.get("death", {})
                killer = dead_action.get("killer", {})
                if death.get("camp") != camp and killer.get("camp") == camp and death.get("sub_type") != self.TOWER_SUB_TYPE:
                    last_hit_count += 1
            return last_hit_count
        return sum(1 for soldier in enemy_soldiers if soldier.get("hp", 1) <= 0)

