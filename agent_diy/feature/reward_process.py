#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math

from agent_diy.conf.conf import GameConfig
from agent_diy.feature.state_info import ACTOR_SUB_SOLDIER, ACTOR_SUB_TOWER, ACTOR_TYPE_ORGAN


class RewardStruct:
    def __init__(self, weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = weight


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_runtime_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}
        self.init_max_exp_of_each_hero()

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        values = {
            1: 160,
            2: 298,
            3: 446,
            4: 524,
            5: 613,
            6: 713,
            7: 825,
            8: 950,
            9: 1088,
            10: 1240,
            11: 1406,
            12: 1585,
            13: 1778,
            14: 1984,
        }
        self.m_each_level_max_exp.update(values)

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        return self.m_reward_value

    def set_cur_calc_frame_vec(self, calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []):
            if hero.get("camp") == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        if main_hero is None:
            return

        main_tower, enemy_tower = None, None
        for npc in frame_data.get("npc_states", []):
            if npc.get("actor_type") != ACTOR_TYPE_ORGAN or npc.get("sub_type") != ACTOR_SUB_TOWER:
                continue
            if npc.get("camp") == camp:
                main_tower = npc
            else:
                enemy_tower = npc
        if main_tower is None:
            main_tower = {"hp": 0, "max_hp": 1, "location": {"x": 0, "z": 0}}
        if enemy_tower is None:
            enemy_tower = {"hp": 0, "max_hp": 1, "location": {"x": 0, "z": 0}}

        hp = main_hero.get("hp", 0)
        hp_max = max(main_hero.get("max_hp", 1), 1)
        ep = main_hero.get("ep", 0)
        ep_max = max(main_hero.get("max_ep", 1), 1)

        for reward_name, reward_struct in calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero.get("money_cnt", main_hero.get("money", 0))
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(max(hp / hp_max, 0.0)))
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = ep / ep_max if hp > 0 else 0.0
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero.get("kill_cnt", 0)
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero.get("dead_cnt", 0)
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = main_tower.get("hp", 0) / max(main_tower.get("max_hp", 1), 1)
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = self.calculate_last_hit(frame_data, main_hero, enemy_hero)
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)

    def calculate_last_hit(self, frame_data, main_hero, enemy_hero):
        if enemy_hero is None:
            return 0.0
        value = 0.0
        dead_actions = (frame_data.get("frame_action") or {}).get("dead_action", [])
        for dead_action in dead_actions:
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            if death.get("sub_type") != ACTOR_SUB_SOLDIER:
                continue
            if killer.get("runtime_id") == main_hero.get("runtime_id"):
                value += 1.0
            elif killer.get("runtime_id") == enemy_hero.get("runtime_id"):
                value -= 1.0
        return value

    def calculate_exp_sum(self, hero):
        exp_sum = 0.0
        for level in range(1, hero.get("level", 1)):
            exp_sum += self.m_each_level_max_exp.get(level, 0)
        exp_sum += hero.get("exp", 0)
        return exp_sum

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        hero_hp_rate = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1)
        if hero_hp_rate <= 0.99:
            return 0.0
        hero_pos = (main_hero.get("location", {}).get("x", 0), main_hero.get("location", {}).get("z", 0))
        main_tower_pos = (main_tower.get("location", {}).get("x", 0), main_tower.get("location", {}).get("z", 0))
        enemy_tower_pos = (enemy_tower.get("location", {}).get("x", 0), enemy_tower.get("location", {}).get("z", 0))
        dist_hero_enemy = math.dist(hero_pos, enemy_tower_pos)
        dist_main_enemy = max(math.dist(main_tower_pos, enemy_tower_pos), 1.0)
        if dist_hero_enemy > dist_main_enemy:
            return (dist_main_enemy - dist_hero_enemy) / dist_main_enemy
        return 0.0

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data.get("hero_states", []):
            if hero.get("runtime_id") == self.main_hero_runtime_id:
                main_camp = hero.get("camp")
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero.get("camp")
        if main_camp == -1:
            return
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                main_last = self.m_main_calc_frame_map[reward_name].last_frame_value
                enemy_last = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                main_cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                enemy_cur = self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.cur_frame_value = main_cur - enemy_cur
                reward_struct.last_frame_value = main_last - enemy_last
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = (
                    reward_struct.cur_frame_value - reward_struct.last_frame_value
                    if reward_struct.last_frame_value > 0
                    else 0
                )
            elif reward_name == "exp":
                main_cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                enemy_cur = self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                main_last = self.m_main_calc_frame_map[reward_name].last_frame_value
                enemy_last = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.cur_frame_value = main_cur - enemy_cur
                reward_struct.last_frame_value = main_last - enemy_last
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                if GameConfig.REMOVE_FORWARD_AFTER is not None and frame_no > GameConfig.REMOVE_FORWARD_AFTER:
                    reward_struct.value = 0.0
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                main_cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                enemy_cur = self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                main_last = self.m_main_calc_frame_map[reward_name].last_frame_value
                enemy_last = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.cur_frame_value = main_cur - enemy_cur
                reward_struct.last_frame_value = main_last - enemy_last
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            time_scale = 1.0
            if self.time_scale_arg > 0 and reward_name not in GameConfig.REWARD_WITHOUT_TIME_SCALE:
                time_scale = math.pow(0.6, frame_no / self.time_scale_arg)

            reward_dict[reward_name + "_origin"] = reward_struct.value
            reward_dict[reward_name + "_weight"] = reward_struct.value * reward_struct.weight * time_scale
            reward_sum += reward_dict[reward_name + "_weight"]

        reward_dict["reward_sum"] = reward_sum
