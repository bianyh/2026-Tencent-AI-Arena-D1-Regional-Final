#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math


class SituationProcess:
    FEATURE_DIM = 32
    TOWER_SUB_TYPE = 21
    MAP_COORD_ABS = 60000.0
    MAX_DISTANCE = 90000.0
    MAX_HP = 12000.0
    MAX_EP = 3000.0
    MAX_LEVEL = 15.0
    MAX_EXP = 2000.0
    MAX_MONEY = 20000.0
    MAX_ATTACK = 1500.0
    MAX_DEFENSE = 2000.0
    MAX_SPEED = 1500.0
    MAX_ATTACK_SPEED = 3000.0
    MAX_COOLDOWN = 60000.0
    MAX_SOLDIER_INCOME = 300.0

    def __init__(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"

    def reset(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"

    def process_vec_situation(self, frame_state):
        main_hero, enemy_hero = self._get_heroes(frame_state)
        main_tower, enemy_tower = self._get_towers(frame_state)
        soldiers = self._get_soldiers(frame_state)

        if main_hero is None:
            return [0.0] * self.FEATURE_DIM

        feature = []
        feature.extend(self._hero_status(main_hero))
        feature.extend(self._hero_status(enemy_hero))
        feature.extend(self._skill_status(main_hero))
        feature.extend(self._relative_status(main_hero, enemy_hero, main_tower, enemy_tower))
        feature.extend(self._soldier_status(main_hero, soldiers))

        if len(feature) < self.FEATURE_DIM:
            feature.extend([0.0] * (self.FEATURE_DIM - len(feature)))
        elif len(feature) > self.FEATURE_DIM:
            feature = feature[: self.FEATURE_DIM]
        return feature

    def _get_heroes(self, frame_state):
        main_hero, enemy_hero = None, None
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _get_towers(self, frame_state):
        main_tower, enemy_tower = None, None
        for npc in frame_state["npc_states"]:
            if npc["sub_type"] != self.TOWER_SUB_TYPE:
                continue
            if npc["camp"] == self.main_camp:
                main_tower = npc
            else:
                enemy_tower = npc
        return main_tower, enemy_tower

    def _get_soldiers(self, frame_state):
        return [npc for npc in frame_state["npc_states"] if npc["sub_type"] != self.TOWER_SUB_TYPE]

    def _hero_status(self, hero):
        if hero is None:
            return [0.0] * 8
        hp_rate = self._rate(hero.get("hp", 0), hero.get("max_hp", 0))
        ep_rate = self._rate(hero.get("ep", 0), hero.get("max_ep", 0))
        return [
            1.0 if hero.get("hp", 0) > 0 else 0.0,
            hp_rate,
            ep_rate,
            self._clip01(hero.get("level", 0) / self.MAX_LEVEL),
            self._clip01(hero.get("exp", 0) / self.MAX_EXP),
            self._clip01(hero.get("money", hero.get("money_cnt", 0)) / self.MAX_MONEY),
            self._clip01(hero.get("phy_atk", 0) / self.MAX_ATTACK),
            self._clip01(hero.get("phy_def", 0) / self.MAX_DEFENSE),
        ]

    def _skill_status(self, hero):
        slots = []
        if hero is not None:
            skill_state = hero.get("skill_state", {})
            slots = skill_state.get("slot_states", []) or []

        feature = []
        for slot in slots[:4]:
            feature.append(1.0 if slot.get("usable", False) else 0.0)
            cooldown_max = max(float(slot.get("cooldown_max", 0)), 1.0)
            cooldown = float(slot.get("cooldown", 0))
            feature.append(self._clip01(1.0 - cooldown / cooldown_max))
        while len(feature) < 8:
            feature.append(0.0)
        return feature[:8]

    def _relative_status(self, main_hero, enemy_hero, main_tower, enemy_tower):
        hero_pos = self._pos(main_hero)
        enemy_pos = self._pos(enemy_hero)
        main_tower_pos = self._pos(main_tower)
        enemy_tower_pos = self._pos(enemy_tower)

        hero_to_enemy = self._relative_pair(hero_pos, enemy_pos)
        hero_to_enemy_tower = self._relative_pair(hero_pos, enemy_tower_pos)
        hero_to_main_tower = self._relative_pair(hero_pos, main_tower_pos)
        enemy_to_enemy_tower = self._relative_pair(enemy_pos, enemy_tower_pos)

        if main_tower_pos is None or enemy_tower_pos is None or hero_pos is None:
            lane_progress = 0.0
        else:
            total = max(self._dist(main_tower_pos, enemy_tower_pos), 1.0)
            lane_progress = 1.0 - self._clip01(self._dist(hero_pos, enemy_tower_pos) / total)

        return [
            hero_to_enemy[0],
            hero_to_enemy[1],
            hero_to_enemy[2],
            hero_to_enemy_tower[2],
            hero_to_main_tower[2],
            enemy_to_enemy_tower[2],
            lane_progress,
            self._tower_risk(main_hero, enemy_tower),
        ]

    def _soldier_status(self, main_hero, soldiers):
        hero_pos = self._pos(main_hero)
        main_soldiers = [soldier for soldier in soldiers if soldier["camp"] == self.main_camp]
        enemy_soldiers = [soldier for soldier in soldiers if soldier["camp"] != self.main_camp]
        nearest_enemy = self._nearest(hero_pos, enemy_soldiers)
        nearest_main = self._nearest(hero_pos, main_soldiers)
        enemy_hp_rate = self._rate(nearest_enemy.get("hp", 0), nearest_enemy.get("max_hp", 0)) if nearest_enemy else 0.0
        main_hp_rate = self._rate(nearest_main.get("hp", 0), nearest_main.get("max_hp", 0)) if nearest_main else 0.0
        enemy_dist = self._relative_pair(hero_pos, self._pos(nearest_enemy))[2] if nearest_enemy else 1.0
        main_dist = self._relative_pair(hero_pos, self._pos(nearest_main))[2] if nearest_main else 1.0

        return [
            self._clip01(len(main_soldiers) / 8.0),
            self._clip01(len(enemy_soldiers) / 8.0),
            enemy_hp_rate,
            enemy_dist,
            main_hp_rate,
            main_dist,
            self._clip01(sum(s.get("kill_income", 0) for s in enemy_soldiers[:4]) / (4.0 * self.MAX_SOLDIER_INCOME)),
            self._clip01((len(main_soldiers) - len(enemy_soldiers) + 8.0) / 16.0),
        ]

    def _tower_risk(self, hero, tower):
        if hero is None or tower is None:
            return 0.0
        attack_range = max(float(tower.get("attack_range", 0)), 1.0)
        distance = self._dist(self._pos(hero), self._pos(tower))
        return 1.0 if distance <= attack_range else 0.0

    def _nearest(self, position, units):
        if position is None or not units:
            return None
        return min(units, key=lambda unit: self._dist(position, self._pos(unit)))

    def _relative_pair(self, pos_a, pos_b):
        if pos_a is None or pos_b is None:
            return 0.5, 0.5, 1.0
        dx = pos_b[0] - pos_a[0]
        dz = pos_b[1] - pos_a[1]
        if self.transform_camp2_to_camp1:
            dx = -dx
            dz = -dz
        return (
            self._clip01((dx + self.MAP_COORD_ABS) / (2.0 * self.MAP_COORD_ABS)),
            self._clip01((dz + self.MAP_COORD_ABS) / (2.0 * self.MAP_COORD_ABS)),
            self._clip01(math.sqrt(dx * dx + dz * dz) / self.MAX_DISTANCE),
        )

    def _pos(self, obj):
        if obj is None:
            return None
        location = obj.get("location", {})
        x = location.get("x", 100000)
        z = location.get("z", 100000)
        if x == 100000 or z == 100000:
            return None
        return float(x), float(z)

    def _dist(self, pos_a, pos_b):
        if pos_a is None or pos_b is None:
            return self.MAX_DISTANCE
        return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def _rate(self, value, max_value):
        max_value = float(max_value)
        if max_value <= 0:
            return 0.0
        return self._clip01(float(value) / max_value)

    def _clip01(self, value):
        return max(0.0, min(1.0, float(value)))
