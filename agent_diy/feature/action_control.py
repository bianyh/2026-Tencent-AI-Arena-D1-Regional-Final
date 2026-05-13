#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math

import numpy as np

from agent_diy.conf.conf import Args, Config, GameConfig
from agent_diy.feature.state_info import ACTOR_SUB_TOWER, ACTOR_TYPE_ORGAN, Info


BUTTON_MOVE = 2
BUTTON_ATTACK = 3
BUTTON_SKILL_1 = 4
BUTTON_SKILL_2 = 5
BUTTON_SKILL_3 = 6
BUTTON_HEAL = 7
BUTTON_SUMMONER = 8

TARGET_NONE = 0
TARGET_ENEMY_HERO = 1
TARGET_SELF = 2
TARGET_SOLDIER_BASE = 3
TARGET_TOWER = 7
TARGET_MONSTER = 8


class ActionController:
    def __init__(self):
        self.pending_attack_move = False

    def refine_legal_action(self, legal_action, info: Info):
        refined = np.array(legal_action, dtype=np.float32).copy()
        if refined.size != sum(Config.LEGAL_ACTION_SIZE_LIST):
            return refined

        buttons, move_x, move_z, skill_x, skill_z, target_mask = self.split_legal_action(refined)
        attack_targets = self.attackable_target_indices(info)
        if not attack_targets:
            buttons[BUTTON_ATTACK] = 0.0
        else:
            self.keep_targets(target_mask, BUTTON_ATTACK, attack_targets)

        for button in (BUTTON_SKILL_1, BUTTON_SKILL_2, BUTTON_SKILL_3):
            if buttons[button] > 0:
                targets = self.skill_target_indices(info, button)
                self.keep_targets(target_mask, button, targets)

        if buttons[BUTTON_HEAL] > 0:
            self.keep_targets(target_mask, BUTTON_HEAL, [TARGET_SELF, TARGET_NONE])
        if buttons[BUTTON_SUMMONER] > 0:
            self.keep_targets(target_mask, BUTTON_SUMMONER, self.summoner_target_indices(info))

        if buttons.sum() <= 0:
            buttons[1] = 1.0
            buttons[BUTTON_MOVE] = 1.0
        return self.join_legal_action([buttons, move_x, move_z, skill_x, skill_z, target_mask])

    def build_logit_bias(self, info: Info):
        bias = np.zeros(sum(Config.LABEL_SIZE_LIST), dtype=np.float32)
        button_bias = bias[: Config.LABEL_SIZE_LIST[0]]
        hero = info.hero_our
        hp_rate = hero.info.hp / max(hero.info.hp_max, 1)

        if self.pending_attack_move and not self.is_luban_enhanced_attack_state(info):
            button_bias[BUTTON_MOVE] += GameConfig.ATTACK_MOVE_LOGIT_BIAS
            button_bias[BUTTON_ATTACK] -= GameConfig.ATTACK_MOVE_LOGIT_BIAS * 0.4

        if hp_rate < GameConfig.LOW_HP_HEAL_RATE:
            button_bias[BUTTON_MOVE] += 0.4
            if hero.skill.recover.usable:
                button_bias[BUTTON_HEAL] += 1.0
            if hero.skill.summoner.usable and self.is_defensive_summoner(hero.skill.summoner.config_id, info):
                button_bias[BUTTON_SUMMONER] += 0.65
        else:
            if self.has_skill_target(info):
                for button, slot in (
                    (BUTTON_SKILL_1, hero.skill.first),
                    (BUTTON_SKILL_2, hero.skill.second),
                    (BUTTON_SKILL_3, hero.skill.third),
                ):
                    if slot.usable and slot.level > 0:
                        button_bias[button] += 0.25
            if self.attackable_target_indices(info):
                button_bias[BUTTON_ATTACK] += 0.15
        return bias

    def fallback_action(self, action, info: Info):
        action = list(action)
        self.pending_attack_move = False
        hero = info.hero_our
        hp_rate = hero.info.hp / max(hero.info.hp_max, 1)

        if hp_rate < GameConfig.CRITICAL_HP_RATE:
            if hero.skill.recover.usable:
                return self.mask_action([BUTTON_HEAL, 0, 0, 0, 0, TARGET_SELF], info)
            if hero.skill.summoner.usable and hero.skill.summoner.config_id in (80102, 80107, 80109, 80115):
                return self.mask_action([BUTTON_SUMMONER, 0, 0, 0, 0, TARGET_SELF], info)

        if action[0] == BUTTON_ATTACK and not self.attackable_target_indices(info):
            target = self.safe_target_position(info)
            return self.move_to(info.hero_our.info.position, target)

        if hp_rate < GameConfig.LOW_HP_HEAL_RATE and action[0] not in (BUTTON_HEAL, BUTTON_SUMMONER):
            if info.cake_our is not None:
                return self.move_to(info.hero_our.info.position, info.cake_our.position)
            return self.move_to(info.hero_our.info.position, info.organ_our.sub_tower.position)

        if action[0] in (BUTTON_SKILL_1, BUTTON_SKILL_2, BUTTON_SKILL_3):
            target_idx = self.preferred_skill_target(info, action[0])
            action[-1] = target_idx
            if target_idx != TARGET_NONE:
                action[3], action[4] = 8, 8
        elif action[0] == BUTTON_SUMMONER:
            targets = self.summoner_target_indices(info)
            if action[-1] not in targets:
                action[-1] = targets[0] if targets else TARGET_NONE
        return self.mask_action(action, info)

    def record_executed_action(self, action, info: Info):
        self.pending_attack_move = (
            action is not None
            and len(action) > 0
            and int(action[0]) == BUTTON_ATTACK
            and bool(self.attackable_target_indices(info))
            and not self.is_luban_enhanced_attack_state(info)
        )

    def split_legal_action(self, legal_action):
        cuts = np.cumsum(Config.LEGAL_ACTION_SIZE_LIST)[:-1]
        buttons, move_x, move_z, skill_x, skill_z, target = np.split(legal_action, cuts)
        return buttons, move_x, move_z, skill_x, skill_z, target.reshape(Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1])

    def join_legal_action(self, parts):
        buttons, move_x, move_z, skill_x, skill_z, target = parts
        return np.concatenate([buttons, move_x, move_z, skill_x, skill_z, target.reshape(-1)], axis=0)

    def keep_targets(self, target_mask, button, keep_indices):
        if not keep_indices:
            return
        current = target_mask[button].copy()
        new_mask = np.zeros_like(current)
        for idx in keep_indices:
            if 0 <= idx < len(current) and current[idx] > 0:
                new_mask[idx] = 1.0
        if new_mask.sum() > 0:
            target_mask[button] = new_mask

    def attackable_target_indices(self, info: Info):
        hero = info.hero_our.info
        attack_range = max(hero.attack_range, 1)
        targets = []
        if self.in_range(hero.position, info.hero_enemy.info.position, attack_range):
            targets.append(TARGET_ENEMY_HERO)
        for idx, soldier in enumerate(self.sorted_enemy_soldiers(info)[: Args.SOLDIER_MAX_NUM]):
            if self.in_range(hero.position, soldier.position, attack_range + 500):
                targets.append(TARGET_SOLDIER_BASE + idx)
        if self.in_range(hero.position, info.organ_enemy.sub_tower.position, attack_range + 500):
            targets.append(TARGET_TOWER)
        if info.river_crab is not None and self.in_range(hero.position, info.river_crab.position, attack_range + 500):
            targets.append(TARGET_MONSTER)
        return targets

    def skill_target_indices(self, info: Info, button):
        hero = info.hero_our.info
        radius = 13000 if button in (BUTTON_SKILL_1, BUTTON_SKILL_2) else 16000
        targets = []
        if self.in_range(hero.position, info.hero_enemy.info.position, radius):
            targets.append(TARGET_ENEMY_HERO)
        for idx, soldier in enumerate(self.sorted_enemy_soldiers(info)[: Args.SOLDIER_MAX_NUM]):
            if self.in_range(hero.position, soldier.position, radius):
                targets.append(TARGET_SOLDIER_BASE + idx)
        if self.in_range(hero.position, info.organ_enemy.sub_tower.position, radius):
            targets.append(TARGET_TOWER)
        if info.river_crab is not None and self.in_range(hero.position, info.river_crab.position, radius):
            targets.append(TARGET_MONSTER)
        targets.append(TARGET_NONE)
        return targets

    def summoner_target_indices(self, info: Info):
        skill_id = info.hero_our.skill.summoner.config_id
        if skill_id in (80102, 80107, 80109, 80110, 80115):
            return [TARGET_SELF, TARGET_NONE]
        if skill_id in (80103, 80108, 80121):
            return [TARGET_ENEMY_HERO, TARGET_NONE]
        if skill_id == 80104:
            return [
                TARGET_MONSTER,
                TARGET_SOLDIER_BASE,
                TARGET_SOLDIER_BASE + 1,
                TARGET_SOLDIER_BASE + 2,
                TARGET_SOLDIER_BASE + 3,
                TARGET_NONE,
            ]
        if skill_id == 80105:
            return [TARGET_TOWER, TARGET_NONE]
        return [TARGET_NONE, TARGET_ENEMY_HERO, TARGET_SOLDIER_BASE, TARGET_MONSTER, TARGET_TOWER]

    def has_skill_target(self, info: Info):
        return any(target != TARGET_NONE for target in self.skill_target_indices(info, BUTTON_SKILL_1))

    def preferred_combat_target(self, info: Info):
        targets = self.attackable_target_indices(info)
        if TARGET_ENEMY_HERO in targets:
            return TARGET_ENEMY_HERO
        for target in targets:
            if TARGET_SOLDIER_BASE <= target < TARGET_SOLDIER_BASE + Args.SOLDIER_MAX_NUM:
                return target
        if TARGET_TOWER in targets:
            return TARGET_TOWER
        return TARGET_NONE

    def preferred_skill_target(self, info: Info, button):
        targets = self.skill_target_indices(info, button)
        if TARGET_ENEMY_HERO in targets:
            return TARGET_ENEMY_HERO
        for target in targets:
            if TARGET_SOLDIER_BASE <= target < TARGET_SOLDIER_BASE + Args.SOLDIER_MAX_NUM:
                return target
        if TARGET_TOWER in targets:
            return TARGET_TOWER
        if TARGET_MONSTER in targets:
            return TARGET_MONSTER
        return TARGET_NONE

    def is_defensive_summoner(self, skill_id, info: Info):
        if skill_id in (80102, 80107, 80109, 80115, 80121):
            return True
        if skill_id == 80103:
            return self.distance(info.hero_our.info.position, info.hero_enemy.info.position) <= GameConfig.CLOSE_ENEMY_RANGE
        return False

    def is_luban_enhanced_attack_state(self, info: Info):
        hero = info.hero_our
        if hero.info.config_id != 112:
            return False
        normal_attack = hero.skill.normal_attack
        if normal_attack.config_id in GameConfig.LUBAN_ENHANCED_ATTACK_IDS:
            return True
        if normal_attack.next_config_id in GameConfig.LUBAN_ENHANCED_ATTACK_IDS:
            return True
        return bool(set(hero.info.buff.skill_ids) & GameConfig.LUBAN_ENHANCED_ATTACK_BUFFS)

    def sorted_enemy_soldiers(self, info: Info):
        hero_pos = info.hero_our.info.position
        return sorted(info.soldiers_enemy.merge, key=lambda actor: self.distance(hero_pos, actor.position))

    def safe_target_position(self, info: Info):
        if info.cake_our is not None:
            return info.cake_our.position
        return info.organ_our.sub_tower.position

    def move_to(self, current, target):
        delta_x, delta_z = self.delta_action_16x16(current, target)
        return [BUTTON_MOVE, delta_x, delta_z, 0, 0, TARGET_NONE]

    def mask_action(self, action, info: Info):
        mask = info.sub_action_mask.get(str(action[0])) if isinstance(info.sub_action_mask, dict) else None
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if isinstance(mask, tuple):
            mask = list(mask)
        if not isinstance(mask, list) or len(mask) != len(Config.LABEL_SIZE_LIST):
            return action
        fixed_mask = []
        for value in mask:
            try:
                fixed_mask.append(1 if int(value) > 0 else 0)
            except (TypeError, ValueError):
                fixed_mask.append(1)
        return (np.array(action, dtype=np.int32) * np.array(fixed_mask, dtype=np.int32)).astype(np.int32).tolist()

    def delta_action_16x16(self, current, target):
        if current[0] == Info.UNSEEN_PADDING or target[0] == Info.UNSEEN_PADDING:
            return 8, 8
        delta = np.array(target, dtype=np.float32) - np.array(current, dtype=np.float32)
        max_abs = float(np.max(np.abs(delta)))
        if max_abs < 1e-6:
            return 8, 8
        result = np.ceil(delta / max_abs * 7).astype(np.int32) + np.array([8, 8], dtype=np.int32)
        return int(np.clip(result[0], 0, 15)), int(np.clip(result[1], 0, 15))

    def in_range(self, source, target, radius):
        return self.distance(source, target) <= radius

    def distance(self, source, target):
        if source[0] == Info.UNSEEN_PADDING or target[0] == Info.UNSEEN_PADDING:
            return float("inf")
        return math.dist(source, target)
