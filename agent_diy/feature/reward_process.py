#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math

from agent_diy.conf.conf import GameConfig
from agent_diy.feature.state_info import (
    ACTOR_SUB_CRYSTAL,
    ACTOR_SUB_SOLDIER,
    ACTOR_SUB_TOWER,
    ACTOR_SUB_TOWER_SPRING,
    ACTOR_TYPE_MONSTER,
    ACTOR_TYPE_ORGAN,
    RIVER_CRAB_CONFIG_ID,
    UNSEEN_PADDING,
)


ACTION_REWARD_NAMES = (
    "skill_hit_hero",
    "skill_hit_unit",
    "skill_empty",
    "valid_attack",
    "empty_attack",
    "attack_move",
    "attack_no_move",
    "low_hp_heal",
    "bad_heal",
    "eat_cake",
    "summoner_heal",
    "summoner_stun",
    "summoner_smite",
    "summoner_interfere",
    "summoner_purify",
    "summoner_execute",
    "summoner_sprint",
    "summoner_frenzy",
    "summoner_flash",
    "summoner_weaken",
    "summoner_bad",
)


SUMMONER_REWARD_BY_ID = {
    80102: "summoner_heal",
    80103: "summoner_stun",
    80104: "summoner_smite",
    80105: "summoner_interfere",
    80107: "summoner_purify",
    80108: "summoner_execute",
    80109: "summoner_sprint",
    80110: "summoner_frenzy",
    80115: "summoner_flash",
    80121: "summoner_weaken",
}


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
        self.prev_metrics = {}
        self.pending_actions = []
        self.pending_attack_moves = []
        self.last_policy_action = None

    def set_last_action(self, action):
        self.last_policy_action = list(action) if action is not None else None

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
        main_base, main_spring = None, None
        for npc in frame_data.get("npc_states", []):
            if npc.get("actor_type") != ACTOR_TYPE_ORGAN:
                continue
            sub_type = npc.get("sub_type")
            if sub_type == ACTOR_SUB_TOWER:
                if npc.get("camp") == camp:
                    main_tower = npc
                else:
                    enemy_tower = npc
            elif npc.get("camp") == camp and sub_type in (ACTOR_SUB_TOWER_SPRING, ACTOR_SUB_CRYSTAL):
                if sub_type == ACTOR_SUB_TOWER_SPRING or main_base is None:
                    main_base = npc
                if sub_type == ACTOR_SUB_TOWER_SPRING:
                    main_spring = npc
        if main_tower is None:
            main_tower = {"hp": 0, "max_hp": 1, "location": {"x": 0, "z": 0}}
        if enemy_tower is None:
            enemy_tower = {"hp": 0, "max_hp": 1, "location": {"x": 0, "z": 0}}
        if main_base is None:
            main_base = main_tower
        if main_spring is None:
            main_spring = main_base

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
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_base, enemy_tower)
            elif reward_name == "hero_damage":
                reward_struct.cur_frame_value = main_hero.get("total_hurt_to_hero", 0)
            elif reward_name in ACTION_REWARD_NAMES:
                reward_struct.cur_frame_value = 0.0
            elif reward_name == "low_hp_safe":
                reward_struct.cur_frame_value = self.calculate_low_hp_safe(main_hero, main_tower, main_spring, frame_data)
            elif reward_name == "eat_cake":
                reward_struct.cur_frame_value = 0.0

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

    def calculate_forward(self, main_hero, main_base, enemy_tower):
        hero_hp_rate = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1)
        if hero_hp_rate <= 0.0:
            return 0.0
        hero_pos = self.position_2d(main_hero)
        main_base_pos = self.position_2d(main_base)
        enemy_tower_pos = self.position_2d(enemy_tower)
        if not self.is_visible_position(hero_pos) or not self.is_visible_position(main_base_pos) or not self.is_visible_position(enemy_tower_pos):
            return 0.0
        dist_hero_enemy = math.dist(hero_pos, enemy_tower_pos)
        dist_base_enemy = max(math.dist(main_base_pos, enemy_tower_pos), 1.0)
        progress = (dist_base_enemy - dist_hero_enemy) / dist_base_enemy
        return max(min(progress, 1.0), 0.0)

    def calculate_low_hp_safe(self, main_hero, main_tower, main_spring, frame_data):
        hp_rate = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1)
        low_hp_weight = max(0.0, (GameConfig.LOW_HP_SAFE_RATE - hp_rate) / GameConfig.LOW_HP_SAFE_RATE)
        if low_hp_weight <= 0.0:
            return 0.0

        hero_pos = self.position_2d(main_hero)
        cake_pos = self.find_cake_pos(frame_data, main_hero.get("camp"))
        target = cake_pos
        if target is None or not self.is_visible_position(target):
            target = self.position_2d(main_tower)
        if not self.is_visible_position(target) and main_spring is not None:
            target = self.position_2d(main_spring)
        if not self.is_visible_position(hero_pos) or not self.is_visible_position(target):
            return 0.0
        distance_score = 1.0 - min(math.dist(hero_pos, target) / GameConfig.SAFE_DISTANCE_MAX, 1.0)
        return low_hp_weight * max(distance_score, 0.0)

    def position_2d(self, actor):
        loc = actor.get("location", {}) if actor else {}
        if isinstance(loc, (list, tuple)):
            return (loc[0], loc[2] if len(loc) > 2 else loc[1])
        return (loc.get("x", 0), loc.get("z", 0))

    def find_cake_pos(self, frame_data, camp):
        for cake in frame_data.get("cakes") or []:
            if cake.get("configId") != 5:
                continue
            location = (cake.get("collider") or {}).get("location")
            if not location:
                continue
            x = location.get("x", 0) if isinstance(location, dict) else location[0]
            cake_camp = 1 if x < 0 else 2 if x > 0 else -1
            if cake_camp == camp:
                z = location.get("z", 0) if isinstance(location, dict) else location[2]
                return (x, z)
        return None

    def collect_action_metrics(self, frame_data, main_hero, enemy_hero):
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        current = self.build_metrics(frame_data, main_hero, enemy_hero)
        previous = self.prev_metrics.get(main_hero.get("runtime_id"))
        rewards = {name: 0.0 for name in ACTION_REWARD_NAMES}

        if previous is not None:
            hero_damage_delta = max(current["hero_damage"] - previous["hero_damage"], 0)
            total_damage_delta = max(current["total_damage"] - previous["total_damage"], 0)
            tower_damage_delta = max(previous["enemy_tower_hp"] - current["enemy_tower_hp"], 0)
            hit_ids = set(current["hit_ids"])
            hit_slots = set(current["hit_slots"])

            action = self.action_to_dict(self.last_policy_action) or self.extract_last_action(main_hero)
            if action is not None:
                button = action["button"]
                if button == 3:
                    if self.has_attack_value(hero_damage_delta, total_damage_delta, tower_damage_delta, hit_ids):
                        rewards["valid_attack"] += 1.0
                        if self.should_track_attack_move(previous, current):
                            self.pending_attack_moves.append(
                                {
                                    "expire": frame_no + GameConfig.ATTACK_MOVE_PENDING_FRAMES,
                                    "hero_id": main_hero.get("runtime_id"),
                                    "start_pos": current["hero_pos"],
                                }
                            )
                    else:
                        rewards["empty_attack"] += 1.0
                elif button in (4, 5, 6):
                    if hero_damage_delta > 0 or any(slot in (1, 2, 3) for slot in hit_slots):
                        rewards["skill_hit_hero"] += 1.0
                    elif total_damage_delta > 0 or tower_damage_delta > 0 or self.dead_enemy_soldier_by_hero(frame_data, main_hero):
                        rewards["skill_hit_unit"] += 1.0
                    else:
                        self.pending_actions.append(
                            {
                                "expire": frame_no + GameConfig.SKILL_EMPTY_PENDING_FRAMES,
                                "hero_id": main_hero.get("runtime_id"),
                                "base_hero_damage": previous["hero_damage"],
                                "base_total_damage": previous["total_damage"],
                                "base_enemy_tower_hp": previous["enemy_tower_hp"],
                            }
                        )
                elif button == 7:
                    hp_delta = current["hp"] - previous["hp"]
                    hp_rate_before = previous["hp_rate"]
                    if hp_rate_before < GameConfig.LOW_HP_HEAL_RATE and hp_delta > 0:
                        rewards["low_hp_heal"] += 1.0 + min(hp_delta / max(previous["max_hp"], 1), 1.0)
                    elif hp_rate_before > GameConfig.BAD_HEAL_HP_RATE:
                        rewards["bad_heal"] += 1.0
                elif button == 8:
                    summoner_rewards = self.calculate_summoner_reward(previous, current, frame_data, main_hero)
                    for reward_name, value in summoner_rewards.items():
                        rewards[reward_name] += value

            if previous["cake_exists"] and not current["cake_exists"]:
                was_near_cake = (
                    previous["cake_pos"] is not None
                    and math.dist(previous["hero_pos"], previous["cake_pos"]) <= GameConfig.CAKE_PICKUP_RADIUS
                )
                if was_near_cake and previous["hp_rate"] < 0.95:
                    rewards["eat_cake"] += 0.5 + (1.0 - previous["hp_rate"])

        remaining_pending = []
        for pending in self.pending_actions:
            if pending["hero_id"] != main_hero.get("runtime_id"):
                remaining_pending.append(pending)
                continue
            if (
                current["hero_damage"] > pending["base_hero_damage"]
                or current["total_damage"] > pending["base_total_damage"]
                or current["enemy_tower_hp"] < pending["base_enemy_tower_hp"]
            ):
                continue
            if frame_no >= pending["expire"]:
                rewards["skill_empty"] += 1.0
            else:
                remaining_pending.append(pending)
        self.pending_actions = remaining_pending

        remaining_attack_moves = []
        for pending in self.pending_attack_moves:
            if pending["hero_id"] != main_hero.get("runtime_id"):
                remaining_attack_moves.append(pending)
                continue
            moved_distance = self.distance_or_inf(pending["start_pos"], current["hero_pos"])
            if math.isfinite(moved_distance) and moved_distance >= GameConfig.ATTACK_MOVE_MIN_DISTANCE:
                rewards["attack_move"] += 1.0
                continue
            if frame_no >= pending["expire"]:
                rewards["attack_no_move"] += 1.0
            else:
                remaining_attack_moves.append(pending)
        self.pending_attack_moves = remaining_attack_moves

        current["action"] = None
        self.last_policy_action = None
        self.prev_metrics[main_hero.get("runtime_id")] = current
        return rewards

    def build_metrics(self, frame_data, main_hero, enemy_hero):
        own_tower, enemy_tower, own_base, own_spring = self.find_key_organs(frame_data, main_hero.get("camp"))
        if own_tower is None:
            own_tower = self.default_actor()
        if enemy_tower is None:
            enemy_tower = self.default_actor()
        if own_base is None:
            own_base = own_tower
        if own_spring is None:
            own_spring = own_base

        hit_infos = main_hero.get("hit_target_info", []) or []
        cake_pos = self.find_cake_pos(frame_data, main_hero.get("camp"))
        hp = main_hero.get("hp", 0)
        max_hp = max(main_hero.get("max_hp", 1), 1)
        enemy_pos = self.position_2d(enemy_hero)
        enemy_visible = self.is_visible_position(enemy_pos) and enemy_hero is not None and enemy_hero.get("hp", 0) > 0
        enemy_hp = enemy_hero.get("hp", 0) if enemy_hero else 0
        enemy_max_hp = max(enemy_hero.get("max_hp", 1), 1) if enemy_hero else 1
        hero_pos = self.position_2d(main_hero)
        safe_pos = cake_pos
        if safe_pos is None or not self.is_visible_position(safe_pos):
            safe_pos = self.position_2d(own_tower)
        if not self.is_visible_position(safe_pos):
            safe_pos = self.position_2d(own_spring)
        normal_attack_slot = self.find_skill_slot(main_hero, 0)
        summoner_slot = self.find_skill_slot(main_hero, 6)
        buff_ids = self.get_buff_ids(main_hero)
        enemy_distance = self.distance_or_inf(hero_pos, enemy_pos) if enemy_visible else float("inf")
        return {
            "hp": hp,
            "max_hp": max_hp,
            "hp_rate": hp / max_hp,
            "hero_pos": hero_pos,
            "cake_exists": cake_pos is not None,
            "cake_pos": cake_pos,
            "hero_damage": main_hero.get("total_hurt_to_hero", 0),
            "total_damage": main_hero.get("total_hurt", 0),
            "be_hurt": main_hero.get("total_be_hurt_by_hero", 0),
            "kill": main_hero.get("kill_cnt", 0),
            "money": main_hero.get("money_cnt", main_hero.get("money", 0)),
            "exp": self.calculate_exp_sum(main_hero),
            "take_hurt_count": len(main_hero.get("take_hurt_infos", []) or []),
            "enemy_visible": enemy_visible,
            "enemy_hp": enemy_hp,
            "enemy_hp_rate": enemy_hp / enemy_max_hp if enemy_visible else 1.0,
            "enemy_pos": enemy_pos,
            "enemy_distance": enemy_distance,
            "enemy_tower_hp": enemy_tower.get("hp", 0),
            "enemy_tower_max_hp": max(enemy_tower.get("max_hp", 1), 1),
            "enemy_tower_pos": self.position_2d(enemy_tower),
            "enemy_tower_distance": self.distance_or_inf(hero_pos, self.position_2d(enemy_tower)),
            "own_tower_pos": self.position_2d(own_tower),
            "safe_pos": safe_pos,
            "safe_distance": self.distance_or_inf(hero_pos, safe_pos),
            "smite_target_distance": self.min_smite_target_distance(frame_data, main_hero),
            "forward_score": self.calculate_forward(main_hero, own_base, enemy_tower),
            "summoner_id": summoner_slot.get("configId", 0),
            "summoner_used_times": summoner_slot.get("usedTimes", 0),
            "summoner_usable": bool(summoner_slot.get("usable", False)),
            "config_id": main_hero.get("config_id", 0),
            "normal_attack_config_id": normal_attack_slot.get("configId", 0),
            "normal_attack_next_config_id": normal_attack_slot.get("nextConfigID", 0),
            "normal_attack_succ": normal_attack_slot.get("succUsedInFrame", 0),
            "buff_ids": buff_ids,
            "hit_ids": [info.get("hit_target", 0) for info in hit_infos],
            "hit_slots": [info.get("slot_type", -1) for info in hit_infos],
        }

    def should_track_attack_move(self, previous, current):
        if previous["config_id"] == 112 and self.is_luban_enhanced_attack_state(previous, current):
            return False
        return self.is_visible_position(current["hero_pos"])

    def is_luban_enhanced_attack_state(self, previous, current):
        attack_ids = GameConfig.LUBAN_ENHANCED_ATTACK_IDS
        attack_buffs = GameConfig.LUBAN_ENHANCED_ATTACK_BUFFS
        return (
            previous["normal_attack_config_id"] in attack_ids
            or previous["normal_attack_next_config_id"] in attack_ids
            or current["normal_attack_config_id"] in attack_ids
            or current["normal_attack_next_config_id"] in attack_ids
            or bool(previous["buff_ids"] & attack_buffs)
            or bool(current["buff_ids"] & attack_buffs)
        )

    def action_to_dict(self, action):
        if action is None:
            return None
        action = list(action)
        if not action:
            return None
        return {"button": int(action[0]), "raw": action, "target": int(action[-1]) if len(action) >= 6 else 0}

    def calculate_summoner_reward(self, previous, current, frame_data, main_hero):
        rewards = {name: 0.0 for name in ACTION_REWARD_NAMES if name.startswith("summoner_")}
        skill_id = previous.get("summoner_id", 0)
        reward_name = SUMMONER_REWARD_BY_ID.get(skill_id)
        if reward_name is None:
            rewards["summoner_bad"] += 1.0
            return rewards

        hero_damage_delta = max(current["hero_damage"] - previous["hero_damage"], 0)
        total_damage_delta = max(current["total_damage"] - previous["total_damage"], 0)
        tower_damage_delta = max(previous["enemy_tower_hp"] - current["enemy_tower_hp"], 0)
        damage_taken_delta = max(current["be_hurt"] - previous["be_hurt"], 0)
        enemy_hp_delta = (
            max(previous["enemy_hp"] - current["enemy_hp"], 0)
            if previous["enemy_visible"] and current["enemy_visible"]
            else 0
        )
        kill_delta = max(current["kill"] - previous["kill"], 0)
        hp_delta = current["hp"] - previous["hp"]
        hp_loss = max(previous["hp"] - current["hp"], 0)
        money_delta = max(current["money"] - previous["money"], 0)
        exp_delta = max(current["exp"] - previous["exp"], 0)

        enemy_near = previous["enemy_distance"] <= GameConfig.SUMMONER_EFFECT_RANGE
        enemy_close = previous["enemy_distance"] <= GameConfig.CLOSE_ENEMY_RANGE
        unit_near = previous["smite_target_distance"] <= GameConfig.SUMMONER_EFFECT_RANGE
        tower_near = previous["enemy_tower_distance"] <= GameConfig.ENEMY_TOWER_PRESSURE_RANGE
        low_hp = previous["hp_rate"] < GameConfig.LOW_HP_HEAL_RATE
        high_hp = previous["hp_rate"] > GameConfig.BAD_HEAL_HP_RATE
        safe_progress = self.progress_towards_safety(previous, current)
        escape_progress = self.escape_progress(previous, current)
        chase_progress = self.chase_progress(previous, current)
        forward_progress = max(current["forward_score"] - previous["forward_score"], 0.0)
        displacement = self.distance_or_inf(previous["hero_pos"], current["hero_pos"])
        moved = math.isfinite(displacement) and displacement > 1000
        dead_unit = self.dead_unit_by_hero(frame_data, main_hero)
        killed_enemy = bool(kill_delta > 0 or self.dead_enemy_hero_by_hero(frame_data, main_hero))
        took_recent_damage = previous["take_hurt_count"] > 0 or damage_taken_delta > 0
        purify_buff = 801070 in current["buff_ids"]

        if skill_id == 80102:
            if previous["hp_rate"] < 0.75 and hp_delta > 0:
                rewards[reward_name] += 1.0 + min(hp_delta / max(previous["max_hp"], 1), 0.8)
            elif high_hp:
                rewards["summoner_bad"] += 1.0
        elif skill_id == 80103:
            if enemy_close and (hero_damage_delta > 0 or escape_progress > 0.05 or chase_progress > 0.05):
                rewards[reward_name] += 0.7 + min(hero_damage_delta / 1200.0, 0.6) + max(escape_progress, chase_progress)
            elif not enemy_near:
                rewards["summoner_bad"] += 1.0
        elif skill_id == 80104:
            if unit_near and (total_damage_delta > 0 or dead_unit or money_delta > 0 or exp_delta > 0):
                rewards[reward_name] += 0.6 + min(total_damage_delta / 1000.0, 0.5)
                if dead_unit:
                    rewards[reward_name] += 0.6
            elif not unit_near:
                rewards["summoner_bad"] += 1.0
        elif skill_id == 80105:
            if tower_near and (tower_damage_delta > 0 or hero_damage_delta > 0 or safe_progress > 0.05):
                rewards[reward_name] += 0.6 + min(tower_damage_delta / 1500.0, 0.7) + min(hero_damage_delta / 1200.0, 0.4)
            elif not tower_near:
                rewards["summoner_bad"] += 1.0
        elif skill_id == 80107:
            if (took_recent_damage or enemy_near or previous["hp_rate"] < 0.7) and (
                purify_buff or safe_progress > 0.05 or escape_progress > 0.05 or hp_loss < previous["max_hp"] * 0.08
            ):
                rewards[reward_name] += 0.6 + max(safe_progress, escape_progress)
            elif high_hp and not enemy_near and not took_recent_damage:
                rewards["summoner_bad"] += 1.0
        elif skill_id == 80108:
            if enemy_near and previous["enemy_hp_rate"] < 0.4 and (hero_damage_delta > 0 or enemy_hp_delta > 0 or killed_enemy):
                rewards[reward_name] += 0.8 + min(max(hero_damage_delta, enemy_hp_delta) / 1200.0, 0.8)
                if killed_enemy:
                    rewards[reward_name] += 0.8
            elif (not enemy_near) or previous["enemy_hp_rate"] > 0.6:
                rewards["summoner_bad"] += 1.0
        elif skill_id == 80109:
            if low_hp and (safe_progress > 0.05 or escape_progress > 0.05):
                rewards[reward_name] += 0.5 + safe_progress + escape_progress
            elif not low_hp and ((enemy_near and (chase_progress > 0.05 or hero_damage_delta > 0)) or forward_progress > 0.02):
                rewards[reward_name] += 0.35 + chase_progress + min(hero_damage_delta / 1600.0, 0.4) + min(forward_progress, 0.4)
            elif not moved:
                rewards["summoner_bad"] += 0.7
        elif skill_id == 80110:
            damage_followup = hero_damage_delta > 0 or total_damage_delta > 0 or tower_damage_delta > 0
            if (enemy_near or tower_near or unit_near) and damage_followup:
                rewards[reward_name] += (
                    0.6
                    + min(hero_damage_delta / 1200.0, 0.6)
                    + min(total_damage_delta / 1800.0, 0.3)
                    + min(tower_damage_delta / 1500.0, 0.4)
                )
            elif not (enemy_near or tower_near or unit_near):
                rewards["summoner_bad"] += 1.0
        elif skill_id == 80115:
            if low_hp and (safe_progress > 0.05 or escape_progress > 0.05):
                rewards[reward_name] += 0.6 + safe_progress + escape_progress
            elif enemy_near and (chase_progress > 0.05 or hero_damage_delta > 0):
                rewards[reward_name] += 0.5 + chase_progress + min(hero_damage_delta / 1200.0, 0.5)
            elif not moved:
                rewards["summoner_bad"] += 1.0
        elif skill_id == 80121:
            if enemy_close:
                rewards[reward_name] += 0.5
                if hp_loss < previous["max_hp"] * 0.1:
                    rewards[reward_name] += 0.3
                if hero_damage_delta > 0 or escape_progress > 0.05:
                    rewards[reward_name] += min(hero_damage_delta / 1500.0, 0.4) + escape_progress
            else:
                rewards["summoner_bad"] += 1.0

        return rewards

    def find_key_organs(self, frame_data, camp):
        own_tower, enemy_tower, own_base, own_spring = None, None, None, None
        for npc in frame_data.get("npc_states", []):
            if npc.get("actor_type") != ACTOR_TYPE_ORGAN:
                continue
            sub_type = npc.get("sub_type")
            npc_camp = npc.get("camp")
            if sub_type == ACTOR_SUB_TOWER:
                if npc_camp == camp:
                    own_tower = npc
                else:
                    enemy_tower = npc
            elif npc_camp == camp and sub_type in (ACTOR_SUB_TOWER_SPRING, ACTOR_SUB_CRYSTAL):
                if sub_type == ACTOR_SUB_CRYSTAL:
                    own_base = npc
                elif sub_type == ACTOR_SUB_TOWER_SPRING:
                    own_spring = npc
        return own_tower, enemy_tower, own_base, own_spring

    def default_actor(self):
        return {"hp": 0, "max_hp": 1, "location": {"x": UNSEEN_PADDING, "z": UNSEEN_PADDING}}

    def find_skill_slot(self, hero, slot_type):
        for slot in self.get_slot_states(hero):
            if slot.get("slot_type") == slot_type:
                return slot
        return {}

    def get_slot_states(self, hero):
        return ((hero.get("skill_state") or {}).get("slot_states") or hero.get("slot_states") or [])

    def get_buff_ids(self, hero):
        buff_state = hero.get("buff_state") or {}
        return {buff.get("configId", 0) for buff in buff_state.get("buff_skills", []) or []}

    def min_smite_target_distance(self, frame_data, main_hero):
        hero_pos = self.position_2d(main_hero)
        min_distance = float("inf")
        for npc in frame_data.get("npc_states", []):
            if npc.get("actor_type") != ACTOR_TYPE_MONSTER:
                continue
            if npc.get("camp") == main_hero.get("camp") and npc.get("config_id") != RIVER_CRAB_CONFIG_ID:
                continue
            if npc.get("sub_type") != ACTOR_SUB_SOLDIER and npc.get("config_id") != RIVER_CRAB_CONFIG_ID:
                continue
            if npc.get("hp", 0) <= 0:
                continue
            min_distance = min(min_distance, self.distance_or_inf(hero_pos, self.position_2d(npc)))
        return min_distance

    def progress_towards_safety(self, previous, current):
        if not math.isfinite(previous["safe_distance"]) or not math.isfinite(current["safe_distance"]):
            return 0.0
        progress = previous["safe_distance"] - current["safe_distance"]
        return max(min(progress / GameConfig.SAFE_PROGRESS_DISTANCE, 1.0), 0.0)

    def escape_progress(self, previous, current):
        if not math.isfinite(previous["enemy_distance"]) or not math.isfinite(current["enemy_distance"]):
            return 0.0
        progress = current["enemy_distance"] - previous["enemy_distance"]
        return max(min(progress / GameConfig.SAFE_PROGRESS_DISTANCE, 1.0), 0.0)

    def chase_progress(self, previous, current):
        if not math.isfinite(previous["enemy_distance"]) or not math.isfinite(current["enemy_distance"]):
            return 0.0
        progress = previous["enemy_distance"] - current["enemy_distance"]
        return max(min(progress / GameConfig.CHASE_PROGRESS_DISTANCE, 1.0), 0.0)

    def extract_last_action(self, main_hero):
        real_cmds = main_hero.get("real_cmd", []) or []
        for cmd in reversed(real_cmds):
            command_type = cmd.get("command_type")
            if command_type == 2:
                return {"button": 2}
            if command_type in (4, 5):
                return {"button": 3}
            if command_type in (6, 7, 8):
                slot_type = (
                    (cmd.get("obj_skill") or {}).get("slotType")
                    or (cmd.get("dir_skill") or {}).get("slotType")
                    or (cmd.get("pos_skill") or {}).get("slotType")
                    or (cmd.get("charge_skill") or {}).get("slotType")
                )
                if slot_type in (1, 2, 3):
                    return {"button": 3 + slot_type}
                if slot_type == 5:
                    return {"button": 7}
                if slot_type == 6:
                    return {"button": 8}
                if slot_type == 7:
                    return {"button": 9}
        return None

    def has_attack_value(self, hero_damage_delta, total_damage_delta, tower_damage_delta, hit_ids):
        return hero_damage_delta > 0 or total_damage_delta > 0 or tower_damage_delta > 0 or bool(hit_ids)

    def dead_enemy_soldier_by_hero(self, frame_data, main_hero):
        for dead_action in (frame_data.get("frame_action") or {}).get("dead_action", []):
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            if death.get("sub_type") == ACTOR_SUB_SOLDIER and killer.get("runtime_id") == main_hero.get("runtime_id"):
                return True
        return False

    def dead_unit_by_hero(self, frame_data, main_hero):
        for dead_action in (frame_data.get("frame_action") or {}).get("dead_action", []):
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            if killer.get("runtime_id") != main_hero.get("runtime_id"):
                continue
            if death.get("sub_type") == ACTOR_SUB_SOLDIER or death.get("config_id") == RIVER_CRAB_CONFIG_ID:
                return True
        return False

    def dead_enemy_hero_by_hero(self, frame_data, main_hero):
        for dead_action in (frame_data.get("frame_action") or {}).get("dead_action", []):
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            if killer.get("runtime_id") != main_hero.get("runtime_id"):
                continue
            if death.get("actor_type") == 0 and death.get("camp") != main_hero.get("camp"):
                return True
        return False

    def is_visible_position(self, position):
        return position is not None and len(position) >= 2 and position[0] != UNSEEN_PADDING and position[1] != UNSEEN_PADDING

    def distance_or_inf(self, source, target):
        if not self.is_visible_position(source) or not self.is_visible_position(target):
            return float("inf")
        return math.dist(source, target)

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
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []):
            if hero.get("camp") == main_camp:
                main_hero = hero
            elif hero.get("camp") == enemy_camp:
                enemy_hero = hero
        if main_hero is not None:
            action_rewards = self.collect_action_metrics(frame_data, main_hero, enemy_hero)
            for reward_name, value in action_rewards.items():
                if reward_name in self.m_main_calc_frame_map:
                    self.m_main_calc_frame_map[reward_name].cur_frame_value = value

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
                reward_struct.value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_main_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value *= self.forward_hp_scale(frame_data)
                if GameConfig.REMOVE_FORWARD_AFTER is not None and frame_no > GameConfig.REMOVE_FORWARD_AFTER:
                    reward_struct.value = 0.0
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "hero_damage":
                main_cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                enemy_cur = self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                main_last = self.m_main_calc_frame_map[reward_name].last_frame_value
                enemy_last = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.cur_frame_value = main_cur - enemy_cur
                reward_struct.last_frame_value = main_last - enemy_last
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "low_hp_safe":
                reward_struct.value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_main_calc_frame_map[reward_name].last_frame_value
                )
            elif reward_name in ACTION_REWARD_NAMES:
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

    def forward_hp_scale(self, frame_data):
        main_hero = None
        for hero in frame_data.get("hero_states", []):
            if hero.get("runtime_id") == self.main_hero_runtime_id:
                main_hero = hero
                break
        if main_hero is None:
            return 1.0
        hp_rate = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1)
        if hp_rate < GameConfig.FORWARD_STOP_HP_RATE:
            return 0.0
        if hp_rate < GameConfig.FORWARD_FULL_HP_RATE:
            span = GameConfig.FORWARD_FULL_HP_RATE - GameConfig.FORWARD_STOP_HP_RATE
            return (hp_rate - GameConfig.FORWARD_STOP_HP_RATE) / max(span, 1e-6)
        return 1.0
