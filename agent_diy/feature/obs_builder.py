#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
from typing import List

import numpy as np

from agent_diy.conf.conf import Args
from agent_diy.feature.state_info import ActorInfo, BulletInfo, CakeInfo, Info, SlotInfo


def clip(value, mn, mx):
    return max(min(value, mx), mn)


def fix(value):
    return math.floor(value) if value > 0 else math.ceil(value)


def dist(a, b):
    if a[0] == Info.UNSEEN_PADDING or b[0] == Info.UNSEEN_PADDING:
        return float("inf")
    return math.dist(a, b)


class ObsBuilder:
    def __init__(self, logger=None):
        self.logger = logger
        self.reset()

    def reset(self):
        self.last_money = [None, None]
        self.next_cake_frame = [1778, 1778]
        self.info = None
        self.pos = [0, 0]
        self.n_frame = 0

    def process_position(self, position):
        if position[0] == Info.UNSEEN_PADDING:
            return [0.0] * Args.DIM_DISTANCE

        unit_size, max_size = Args.RELATIVE_DISTANCE_UNIT_SIZE, Args.RELATIVE_DISTANCE_MAX_SIZE
        max_idx = max_size / unit_size + 1
        max_idx = max_idx / 2
        rel_pos = [
            int(clip(fix((position[i] - self.pos[i]) / unit_size), -max_idx, max_idx) + max_idx)
            for i in range(2)
        ]
        rel_dim = int(2 * max_idx + 1)
        rel_distance = clip(math.dist(position, self.pos) / (max_size / 2), 0, 1)
        x_rel = [0.0] * (rel_dim * 2 + 1)
        x_rel[rel_pos[0]] = 1.0
        x_rel[rel_pos[1] + rel_dim] = 1.0
        x_rel[-1] = rel_distance

        unit_size = Args.WHOLE_DISTANCE_UNIT_SIZE
        max_size = Args.WHOLE_DISTANCE_MAX_SIZE
        whole_dim = int(max_size / unit_size)
        whole_pos = [
            int(clip(math.floor((position[i] + max_size / 2) / unit_size), 0, whole_dim - 1))
            for i in range(2)
        ]
        whole_ratio = [
            clip((position[i] + max_size / 2) / unit_size - whole_pos[i], -1, 1)
            for i in range(2)
        ]
        x_whole = [0.0] * (whole_dim * 2 + 2)
        x_whole[whole_pos[0]] = 1.0
        x_whole[whole_pos[1] + whole_dim] = 1.0
        x_whole[-2] = whole_ratio[0]
        x_whole[-1] = whole_ratio[1]
        result = x_rel + x_whole
        assert len(result) == Args.DIM_DISTANCE
        return result

    def process_unit(self, actor: ActorInfo):
        hp_dim = int(Args.HP_MAX_SIZE / Args.HP_UNIT_SIZE) + 2
        x_hp = [0.0] * (1 + hp_dim)
        x_hp[0] = actor.hp / actor.hp_max if actor.hp_max > 0 else 0.0
        hp_idx = min(int(math.ceil(max(actor.hp, 0) / Args.HP_UNIT_SIZE)), hp_dim - 1)
        x_hp[1 + hp_idx] = 1.0

        x_mark = [0.0] * Args.DIM_MARK
        if actor.buff.marks_ids:
            x_mark[-1] = 1.0

        result = x_hp + x_mark + self.process_position(actor.position)
        assert len(result) == Args.DIM_UNIT
        return result

    def process_skill(self, slot: SlotInfo):
        cd_bucket_dim = int(Args.CD_MAX_SIZE / Args.CD_UNIT_SIZE) + 2
        x_cd = [0.0] * (4 + cd_bucket_dim)
        x_cd[0] = clip(slot.cd / slot.cd_max, 0, 1) if slot.cd_max > 0 else 0.0
        x_cd[1] = float(slot.usable)
        x_cd[2] = float(slot.level > 0)
        x_cd[3] = float(slot.flag_used > 0)
        cd_idx = min(int(math.ceil(max(slot.cd, 0) / Args.CD_UNIT_SIZE)), cd_bucket_dim - 1)
        x_cd[4 + cd_idx] = 1.0
        assert len(x_cd) == Args.DIM_SKILL
        return x_cd

    def process_summoner(self, slot: SlotInfo):
        x = [0.0] * Args.DIM_SUMMONER
        if slot.config_id in Args.SUMMONER_SKILL_IDS:
            x[Args.SUMMONER_SKILL_IDS.index(slot.config_id)] = 1.0
        base = len(Args.SUMMONER_SKILL_IDS)
        x[base] = clip(slot.cd / slot.cd_max, 0, 1) if slot.cd_max > 0 else 0.0
        x[base + 1] = float(slot.usable)
        x[base + 2] = float(slot.level > 0)
        x[base + 3] = float(slot.flag_used > 0)
        x[base + 4] = float(slot.config_id not in Args.SUMMONER_SKILL_IDS)
        return x

    def process_money(self, money, is_enemy):
        idx = int(is_enemy)
        if self.last_money[idx] is None:
            self.last_money[idx] = money
        delta = money - self.last_money[idx]
        self.last_money[idx] = money

        money_dim = int(Args.MONEY_MAX_SIZE / Args.MONEY_UNIT_SIZE) + 1
        x_money = [0.0] * (2 + money_dim)
        if 0 < delta < 20:
            x_money[-2] = 1.0
        else:
            bucket = max(min(int(math.floor(delta / Args.MONEY_UNIT_SIZE)), money_dim - 1), 0)
            x_money[bucket] = 1.0
        x_money[-1] = min(money / 10000, 1.0)
        return x_money

    def process_organ_flags(self, hero, enemy_sub_tower):
        return [
            float(dist(hero.info.position, enemy_sub_tower.position) <= enemy_sub_tower.attack_range),
            float(hero.info.id != 0 and hero.info.id == enemy_sub_tower.attack_target),
        ]

    def process_hero(self, hero, is_enemy, enemy_sub_tower):
        hero_head = Args.HERO_CONFIG_ID.index(hero.info.config_id) if hero.info.config_id in Args.HERO_CONFIG_ID else 0
        x_hero_id = [float(hero_head)]

        x_behave = [0.0] * (len(Args.HERO_BEHAVE) + 1)
        if hero.info.behave in Args.HERO_BEHAVE:
            x_behave[Args.HERO_BEHAVE.index(hero.info.behave)] = 1.0
        else:
            x_behave[-1] = 1.0

        ep_dim = int(Args.EP_MAX_SIZE / Args.EP_UNIT_SIZE) + 1
        x_ep = [0.0] * (1 + ep_dim)
        x_ep[0] = hero.info.ep / hero.info.ep_max if hero.info.ep_max > 0 else 0.0
        ep_idx = min(int(math.floor(max(hero.info.ep, 0) / Args.EP_UNIT_SIZE)), ep_dim - 1)
        x_ep[1 + ep_idx] = 1.0

        skill = hero.skill
        x_skills = (
            self.process_skill(skill.first)
            + self.process_skill(skill.second)
            + self.process_skill(skill.third)
            + self.process_skill(skill.recover)
            + self.process_skill(skill.normal_attack)
        )
        x_summoner = self.process_summoner(skill.summoner)

        x_level = [0.0] * Args.LEVEL_MAX
        level = max(min(hero.level, Args.LEVEL_MAX), 1)
        x_level[level - 1] = 1.0

        x_money = self.process_money(hero.money_total, is_enemy)
        x_grass = [float(hero.flag_in_grass)]
        x_organ = self.process_organ_flags(hero, enemy_sub_tower)

        x_buff = [0.0] * Args.DIM_BUFF
        for buff_id in hero.info.buff.skill_ids:
            if buff_id in Args.BUFFS:
                x_buff[Args.BUFFS.index(buff_id)] = 1.0
            else:
                x_buff[-1] = 1.0

        result = (
            x_hero_id
            + x_behave
            + x_ep
            + x_skills
            + x_summoner
            + x_level
            + x_money
            + x_grass
            + x_organ
            + x_buff
            + self.process_unit(hero.info)
        )
        assert len(result) == Args.DIM_HERO, f"{len(result)=}, {Args.DIM_HERO=}"
        return result

    def process_soldiers(self, soldiers: List[ActorInfo], opposed_sub_tower: ActorInfo):
        soldiers = sorted(soldiers, key=lambda actor: dist(actor.position, self.pos))
        soldiers = soldiers[: Args.SOLDIER_MAX_NUM]
        x_soldiers = []
        mask_soldiers = [0.0] * Args.SOLDIER_MAX_NUM

        for idx, soldier in enumerate(soldiers):
            x = [0.0] * (Args.DIM_SOLDIER - Args.DIM_UNIT)
            base = 0
            if soldier.behave in Args.SOLDIER_BEHAVE:
                x[base + Args.SOLDIER_BEHAVE.index(soldier.behave)] = 1.0
            else:
                x[base + len(Args.SOLDIER_BEHAVE)] = 1.0
            base += len(Args.SOLDIER_BEHAVE) + 1

            for group_idx, config_ids in enumerate(Args.SOLDIER_CONFIG_ID):
                if soldier.config_id in config_ids:
                    x[base + group_idx] = 1.0
                    break
            base += len(Args.SOLDIER_CONFIG_ID)

            x[base] = float(dist(soldier.position, opposed_sub_tower.position) <= opposed_sub_tower.attack_range)
            x[base + 1] = float(soldier.id != 0 and soldier.id == opposed_sub_tower.attack_target)
            x_soldiers += x + self.process_unit(soldier)
            mask_soldiers[idx] = 1.0

        x_soldiers += [0.0] * (Args.DIM_SOLDIERS - len(x_soldiers))
        return x_soldiers, mask_soldiers

    def process_river_crab(self):
        x_behave = [0.0] * (len(Args.RIVER_CRAB_BEHAVE) + 1)
        mask = [0.0]
        crab = self.info.river_crab
        if crab is None:
            return [0.0] * Args.DIM_RIVER_CRAB, mask
        mask[0] = 1.0
        if crab.behave in Args.RIVER_CRAB_BEHAVE:
            x_behave[Args.RIVER_CRAB_BEHAVE.index(crab.behave)] = 1.0
        else:
            x_behave[-1] = 1.0
        result = x_behave + self.process_unit(crab)
        assert len(result) == Args.DIM_RIVER_CRAB
        return result, mask

    def process_sub_tower(self, sub_tower: ActorInfo, cake: CakeInfo, is_enemy):
        x_target = [0.0] * 5
        if sub_tower.attack_target == 0:
            x_target[0] = 1.0
        else:
            target_type = self.info.id2type.get(sub_tower.attack_target)
            if target_type == "hero":
                x_target[1] = 1.0
            elif target_type == "soldier":
                x_target[2] = 1.0

        x_target[3] = float(cake is not None)
        cake_idx = int(is_enemy)
        if cake is not None:
            self.next_cake_frame[cake_idx] = self.n_frame + 76 * 30
            x_target[4] = 0.0
        else:
            x_target[4] = clip((self.next_cake_frame[cake_idx] - self.n_frame) / (75 * 30), 0, 1)

        result = x_target + self.process_unit(sub_tower)
        assert len(result) == Args.DIM_ORGAN
        return result

    def process_bullet(self, bullet: BulletInfo):
        x = [0.0] * (Args.DIM_BULLET - Args.DIM_DISTANCE)
        if bullet.slot_type in Args.BULLET_SLOT:
            x[Args.BULLET_SLOT.index(bullet.slot_type)] = 1.0
        else:
            x[len(Args.BULLET_SLOT)] = 1.0
        return x + self.process_position(bullet.position)

    def process_bullets(self):
        hero_bullets = sorted(self.info.bullets_enemy.hero, key=lambda b: dist(b.position, self.pos))
        hero_bullets = hero_bullets[: Args.BULLET_MAX_NUM - 1]
        x_bullets = []
        mask_bullets = [0.0] * Args.BULLET_MAX_NUM
        for idx, bullet in enumerate(hero_bullets):
            x_bullets += self.process_bullet(bullet)
            mask_bullets[idx] = 1.0
        x_bullets += [0.0] * ((Args.BULLET_MAX_NUM - 1) * Args.DIM_BULLET - len(x_bullets))

        organ_bullets = sorted(self.info.bullets_enemy.organ, key=lambda b: dist(b.position, self.pos))
        if organ_bullets:
            x_bullets += self.process_bullet(organ_bullets[0])
            mask_bullets[-1] = 1.0
        x_bullets += [0.0] * (Args.DIM_BULLETS - len(x_bullets))
        return x_bullets, mask_bullets

    def build_observation(self, info: Info, need_mask=False):
        self.info = info
        self.n_frame = info.n_frame
        self.pos = info.hero_our.info.position
        if self.last_money[0] is None:
            self.last_money = [info.hero_our.money_total, info.hero_enemy.money_total]

        x_hero_our = self.process_hero(info.hero_our, False, info.organ_enemy.sub_tower)
        x_hero_enemy = self.process_hero(info.hero_enemy, True, info.organ_our.sub_tower)
        x_soldier_our, mask_soldier_our = self.process_soldiers(info.soldiers_our.merge, info.organ_enemy.sub_tower)
        x_soldier_enemy, mask_soldier_enemy = self.process_soldiers(info.soldiers_enemy.merge, info.organ_our.sub_tower)
        x_river_crab, mask_river_crab = self.process_river_crab()
        x_sub_tower_our = self.process_sub_tower(info.organ_our.sub_tower, info.cake_our, False)
        x_sub_tower_enemy = self.process_sub_tower(info.organ_enemy.sub_tower, info.cake_enemy, True)

        x_all_units = (
            x_hero_our
            + x_hero_enemy
            + x_soldier_our
            + x_soldier_enemy
            + x_river_crab
            + x_sub_tower_our
            + x_sub_tower_enemy
        )
        assert len(x_all_units) == Args.DIM_ALL_UNITS, f"{len(x_all_units)=}, {Args.DIM_ALL_UNITS=}"

        x_bullets, mask_bullets = self.process_bullets()
        feature = np.array(x_all_units + x_bullets, dtype=np.float32)
        assert feature.shape[0] == Args.DIM_ALL
        if need_mask:
            return feature, (mask_soldier_our + mask_soldier_enemy, mask_river_crab, mask_bullets)
        return feature
