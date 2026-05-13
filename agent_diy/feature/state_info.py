#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Optional


ACTOR_TYPE_HERO = 0
ACTOR_TYPE_MONSTER = 1
ACTOR_TYPE_ORGAN = 2

ACTOR_SUB_SOLDIER = 11
ACTOR_SUB_TOWER = 21
ACTOR_SUB_TOWER_SPRING = 23
ACTOR_SUB_CRYSTAL = 24

SOLDIER_CONFIG_GROUPS = [[6801, 6804], [6800, 6803], [6802, 6805]]
RIVER_CRAB_CONFIG_ID = 6827
CAKE_CONFIG_ID = 5
UNSEEN_PADDING = 100000


def camp_to_zero(camp):
    if camp in (1, 2):
        return int(camp) - 1
    return -1


def cvt_position(position, inverse=False):
    if position is None:
        return [UNSEEN_PADDING, UNSEEN_PADDING]
    if isinstance(position, dict):
        x, z = position.get("x", UNSEEN_PADDING), position.get("z", UNSEEN_PADDING)
    else:
        x, z = position[0], position[2] if len(position) > 2 else position[1]
    if x == UNSEEN_PADDING or z == UNSEEN_PADDING:
        return [UNSEEN_PADDING, UNSEEN_PADDING]
    flag = -1 if inverse else 1
    return [flag * x, flag * z]


class Info:
    UNSEEN_PADDING = UNSEEN_PADDING

    def __init__(self, observation=None, reverse_camp1=True):
        self.reverse_camp1 = reverse_camp1
        self.id2type = {}
        self.state_dict = None
        if observation is not None:
            self.update(observation)

    def reset(self):
        self.id2type = {}

    def update(self, observation):
        self.state_dict = observation
        self.player_id = observation["player_id"]
        self.player_camp = camp_to_zero(observation["camp"])
        self.inverse_position = self.player_camp == 1 and self.reverse_camp1
        self.game_id = observation.get("env_id", "")
        self.legal_action = observation.get("legal_action", [])
        self.sub_action_mask = observation.get("sub_action_mask", {})
        self.frame_state = frame = observation["frame_state"]
        self.n_frame = frame.get("frame_no", frame.get("frameNo", 0))
        self.map_state = frame.get("map_state", False)

        hero_states = frame.get("hero_states", [])
        self.hero_our = HeroInfo(hero_states, self.player_camp, self.inverse_position)
        self.hero_enemy = HeroInfo(hero_states, 1 - self.player_camp, self.inverse_position)
        for hero in [self.hero_our, self.hero_enemy]:
            if hero.info.id:
                self.id2type[hero.info.id] = "hero"

        npc_states = frame.get("npc_states", [])
        self.organ_our = OrganInfo(npc_states, self.player_camp, self.inverse_position)
        self.organ_enemy = OrganInfo(npc_states, 1 - self.player_camp, self.inverse_position)
        for organ in [self.organ_our, self.organ_enemy]:
            for actor in [organ.sub_tower, organ.crystal, organ.spring]:
                if actor.id:
                    self.id2type[actor.id] = "organ"

        self.soldiers_our = SoldierInfo(npc_states, self.player_camp, self.inverse_position)
        self.soldiers_enemy = SoldierInfo(npc_states, 1 - self.player_camp, self.inverse_position)
        for soldiers in [self.soldiers_our, self.soldiers_enemy]:
            for soldier in soldiers.merge:
                if soldier.id:
                    self.id2type[soldier.id] = "soldier"

        self.river_crab = None
        for npc in npc_states:
            if npc.get("config_id") == RIVER_CRAB_CONFIG_ID:
                self.river_crab = ActorInfo(npc, self.inverse_position)
                self.id2type[self.river_crab.id] = "river_crab"
                break

        cakes = frame.get("cakes") or []
        self.cake_our = CakeInfo(cakes, self.player_camp, self.inverse_position)
        self.cake_enemy = CakeInfo(cakes, 1 - self.player_camp, self.inverse_position)
        if self.cake_our.position is None:
            self.cake_our = None
        if self.cake_enemy.position is None:
            self.cake_enemy = None

        bullets = frame.get("bullets") or []
        self.bullets_our = BulletsInfo(bullets, self.player_camp, self.id2type, self.inverse_position)
        self.bullets_enemy = BulletsInfo(bullets, 1 - self.player_camp, self.id2type, self.inverse_position)

        dead_actions = (frame.get("frame_action") or {}).get("dead_action", [])
        self.deads = DeadsInfo(dead_actions)


class HeroInfo:
    def __init__(self, hero_states, camp, inverse_position):
        self.hero_state = None
        for hero in hero_states:
            if hero.get("actor_type") == ACTOR_TYPE_HERO and camp_to_zero(hero.get("camp")) == camp:
                self.hero_state = hero
                break
        if self.hero_state is None:
            self.hero_state = {}
        s = self.hero_state
        self.player_id = s.get("player_id", 0)
        self.info = ActorInfo(s, inverse_position)
        self.skill = SkillInfo(s.get("skill_state", {"slot_states": s.get("slot_states", [])}))
        self.equip_state = s.get("equip_state", {})
        self.level = s.get("level", 1)
        self.exp = s.get("exp", 0)
        self.money = s.get("money", 0)
        self.money_total = s.get("money_cnt", s.get("moneyCnt", self.money))
        self.revive_time = s.get("revive_time", 0)
        self.kda = [s.get("kill_cnt", 0), s.get("dead_cnt", 0), s.get("assist_cnt", 0)]
        self.hurt_total = s.get("total_hurt", 0)
        self.hurt_hero_total = s.get("total_hurt_to_hero", 0)
        self.be_hurt_total = s.get("total_be_hurt_by_hero", 0)
        self.flag_in_grass = s.get("is_in_grass", False)
        self.passive_skill = s.get("passive_skill", [])


class ActorInfo:
    def __init__(self, actor_state=None, inverse_position=False):
        s = actor_state or {}
        self.actor_state = s
        self.config_id = s.get("config_id", 0)
        self.id = s.get("runtime_id", 0)
        self.type = [s.get("actor_type", -1), s.get("sub_type", -1)]
        self.camp = camp_to_zero(s.get("camp", 0))
        self.behave = s.get("behav_mode", -1)
        self.position = cvt_position(s.get("location"), inverse_position)
        self.forward = cvt_position(s.get("forward"), inverse_position)
        self.hp = s.get("hp", 0)
        self.hp_max = max(s.get("max_hp", 1), 1)
        self.ep = s.get("ep", 0)
        self.ep_max = max(s.get("max_ep", 1), 1)
        self.hp_recover = s.get("hp_recover", 0)
        self.ep_recover = s.get("ep_recover", 0)
        self.attack_range = s.get("attack_range", 0)
        self.attack_target = s.get("attack_target", 0)
        self.kill_bonus = s.get("kill_income", 0)
        self.hit_target_infos = [HitTargetInfo(x) for x in s.get("hit_target_info", [])]
        self.sight_range = s.get("sight_area", 0)
        self.buff_state = s.get("buff_state", {})
        self.buff = BuffInfo(self.buff_state)


class SkillInfo:
    def __init__(self, skill_state):
        slots = skill_state.get("slot_states", [])
        self.slot_by_type = {slot.get("slot_type"): SlotInfo(slot) for slot in slots}
        self.normal_attack = self.slot_by_type.get(0, SlotInfo())
        self.first = self.slot_by_type.get(1, SlotInfo())
        self.second = self.slot_by_type.get(2, SlotInfo())
        self.third = self.slot_by_type.get(3, SlotInfo())
        self.recover = self.slot_by_type.get(5, SlotInfo())
        self.summoner = self.slot_by_type.get(6, SlotInfo())
        self.back = self.slot_by_type.get(7, SlotInfo())


class SlotInfo:
    def __init__(self, slot_state=None):
        s = slot_state or {}
        self.slot_state = s
        self.config_id = s.get("configId", 0)
        self.slot_type = s.get("slot_type", -1)
        self.level = s.get("level", 0)
        self.usable = bool(s.get("usable", False))
        self.cd = s.get("cooldown", 0)
        self.cd_max = max(s.get("cooldown_max", 1), 1)
        self.count = s.get("usedTimes", 0)
        self.hit_hero_count = s.get("hitHeroTimes", 0)
        self.flag_used = s.get("succUsedInFrame", 0)
        self.next_config_id = s.get("nextConfigID", 0)
        self.combo_effect_time = s.get("comboEffectTime", 0)


class OrganInfo:
    def __init__(self, npc_states, camp, inverse_position):
        self.sub_tower = ActorInfo(None, inverse_position)
        self.crystal = ActorInfo(None, inverse_position)
        self.spring = ActorInfo(None, inverse_position)
        for npc in npc_states:
            if npc.get("actor_type") != ACTOR_TYPE_ORGAN:
                continue
            if camp_to_zero(npc.get("camp")) != camp:
                continue
            sub_type = npc.get("sub_type")
            if sub_type == ACTOR_SUB_TOWER:
                self.sub_tower = ActorInfo(npc, inverse_position)
            elif sub_type == ACTOR_SUB_CRYSTAL:
                self.crystal = ActorInfo(npc, inverse_position)
            elif sub_type == ACTOR_SUB_TOWER_SPRING:
                self.spring = ActorInfo(npc, inverse_position)


class SoldierInfo:
    def __init__(self, npc_states, camp, inverse_position):
        self.close: List[ActorInfo] = []
        self.remote: List[ActorInfo] = []
        self.car: List[ActorInfo] = []
        self.merge: List[ActorInfo] = []
        for npc in npc_states:
            if npc.get("actor_type") != ACTOR_TYPE_MONSTER:
                continue
            if npc.get("sub_type") != ACTOR_SUB_SOLDIER:
                continue
            if camp_to_zero(npc.get("camp")) != camp:
                continue
            actor = ActorInfo(npc, inverse_position)
            if actor.config_id in SOLDIER_CONFIG_GROUPS[0]:
                self.close.append(actor)
            elif actor.config_id in SOLDIER_CONFIG_GROUPS[1]:
                self.remote.append(actor)
            elif actor.config_id in SOLDIER_CONFIG_GROUPS[2]:
                self.car.append(actor)
        for group in [self.close, self.remote, self.car]:
            self.merge.extend(group)


class CakeInfo:
    def __init__(self, cakes, camp, inverse_position):
        self.position = None
        self.camp = None
        for cake in cakes:
            if cake.get("configId") != CAKE_CONFIG_ID:
                continue
            location = (cake.get("collider") or {}).get("location")
            if not location:
                continue
            raw_x = location.get("x", 0) if isinstance(location, dict) else location[0]
            cake_camp = 0 if raw_x < 0 else 1 if raw_x > 0 else -1
            if cake_camp == camp:
                self.position = cvt_position(location, inverse_position)
                self.camp = cake_camp
                break


class BulletsInfo:
    def __init__(self, bullets, camp, id2type, inverse_position):
        self.id2type = id2type
        self.soldier: List[BulletInfo] = []
        self.hero: List[BulletInfo] = []
        self.organ: List[BulletInfo] = []
        self.merge: List[BulletInfo] = []
        for bullet in bullets:
            if camp_to_zero(bullet.get("camp")) != camp:
                continue
            source = bullet.get("source_actor")
            source_type = id2type.get(source)
            if source_type is None:
                continue
            info = BulletInfo(bullet, source_type, inverse_position)
            if source_type == "soldier":
                self.soldier.append(info)
            elif source_type == "hero":
                self.hero.append(info)
            elif source_type == "organ":
                self.organ.append(info)
        for group in [self.soldier, self.hero, self.organ]:
            self.merge.extend(group)


class BulletInfo:
    def __init__(self, bullet, source_type, inverse_position):
        self.id = bullet.get("runtime_id", 0)
        self.source_id = bullet.get("source_actor", 0)
        self.camp = camp_to_zero(bullet.get("camp"))
        self.slot_type = bullet.get("slot_type", -1)
        self.skill_id = bullet.get("skill_id", 0)
        self.position = cvt_position(bullet.get("location"), inverse_position)
        self.type = source_type


class DeadsInfo:
    def __init__(self, deads):
        self.soldier: List[DeadInfo] = []
        self.organ: List[DeadInfo] = []
        self.hero: List[DeadInfo] = []
        self.river_crab: List[DeadInfo] = []
        self.merge: List[DeadInfo] = []
        for dead in deads:
            info = DeadInfo(dead)
            if info.death.type[1] == ACTOR_SUB_SOLDIER:
                self.soldier.append(info)
            elif info.death.type[0] == ACTOR_TYPE_ORGAN:
                self.organ.append(info)
            elif info.death.type[0] == ACTOR_TYPE_HERO:
                self.hero.append(info)
            elif info.death.config_id == RIVER_CRAB_CONFIG_ID:
                self.river_crab.append(info)
        for group in [self.soldier, self.hero, self.organ, self.river_crab]:
            self.merge.extend(group)


class DeadInfo:
    def __init__(self, dead):
        self.death = ActionActorInfo(dead.get("death", {}))
        self.killer = ActionActorInfo(dead.get("killer", {}))


class ActionActorInfo:
    def __init__(self, info):
        self.config_id = info.get("config_id", 0)
        self.id = info.get("runtime_id", 0)
        self.type = [info.get("actor_type", -1), info.get("sub_type", -1)]
        self.camp = camp_to_zero(info.get("camp", 0))


class BuffInfo:
    def __init__(self, buff_state):
        self.skills: List[BuffSkillInfo] = []
        self.skill_ids: List[int] = []
        self.marks: List[BuffMarkInfo] = []
        self.marks_ids: List[int] = []
        self.marks_layers: List[int] = []
        for buff_skill in buff_state.get("buff_skills", []):
            self.skills.append(BuffSkillInfo(buff_skill))
            self.skill_ids.append(self.skills[-1].id)
        for buff_mark in buff_state.get("buff_marks", []):
            self.marks.append(BuffMarkInfo(buff_mark))
            self.marks_ids.append(self.marks[-1].id)
            self.marks_layers.append(self.marks[-1].layer)


class BuffSkillInfo:
    def __init__(self, buff_skill):
        self.id = buff_skill.get("configId", 0)
        self.start_time = buff_skill.get("startTime", 0)
        self.times = buff_skill.get("times", 0)
        self.buff_skill_id = self.id


class BuffMarkInfo:
    def __init__(self, buff_mark):
        self.actor_id = buff_mark.get("origin_actorId", 0)
        self.id = buff_mark.get("configId", 0)
        self.layer = buff_mark.get("layer", 0)
        self.buff_mark_id = self.id


class HitTargetInfo:
    def __init__(self, info):
        self.hit_target = info.get("hit_target", 0)
        self.skill_id = info.get("skill_id", 0)
        self.slot_type = info.get("slot_type", 0)
        self.conti_hit_count = info.get("conti_hit_count", 0)
