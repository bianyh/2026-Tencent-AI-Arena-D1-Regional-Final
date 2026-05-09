#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_diy.feature.feature_process.hero_process import HeroProcess
from agent_diy.feature.feature_process.organ_process import OrganProcess
from agent_diy.feature.feature_process.situation_process import SituationProcess


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.situation_process = SituationProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.situation_process = SituationProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_situation_feature(self, frame_state):
        return self.situation_process.process_vec_situation(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature = self.process_hero_feature(frame_state)
        organ_feature = self.process_organ_feature(frame_state)
        situation_feature = self.process_situation_feature(frame_state)

        feature = main_camp_hero_vector_feature + organ_feature + situation_feature

        return feature

