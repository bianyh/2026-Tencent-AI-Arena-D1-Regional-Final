#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


class GameConfig:
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 10.0,
        "money": 4e-3,
        "exp": 4e-3,
        "ep_rate": 0.75,
        "death": -1.0,
        "kill": -0.6,
        "last_hit": 0.5,
        "forward": 0.3,
        "hero_damage": 0.002,
        "skill_hit_hero": 0.2,
        "skill_hit_unit": 0.05,
        "skill_empty": -0.04,
        "valid_attack": 0.04,
        "empty_attack": -0.04,
        "attack_move": 0.08,
        "attack_no_move": -0.03,
        "low_hp_safe": 0.5,
        "eat_cake": 1.0,
        "low_hp_heal": 0.6,
        "bad_heal": -0.2,
        "summoner_heal": 0.6,
        "summoner_stun": 0.4,
        "summoner_smite": 0.25,
        "summoner_interfere": 0.4,
        "summoner_purify": 0.4,
        "summoner_execute": 0.8,
        "summoner_sprint": 0.35,
        "summoner_frenzy": 0.35,
        "summoner_flash": 0.35,
        "summoner_weaken": 0.4,
        "summoner_bad": -0.2,
    }
    REMOVE_FORWARD_AFTER = 5000
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = set()
    MODEL_SAVE_INTERVAL = 1800
    CAMP_HEROES = [112, 133]
    FORWARD_FULL_HP_RATE = 0.75
    FORWARD_STOP_HP_RATE = 0.55
    LOW_HP_SAFE_RATE = 0.65
    LOW_HP_HEAL_RATE = 0.55
    BAD_HEAL_HP_RATE = 0.8
    CRITICAL_HP_RATE = 0.35
    SAFE_DISTANCE_MAX = 50000
    CAKE_PICKUP_RADIUS = 3500
    SKILL_EMPTY_PENDING_FRAMES = 18
    CLOSE_ENEMY_RANGE = 6500
    SUMMONER_EFFECT_RANGE = 7000
    ENEMY_TOWER_PRESSURE_RANGE = 11500
    SAFE_PROGRESS_DISTANCE = 5000
    CHASE_PROGRESS_DISTANCE = 7000
    ATTACK_MOVE_PENDING_FRAMES = 12
    ATTACK_MOVE_MIN_DISTANCE = 1200
    ATTACK_MOVE_LOGIT_BIAS = 0.35
    LUBAN_ENHANCED_ATTACK_IDS = {11201, 11203, 11204}
    LUBAN_ENHANCED_ATTACK_BUFFS = {112015, 112044, 112045, 112046, 112047, 112048}


class Args:
    HERO_CONFIG_ID = [112, 133]
    HERO_HEAD_NUM = len(HERO_CONFIG_ID)

    SUMMONER_SKILL_IDS = [
        80102,
        80103,
        80104,
        80105,
        80107,
        80108,
        80109,
        80110,
        80115,
        80121,
    ]

    RELATIVE_DISTANCE_UNIT_SIZE = 600
    RELATIVE_DISTANCE_MAX_SIZE = 24600
    DIM_RELATIVE_DISTANCE = (RELATIVE_DISTANCE_MAX_SIZE // RELATIVE_DISTANCE_UNIT_SIZE + 2) * 2 + 1
    WHOLE_DISTANCE_UNIT_SIZE = 5000
    WHOLE_DISTANCE_MAX_SIZE = int(9e4)
    DIM_WHOLE_DISTANCE = (WHOLE_DISTANCE_MAX_SIZE // WHOLE_DISTANCE_UNIT_SIZE) * 2 + 2
    DIM_DISTANCE = DIM_RELATIVE_DISTANCE + DIM_WHOLE_DISTANCE

    HP_UNIT_SIZE = 100
    HP_MAX_SIZE = 6000
    DIM_MARK = 1
    DIM_UNIT = DIM_DISTANCE + HP_MAX_SIZE // HP_UNIT_SIZE + 3 + DIM_MARK

    HERO_BEHAVE = [0, 2, 4, 9]
    EP_UNIT_SIZE = 30
    EP_MAX_SIZE = 600
    CD_UNIT_SIZE = 1000
    CD_MAX_SIZE = 120000
    LEVEL_MAX = 15
    MONEY_UNIT_SIZE = 20
    MONEY_MAX_SIZE = 300
    BUFFS = [
        11001,
        90015,
        90019,
        112001,
        112015,
        112044,
        112045,
        112046,
        112047,
        112048,
        112890,
        133001,
        133950,
        133951,
        801070,
        801090,
        801100,
        914110,
        914210,
        914211,
    ]
    DIM_BUFF = len(BUFFS) + 1
    DIM_SUMMONER = len(SUMMONER_SKILL_IDS) + 5
    DIM_CAKE = 2 + DIM_DISTANCE
    DIM_SKILL = 4 + int(CD_MAX_SIZE / CD_UNIT_SIZE) + 2
    DIM_HERO = (
        DIM_UNIT
        + 1
        + len(HERO_BEHAVE)
        + 1
        + EP_MAX_SIZE // EP_UNIT_SIZE
        + 2
        + DIM_SKILL * 5
        + DIM_SUMMONER
        + LEVEL_MAX
        + MONEY_MAX_SIZE // MONEY_UNIT_SIZE
        + 3
        + 1
        + 2
        + DIM_BUFF
    )

    SOLDIER_MAX_NUM = 4
    SOLDIER_BEHAVE = [1, 6]
    SOLDIER_CONFIG_ID = [[6801, 6804], [6800, 6803], [6802, 6805]]
    DIM_SOLDIER = DIM_UNIT + len(SOLDIER_BEHAVE) + 1 + len(SOLDIER_CONFIG_ID) + 2
    DIM_SOLDIERS = DIM_SOLDIER * SOLDIER_MAX_NUM

    RIVER_CRAB_BEHAVE = [0]
    DIM_RIVER_CRAB = DIM_UNIT + len(RIVER_CRAB_BEHAVE) + 1

    DIM_ORGAN = DIM_UNIT + 5

    BULLET_MAX_NUM = 10
    BULLET_SLOT = [0, 1, 2, 3, 5, 6, 7, 16]
    DIM_BULLET = len(BULLET_SLOT) + 1 + DIM_DISTANCE
    DIM_BULLETS = DIM_BULLET * BULLET_MAX_NUM

    DIM_ALL_UNITS = DIM_HERO * 2 + DIM_SOLDIERS * 2 + DIM_RIVER_CRAB + DIM_ORGAN * 2 + DIM_CAKE * 2
    DIM_ALL = DIM_ALL_UNITS + DIM_BULLETS


class DimConfig:
    DIM_OF_HERO_FRD = [Args.DIM_HERO]
    DIM_OF_HERO_EMY = [Args.DIM_HERO]
    DIM_OF_SOLDIER_1_4 = [Args.DIM_SOLDIER] * Args.SOLDIER_MAX_NUM
    DIM_OF_SOLDIER_5_8 = [Args.DIM_SOLDIER] * Args.SOLDIER_MAX_NUM
    DIM_OF_RIVER_CRAB = [Args.DIM_RIVER_CRAB]
    DIM_OF_ORGAN_1 = [Args.DIM_ORGAN]
    DIM_OF_ORGAN_2 = [Args.DIM_ORGAN]
    DIM_OF_CAKE_1 = [Args.DIM_CAKE]
    DIM_OF_CAKE_2 = [Args.DIM_CAKE]
    DIM_OF_BULLET_1_9 = [Args.DIM_BULLET] * (Args.BULLET_MAX_NUM - 1)
    DIM_OF_BULLET_10 = [Args.DIM_BULLET]


class Config:
    NETWORK_NAME = "network"
    LSTM_DROPOUT = 0
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DIM_PUBLIC = 512
    MULTI_HEAD = True
    SYNC_HERO_HEAD_INIT = True

    DATA_SPLIT_SHAPE = [
        Args.DIM_ALL + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(Args.DIM_ALL,), (85,)]
    INIT_LEARNING_RATE_START = 1e-5
    TARGET_LR = 1e-5
    TARGET_STEP = 1
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]
    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(Args.DIM_ALL + 85) * LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [12 * LSTM_TIME_STEPS],
        [16 * LSTM_TIME_STEPS],
        [16 * LSTM_TIME_STEPS],
        [16 * LSTM_TIME_STEPS],
        [16 * LSTM_TIME_STEPS],
        [9 * LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_UNIT_SIZE],
        [LSTM_UNIT_SIZE],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
