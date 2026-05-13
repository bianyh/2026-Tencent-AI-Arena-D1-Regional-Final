#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
import random
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None

try:
    import toml
except ImportError:  # pragma: no cover
    toml = None


class EnvConfManager:
    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.usr_conf = None
        self.raw_conf = {}
        self.episode_cnt = 0
        self.eval_interval = 0
        self.random_eval_start = 0
        self.default_opponent_agent = None
        self.eval_opponent_types = []
        self.auto_switch_monitor_side = False
        self.monitor_side = 0
        self.initialize()

    def initialize(self):
        self.raw_conf = self._read_raw_conf()
        self.usr_conf = self._validate_usr_conf(self.raw_conf)

        if self.usr_conf is None:
            raise ValueError("usr_conf is None, please check the configuration file")

        episode_conf = self.usr_conf["episode"]
        self.eval_interval = episode_conf["eval_interval"] + 1
        self.default_opponent_agent = episode_conf["opponent_agent"]
        self.eval_opponent_types = self._build_eval_opponent_types()

        self.auto_switch_monitor_side = self.usr_conf["monitor"]["auto_switch_monitor_side"]
        self.monitor_side = self.usr_conf["monitor"]["monitor_side"]
        self.random_eval_start = random.randint(0, self.eval_interval) if self.eval_interval != 0 else 0

    def _read_raw_conf(self):
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"env config file not found: {self.config_path}")
        if tomllib is not None:
            with config_file.open("rb") as file_obj:
                return tomllib.load(file_obj)
        if toml is not None:
            return toml.load(str(config_file))
        raise RuntimeError("No TOML parser available. Need Python 3.11+ tomllib or the toml package.")

    def _validate_usr_conf(self, usr_conf):
        monitor_conf = usr_conf.setdefault("monitor", {})
        monitor_conf["monitor_side"] = int(monitor_conf.get("monitor_side", 0))
        monitor_conf["auto_switch_monitor_side"] = bool(monitor_conf.get("auto_switch_monitor_side", True))
        if monitor_conf["monitor_side"] not in (0, 1):
            raise ValueError("monitor.monitor_side must be 0 or 1")

        episode_conf = usr_conf.setdefault("episode", {})
        episode_conf["opponent_agent"] = str(episode_conf.get("opponent_agent", "selfplay"))
        episode_conf["eval_interval"] = int(episode_conf.get("eval_interval", 10))
        episode_conf["eval_opponent_type"] = str(episode_conf.get("eval_opponent_type", "common_ai"))
        episode_conf["eval_include_model_pool"] = bool(episode_conf.get("eval_include_model_pool", False))
        if episode_conf["eval_interval"] < 1:
            raise ValueError("episode.eval_interval must be >= 1")

        lineups = usr_conf.setdefault("lineups", {})
        for camp_key in ("blue_camp", "red_camp"):
            camp_lineup = lineups.get(camp_key)
            if not camp_lineup or not isinstance(camp_lineup, list):
                raise ValueError(f"lineups.{camp_key} must contain one hero config")
            for hero_conf in camp_lineup:
                hero_conf["hero_id"] = int(hero_conf["hero_id"])

        return usr_conf

    def _build_eval_opponent_types(self):
        episode_conf = self.usr_conf["episode"]
        opponents = episode_conf.get("eval_opponent_types")
        if not opponents:
            opponents = [episode_conf.get("eval_opponent_type", "common_ai")]
        if isinstance(opponents, str):
            opponents = [opponents]

        opponents = [str(opponent) for opponent in opponents if str(opponent)]
        if episode_conf.get("eval_include_model_pool", False):
            opponents.extend(self._read_model_pool_ids())

        deduped = []
        seen = set()
        for opponent in opponents:
            if opponent in seen:
                continue
            seen.add(opponent)
            deduped.append(opponent)

        if not deduped:
            deduped = ["common_ai"]
        episode_conf["eval_opponent_types"] = deduped
        return deduped

    def _read_model_pool_ids(self):
        kaiwu_json = Path("kaiwu.json")
        if not kaiwu_json.exists():
            return []
        try:
            data = json.loads(kaiwu_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if self.logger:
                self.logger.warning("kaiwu.json is not valid json, skip model_pool opponents")
            return []
        return [str(model_id) for model_id in data.get("model_pool", [])]

    def get_current_config(self):
        return self.usr_conf

    def get_monitor_side(self):
        return self.monitor_side

    def get_opponent_agent(self):
        return self.usr_conf["episode"]["opponent_agent"]

    def update_config(self, lineup=None):
        if lineup:
            if len(lineup) == 2 and all(isinstance(hero_id, int) for hero_id in lineup[:2]):
                self.usr_conf["lineups"]["blue_camp"][0]["hero_id"] = lineup[0]
                self.usr_conf["lineups"]["red_camp"][0]["hero_id"] = lineup[1]
            else:
                raise ValueError("Invalid lineup format, expected list of 2 integers")

        if self.auto_switch_monitor_side:
            self.monitor_side = 1 - self.monitor_side
        self.usr_conf["monitor"]["monitor_side"] = self.monitor_side

        is_eval = (self.episode_cnt + self.random_eval_start) % self.eval_interval == 0
        if is_eval:
            self.usr_conf["episode"]["eval_opponent_type"] = random.choice(self.eval_opponent_types)

        opponent_agent = (
            self.default_opponent_agent
            if not is_eval
            else self.usr_conf["episode"]["eval_opponent_type"]
        )
        self.usr_conf["episode"]["opponent_agent"] = opponent_agent

        self.episode_cnt += 1
        return self.get_current_config(), is_eval, self.get_monitor_side()

    @staticmethod
    def extract_hero_ids_from_usr_conf(usr_conf):
        blue_hero_ids = [hero["hero_id"] for hero in usr_conf["lineups"]["blue_camp"]]
        red_hero_ids = [hero["hero_id"] for hero in usr_conf["lineups"]["red_camp"]]
        return blue_hero_ids, red_hero_ids

    @staticmethod
    def inject_select_skills(usr_conf, camp_key, select_skills):
        for hero_conf in usr_conf["lineups"][camp_key]:
            hero_id = hero_conf["hero_id"]
            if hero_id in select_skills:
                hero_conf["summoner_skill_id"] = select_skills[hero_id]
