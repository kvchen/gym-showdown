#!/usr/bin/env python3

from gym import Env, spaces
from typing import Tuple
import logging
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder

from .showdown_client import ShowdownClient
from .data import ALL_ACTIONS, TERRAINS, WEATHERS, STATUSES, GENDERS, TYPES, CATEGORIES, TARGETS

logger = logging.getLogger(__name__)


def fit_ohe(categories):
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        return ohe.fit(np.reshape(categories, (-1, 1)))

class ShowdownEnv(Env):
    """
    Communicates with the Pokemon Showdown server and exposes actions as an
    OpenAI gym environment.
    """

    NUM_FEATURES = 2519

    def __init__(self, opp_agent, options=None):
        self.action_space = spaces.Discrete(len(ALL_ACTIONS))

        # TODO: Figure out what this observation space should look like
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.NUM_FEATURES,), dtype=np.float
        )

        self.client = ShowdownClient()
        self.opp_agent = opp_agent
        self.options = options or {}

        self.initial_battle_id = None
        self.current_battle = None

        self.terrain_ohe = fit_ohe(TERRAINS)
        self.weather_ohe = fit_ohe(WEATHERS)
        self.status_ohe = fit_ohe(STATUSES)
        self.gender_ohe = fit_ohe(GENDERS)
        self.type_ohe = fit_ohe(TYPES)
        self.category_ohe = fit_ohe(CATEGORIES)
        self.target_ohe = fit_ohe(TARGETS)

    def render(self, mode):
        if mode == "ansi":
            log = self.current_battle["data"]["inputLog"]
            return "\n".join(log)
        else:
            super().render(mode=mode)

    def step(self, action_idx: int):
        assert self.current_battle is not None

        current_battle_id = self.current_battle["id"]
        opp_move_idx = self.opp_agent(self)

        move_idxs = [action_idx, opp_move_idx]
        moves = [self.get_move(move_idx) for move_idx in move_idxs]

        payload = self.client.do_move(current_battle_id, *moves)
        self.current_battle = payload

        sides = payload["data"]["sides"]
        assert not sides[0]["choiceError"], sides[0]["choiceError"]
        assert not sides[1]["choiceError"], sides[1]["choiceError"]

        battle_data = payload["data"]

        features = self._get_features(battle_data, payload["actions"])
        reward = self._get_reward(battle_data)
        is_terminal = self._is_terminal(battle_data)

        return features, reward, is_terminal, {}

    def reset(self):
        self.close()

        payload = self.client.start_battle(self.options)
        initial_battle_id = payload["id"]

        self.initial_battle_id = initial_battle_id
        self.current_battle = payload
        return self._get_features(payload["data"], payload["actions"])

    def seed(self, seed):
        if seed is not None:
            self.options[seed] = seed
        else:
            seed = self.current_battle["seed"]

        return [seed]

    def close(self):
        if self.initial_battle_id is not None:
            self.client.remove_battle(self.initial_battle_id)
            self.initial_battle_id = None
            self.current_battle = None

    # HELPER METHODS

    def get_move(self, move_idx):
        assert move_idx < len(ALL_ACTIONS)
        return ALL_ACTIONS[move_idx]

    def _is_terminal(self, battle_data) -> bool:
        return battle_data["ended"]

    def _get_features(self, battle_data, battle_actions):
        # TODO: Add terrain and weather
        terrain_onehot = self.terrain_ohe.transform([battle_data["terrain"]])
        weather_onehot = self.weather_ohe.transform([battle_data["weather"]])
        side_features = [self._get_side_features(side) for side in battle_data["sides"]]
        features = np.concatenate([*terrain_onehot, *weather_onehot, *side_features])

        # Mask out valid actions
        action_mask = np.zeros(len(ALL_ACTIONS))
        action_mask[battle_actions[0]] = 1
        return features, action_mask[None, :]

    def _get_side_features(self, side_data):
        pokemon_features = [
            self._get_pokemon_features(pokemon) for pokemon in side_data["pokemon"]
        ]
        return np.concatenate(pokemon_features)

    def _get_pokemon_features(self, pokemon_data):
        moves = pokemon_data["moves"]
        move_features = [
            self._get_move_features(move) for move in moves + [None] * (4 - len(moves))
        ]
        stats = pokemon_data["stats"]
        boosts = [(boost + 6) / 12 for boost in pokemon_data["boosts"].values()]

        status_onehot = self.status_ohe.transform([pokemon_data["status"]])
        gender_onehot = self.gender_ohe.transform([pokemon_data["gender"]])
        type_onehot = self.type_ohe.transform([[typ] for typ in pokemon_data["types"]])
        type_onehot = np.sum(type_onehot, axis=0)

        # TODO: Add speciesnum, abilitynum, itemnum
        return np.concatenate(
            [
                [
                    pokemon_data["hp"],
                    pokemon_data["maxhp"],
                    pokemon_data["fainted"],
                    pokemon_data["active"],
                    pokemon_data["happiness"],
                    pokemon_data["level"],
                    # Stats
                    stats["atk"],
                    stats["def"],
                    stats["spa"],
                    stats["spd"],
                    stats["spe"],
                    # Boosts
                    *boosts,
                ],
                *status_onehot,
                *gender_onehot,
                type_onehot,
                *move_features,
            ]
        )

    def _get_move_features(self, move_data):
        if move_data is None:
            return np.zeros((41,))

        accuracy = 100 if type(move_data["accuracy"]) == bool else move_data["accuracy"]
        category_onehot = self.category_ohe.transform([move_data["category"]])
        target_onehot = self.target_ohe.transform([move_data["target"]])
        type_onehot = self.type_ohe.transform([move_data["type"]])

        # TODO: Add movenum
        return np.concatenate(
            [
                [
                    accuracy,
                    move_data["basePower"],
                    move_data["priority"],
                    move_data["pp"],
                    move_data["maxpp"],
                    move_data["disabled"],
                ],
                *category_onehot,
                *target_onehot,
                *type_onehot,
            ]
        )

    def _get_reward(self, battle_data):
        """
        """
        if not self._is_terminal(battle_data) or "winner" not in battle_data:
            return 0

        return 1 if battle_data["winner"] == "Player 1" else -1

    def _has_choice_error(self, battle_data):
        pass
