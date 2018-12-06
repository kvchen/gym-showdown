#!/usr/bin/env python3

from gym import Env, spaces
from typing import Tuple
import logging
import numpy as np
import requests

from .showdown_client import ShowdownClient

logger = logging.getLogger(__name__)


class ShowdownEnv(Env):
    """
    Communicates with the Pokemon Showdown server and exposes actions as an
    OpenAI gym environment.
    """

    NUM_FEATURES = 504
    DEFAULT_ACTION = "default"
    MOVE_ACTIONS = [f"move {slot}" for slot in range(1, 5)]
    SWITCH_ACTIONS = [f"switch {slot}" for slot in range(2, 7)]
    STALL_ACTIONS = ["pass"]

    # Excludes default action! We want the agent to choose the actions.
    ALL_ACTIONS = MOVE_ACTIONS + SWITCH_ACTIONS + STALL_ACTIONS

    def __init__(self, options=None):
        self.num_actions = len(self.ALL_ACTIONS)
        self.action_space = spaces.Discrete(self.num_actions)

        # TODO: Figure out what this observation space should look like
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.NUM_FEATURES,), dtype=np.float
        )

        self.client = ShowdownClient()
        self.options = options or {}

        self.initial_battle_id = None
        self.current_battle = None

    def step(self, action_idx: int):
        """Takes a single step for player 1."""
        assert self.current_battle is not None
        assert action_idx < self.num_actions

        current_battle_id = self.current_battle["id"]
        current_battle_data = self.current_battle["data"]

        # moves = [None, None]
        # for side_idx, side in enumerate(current_battle_data["sides"]):
        #     request = side["currentRequest"]
        #     move_idx = actions[side_idx]
        #     move = self.ALL_ACTIONS[move_idx]

        #     # Ensure that we aren't attempting an invalid move
        #     if request == "move":
        #         pass
        #     elif request == "wait":
        #         move = None
        #     else:
        #         move = self.DEFAULT_ACTION

        #     moves[side_idx] = move

        # Ensure that the agent doesn't try to take an invalid move.
        sides = current_battle_data["sides"]
        request = sides[0]["currentRequest"]

        move = self.ALL_ACTIONS[action_idx]
        if request == "move":
            pass
        elif request == "wait":
            move = None
        else:
            move = self.DEFAULT_ACTION

        payload = self.client.do_move(current_battle_id, move, "default")
        self.current_battle = payload

        battle_data = payload["data"]
        features = self._get_features(battle_data)
        reward = self._get_reward(battle_data)
        is_terminal = self._is_terminal(battle_data)

        return features, reward, is_terminal, {}

    def reset(self):
        self.close()

        payload = self.client.start_battle(self.options)
        initial_battle_id = payload["id"]

        self.initial_battle_id = initial_battle_id
        self.current_battle = payload
        return self._get_features(payload["data"])

    def close(self):
        if self.initial_battle_id is not None:
            self.client.remove_battle(self.initial_battle_id)
            self.initial_battle_id = None
            self.current_battle = None

    # HELPER METHODS

    def _is_terminal(self, battle_data) -> bool:
        return battle_data["ended"]

    def _get_features(self, battle_data):
        # TODO: Add terrain and weather
        side_features = [self._get_side_features(side) for side in battle_data["sides"]]
        return np.concatenate(side_features)

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

        # TODO: Add categorical features
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
                *move_features,
            ]
        )

    def _get_move_features(self, move_data):
        if move_data is None:
            return np.zeros((6,))

        return np.array(
            [
                move_data["accuracy"],
                move_data["basePower"],
                move_data["priority"],
                move_data["pp"],
                move_data["maxpp"],
                move_data["disabled"],
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
