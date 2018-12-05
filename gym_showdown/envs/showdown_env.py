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

    MOVE_ACTIONS = [f"move {slot}" for slot in range(1, 5)]
    SWITCH_ACTIONS = [f"switch {slot}" for slot in range(2, 7)]
    ALL_ACTIONS = MOVE_ACTIONS + SWITCH_ACTIONS

    def __init__(self, options={}):
        self.num_actions = len(self.ALL_ACTIONS)
        self.action_space = spaces.Tuple(
            (spaces.Discrete(self.num_actions) for _ in range(2))
        )

        # TODO: Figure out what this observation space should look like
        self.observation_space = spaces.Discrete(4)

        self.client = ShowdownClient()
        self.options = options

        self.initial_battle_id = None
        self.current_battle = None

    def step(self, actions: Tuple[int, int]):
        assert self.current_battle is not None
        assert all(action < self.num_actions for action in actions)

        sides = self.current_battle.sides
        p1_request, p2_request = sides[0].request, sides[1].request

        battle_id = self.current_battle["id"]
        p1_action, p2_action = actions
        p1_move, p2_move = self.ALL_ACTIONS[p1_action], self.ALL_ACTIONS[p2_action]

        payload = self.client.do_move(battle_id, p1_move, p2_move)
        self.current_battle = payload

        battle_data = payload["data"]
        features = self._get_features(battle_data)
        reward = self._get_reward(battle_data)
        is_terminal = self._is_terminal(battle_data)

        return features, reward, is_terminal

    def reset(self):
        self.close()

        payload = self.client.start_battle(self.options)
        initial_battle_id = payload["id"]

        self.initial_battle_id = initial_battle_id
        self.current_battle = payload
        return payload

    def close(self):
        if self.initial_battle_id is not None:
            self.client.remove_battle(self.initial_battle_id)
            self.initial_battle_id = None
            self.current_battle = None

    # HELPER METHODS

    def _is_terminal(self, battle_data):
        return battle_data["ended"]

    def _get_features(self, battle_data):
        """
        Returns a flat n-dimensional payload.

        TODO: Document feature indices here
        """
        return np.array([])

    def _get_reward(self, battle_data):
        """
        """
        if not self._is_terminal(battle_data) or "winner" not in battle_data:
            return 0

        return 0

    def _has_choice_error(self, battle_data):
        pass
