#!/usr/bin/env python3

import gym
import requests
import logging

from .showdown_client import ShowdownClient

logger = logging.getLogger(__name__)


class ShowdownEnv(gym.Env):
    """
    Communicates with the Pokemon Showdown server and exposes actions as an
    OpenAI gym environment.
    """

    def __init__(self):
        self.client = ShowdownClient()
        self.reset()

    def step(self, action):
        pass

    def reset(self):
        pass
