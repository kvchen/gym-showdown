#!/usr/bin/env python3

import gym


class ShowdownEnv(gym.Env):
    """
    Communicates with the Pokemon Showdown server and exposes actions as an
    OpenAI gym environment.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human", close=False):
        pass
