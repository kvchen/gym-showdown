#!/usr/bin/env python3

from typing import Optional
import requests

from urllib.parse import urljoin


class PlayerChoiceException(Exception):
    def __init__(self, player: int, message: str):
        super().__init__(message)
        self.player = player


class ShowdownClient:
    """
    Talks with a local Pokemon Showdown server to simulate games.

    See https://github.com/kvchen/showdown-rl-server
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 3000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def start_battle(self, options):
        start_url = urljoin(self.base_url, "start")
        return requests.post(start_url, data=options).json()

    def do_move(self, battle_id: str, p1_move: Optional[str], p2_move: Optional[str]):
        move_url = "/".join([self.base_url, battle_id, "move"])
        response = requests.post(move_url, data={"p1Move": p1_move, "p2Move": p2_move})
        return response.json()

        # if not response.ok:
        #     error = payload["error"]
        #     if error["type"] == "invalid_player_choice":
        #         raise PlayerChoiceException(
        #             error["player"], f"Invalid p{error['player']} choice"
        #         )

        # return payload

    def remove_battle(self, battle_id: str) -> None:
        battle_url = "/".join([self.base_url, battle_id])
        requests.delete(battle_url)
