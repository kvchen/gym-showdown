#!/usr/bin/env python3

MOVE_ACTIONS = [f"move {slot}" for slot in range(1, 5)]
SWITCH_ACTIONS = [f"switch {slot}" for slot in range(2, 7)]
STALL_ACTIONS = ["pass"]
NOOP_ACTIONS = [None]

ALL_ACTIONS = MOVE_ACTIONS + SWITCH_ACTIONS + STALL_ACTIONS + NOOP_ACTIONS

TERRAINS = ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
WEATHERS = [
    "raindance",
    "primordialsea",
    "sunnyday",
    "desolateland",
    "sandstorm",
    "hail",
    "deltastream",
]
STATUSES = ["brn", "par", "slp", "frz", "psn", "tox"]
GENDERS = ["M", "F", "N"]
TYPES = [
    "Bug",
    "Datk",
    "Dragon",
    "Electric",
    "Fairy",
    "Fighting",
    "Fire",
    "Flying",
    "Ghost",
    "Grass",
    "Ground",
    "Ice",
    "Normal",
    "Poison",
    "Psychic",
    "Rock",
    "Steel",
    "Water",
]
CATEGORIES = ["Physical", "Special", "Status"]
TARGETS = [
    "all",
    "foeSide",
    "allySide",
    "allyTeam",
    "allAdjacent",
    "allAdjacentFoes",
    "normal",
    "self",
    "any",
    "scripted",
    "adjacentAlly",
    "adjacentFoe",
    "adjacentAllyOrSelf",
    "randomNormal",
]
