"""Gemeinsame Fixtures: das Handbeispiel (n = 3), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lxk_scenario import Instance  # noqa: E402

HAND_A = ((3, 2, 3), (1, 1, 2), (5, 4, 5))
HAND_B = ((4, 5, 7), (5, 2, 6), (4, 6, 4))
HAND_C = ((1, 3, 2), (2, 1, 3), (3, 2, 1))


def make_hand() -> Instance:
    ints = (HAND_A, HAND_B, HAND_C)
    return Instance(tuple(tuple(tuple(float(x) for x in row) for row in m) for m in ints), ints, 1)


@pytest.fixture
def hand():
    """Handbeispiel n = 3 (A Kosten, B Zeit, C Wünsche; Zeile Person, Spalte Aufgabe). Sechs Pläne mit (A, B):
    (0,1,2): (9, 10); (0,2,1): (9, 16); (1,0,2): (8, 14); (1,2,0): (9, 15); (2,0,1): (8, 18); (2,1,0): (9, 13).
    A-optimal: (1,0,2) und (2,0,1); lexikographisch: (1,0,2) mit (8, 14); M* = 4 (Plan (0,1,2): (14 - 10) / (9 - 8))."""
    return make_hand()


@pytest.fixture
def hand_arrays(hand) -> list[np.ndarray]:
    return hand.arrays()
