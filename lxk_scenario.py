"""Szenario: n Personen, n Aufgaben, drei Belastungen je Paar: A (Kosten), B (Zeit), C (Wünsche), drei unabhängige Matrizen. Alles simuliert.

Ganze Zahlen und Dezimalzahlen mit einer Nachkommastelle sind zusätzlich als ganze Zahlen (in Zehnteln) gespeichert: damit lässt sich „gleich“ exakt
prüfen, auch wenn die Gleitkommasumme zweier gleicher Dezimalsummen sich in der letzten Stelle unterscheidet."""

from dataclasses import dataclass

import numpy as np

import lxk_constants as C


@dataclass(frozen=True)
class Instance:
    mats: tuple                    # drei Matrizen (A, B, C) als Tupel von Tupeln, Gleitkomma
    ints: tuple | None             # dieselben Werte als ganze Zahlen (mal INT_SCALE) oder None (reell)
    scale: int                     # INT_SCALE der Familie (0 bei reell)

    @property
    def n(self) -> int:
        return len(self.mats[0])

    def arrays(self) -> list[np.ndarray]:
        return [np.array(m, dtype=float) for m in self.mats]

    def int_arrays(self) -> list[np.ndarray] | None:
        return None if self.ints is None else [np.array(m, dtype=np.int64) for m in self.ints]


def _tup(m) -> tuple:
    return tuple(tuple(x.item() if hasattr(x, "item") else x for x in row) for row in m)


def make_instance(n: int, family: int, seed: int) -> Instance:
    """Deterministisch aus (n, Familie, Seed): gleiche Eingabe, gleiche Instanz.

    Familie 0: ganze Zahlen 1 bis 9; 1: Dezimalzahlen 1,0 bis 9,9 (Schritt 0,1); 2: Dezimalzahlen 0,1 bis 0,9; 3: reell gleichverteilt 0 bis 100."""
    rng = np.random.default_rng(seed)
    key = C.FAMILY_KEYS[family]
    if key == "int":
        ints = [rng.integers(1, 10, (n, n)).astype(np.int64) for _ in range(3)]
        return Instance(tuple(_tup(m.astype(float)) for m in ints), tuple(_tup(m) for m in ints), 1)
    if key == "dec":
        ints = [rng.integers(10, 100, (n, n)).astype(np.int64) for _ in range(3)]
    elif key == "tenth":
        ints = [rng.integers(1, 10, (n, n)).astype(np.int64) for _ in range(3)]
    else:
        return Instance(tuple(_tup(rng.uniform(0.0, 100.0, (n, n))) for _ in range(3)), None, 0)
    return Instance(tuple(_tup(m / 10.0) for m in ints), tuple(_tup(m) for m in ints), 10)
