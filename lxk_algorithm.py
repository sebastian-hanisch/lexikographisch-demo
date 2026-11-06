"""Kern der Demo (numpy, scipy): lexikographische Optimierung der Zuordnung gegen ein Gewicht M und gegen eine Toleranz.

Drei Summenkriterien A (Kosten), B (Zeit), C (Wünsche), alle minimiert. „Zuerst A, dann B“ heißt: unter allen Plänen mit kleinstem A wähle den mit
kleinstem B (dann C). Verglichen wird das mit
  * einem Gewicht: eine Zuordnung mit den Kosten M*A + B (oder A + w*B, oder M^2*A + M*B + C), gerechnet wie ein Löser (Kostenmatrix, Summe über die
    Einträge, float64): `weighted_plan_enum` (aufgezählt, bei Gleichstand der gerechneten Zahl der erste Plan), `lap_plan` (Ungarische Methode),
    `highs_plan` (ganzzahlige Optimierung, HiGHS über scipy.optimize.milp mit den Voreinstellungen);
  * einer Toleranz: erst A minimieren, dann B unter A <= A*(1 + tau): `tolerance_pick_enum`, `tolerance_pick_milp`.
Exakt gerechnet wird in ganzen Zahlen (`enum_sums`, `lex_reference`): Werte mit höchstens einer Nachkommastelle sind als Zehntel gespeichert.
Gleichstandsregeln (sichtbar, siehe README): das lexikographische Optimum ist bei vollem Gleichstand der kleinste Plan in der Reihenfolge von
`itertools.permutations`; das aufgezählte Gewichtsoptimum ebenso, aber nach der gerechneten (float64-)Zahl.
"""

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

import lxk_constants as C


def check_feasible(plan, n: int) -> bool:
    """Unabhängiger Prüfer: der Plan ist eine Zuordnung, jede Aufgabe genau einmal."""
    plan = [int(j) for j in plan]
    return len(plan) == n and sorted(plan) == list(range(n))


@lru_cache(maxsize=None)
def all_plans(n: int) -> np.ndarray:
    """Alle n! Pläne als Zeilen (Reihenfolge von itertools.permutations); nur bis n = 9."""
    if not 1 <= n <= 9:
        raise ValueError(f"n = {n}: Aufzählung nur für 1 bis 9 Personen")
    return np.array(list(itertools.permutations(range(n))), dtype=np.int16)


def plan_sums(mats, plan) -> list:
    """Summe je Matrix für einen Plan (ganze Zahlen bleiben ganze Zahlen)."""
    idx = np.arange(mats[0].shape[0])
    plan = np.asarray(plan, dtype=int)
    return [m[idx, plan].sum() for m in mats]


def enum_sums(mats, n: int) -> list[np.ndarray]:
    """Summe je Matrix für alle Pläne (n <= 9)."""
    idx = np.arange(n)[None, :]
    plans = all_plans(n)
    return [m[idx, plans].sum(axis=1) for m in mats]


def lex_index(sums: list[np.ndarray], levels: int | None = None) -> int:
    """Index des lexikographischen Optimums (erst sums[0], dann sums[1], ...), bei vollem Gleichstand der kleinste Plan."""
    keys = sums if levels is None else sums[:levels]
    return int(np.lexsort(tuple(reversed(keys)))[0])


def optimal_set(first: np.ndarray) -> np.ndarray:
    """Indizes aller Pläne mit kleinstem ersten Kriterium (exakt für ganze Zahlen, bei reellen Werten gilt Gleichheit der Gleitkommazahl)."""
    return np.flatnonzero(first == first.min())


def m_star(a: np.ndarray, b: np.ndarray, ref: int) -> float:
    """Kleinstes M, für das M*A + B am lexikographischen Optimum (Index `ref`) minimal ist: das Maximum über alle Pläne mit A > A* von
    (B* - B) / (A - A*); nie negativ."""
    af, bf = a.astype(float), b.astype(float)
    worse = af > af[ref]
    if not worse.any():
        return 0.0
    return max(0.0, float(((bf[ref] - bf[worse]) / (af[worse] - af[ref])).max()))


def int_bound(b: np.ndarray) -> int:
    """Hinreichende Schranke für ganzzahlige Daten: M = (B-Spanne) + 1 macht M*A + B lexikographisch (A-Schritte sind mindestens 1)."""
    return int(b.max() - b.min()) + 1


def status(vals, ref, exact: bool, levels: int = 2) -> str:
    """„lex“, wenn die Werte `vals` bis zur Stufe `levels` gleich dem lexikographischen Optimum `ref` sind; sonst der Buchstabe der ersten
    Stufe, die schlechter ist („A“: Rangfolge gekippt, „B“: B verloren, „C“: C verloren). Reelle Werte: relative Toleranz C.TOL."""
    for k in range(levels):
        v, r = vals[k], ref[k]
        worse = v > r if exact else v > r + C.TOL * max(1.0, abs(float(r)))
        if worse:
            return "ABC"[k]
    return "lex"


def weighted_plan_enum(mats, coeffs, n: int) -> int:
    """Index des Plans, der sum(coeff * Matrix) in float64 minimiert; gerechnet wie ein Löser: Kostenmatrix bilden, über die Einträge summieren."""
    cost = sum(c * m for c, m in zip(coeffs, mats))
    idx = np.arange(n)[None, :]
    return int(np.argmin(cost[idx, all_plans(n)].sum(axis=1)))


def lap_plan(cost: np.ndarray) -> np.ndarray:
    """Ungarische Methode (scipy.optimize.linear_sum_assignment) auf der Kostenmatrix."""
    rows, cols = linear_sum_assignment(cost)
    plan = np.empty(cost.shape[0], dtype=int)
    plan[rows] = cols
    return plan


def _assignment_constraint(n: int) -> LinearConstraint:
    rows, cols = np.zeros((n, n * n)), np.zeros((n, n * n))
    for i in range(n):
        rows[i, i * n:(i + 1) * n] = 1
        cols[i, i::n] = 1
    return LinearConstraint(np.vstack([rows, cols]), 1, 1)


def highs_plan(cost: np.ndarray, extra: list | None = None) -> np.ndarray | None:
    """Zuordnung als ganzzahlige Optimierung (HiGHS über scipy.optimize.milp, Voreinstellungen); None, wenn unzulässig."""
    n = cost.shape[0]
    res = milp(cost.ravel(), constraints=[_assignment_constraint(n)] + (extra or []), integrality=np.ones(n * n), bounds=Bounds(0, 1))
    return np.round(res.x).reshape(n, n).argmax(axis=1) if res.status == 0 else None


def tolerance_pick_enum(a: np.ndarray, b: np.ndarray, tau: float) -> tuple[int, int]:
    """Sequentiell auf aufgezählten Gleitkommasummen: erlaubt sind alle Pläne mit A <= A*(1 + tau) (Toleranz 0: nur die in float64 gleichen
    Minima), darunter das kleinste B. Rückgabe: (Index des Plans, Zahl der erlaubten Pläne)."""
    allowed = np.flatnonzero(a <= a.min() * (1.0 + tau))
    return int(allowed[np.argmin(b[allowed])]), len(allowed)


def tolerance_pick_milp(a_mat: np.ndarray, b_mat: np.ndarray, tau: float) -> np.ndarray | None:
    """Sequentiell mit HiGHS: erst A minimieren, dann B unter A <= A*(1 + tau) (zusätzlich gilt die Zulässigkeitstoleranz von HiGHS)."""
    first = highs_plan(a_mat)
    a_star = float(plan_sums([a_mat], first)[0])
    limit = a_star * (1.0 + tau) + 1e-9 * max(1.0, a_star)
    return highs_plan(b_mat, [LinearConstraint(a_mat.ravel()[None, :], -np.inf, limit)])


@dataclass
class Reference:
    plan: np.ndarray       # lexikographisches Optimum
    vals: tuple            # exakte Werte (A, B, C) in Zehnteln/ganzen Zahlen bzw. Gleitkomma; C nur bei Aufzählung, sonst None
    exact: bool            # ganzzahlig gerechnet?
    index: int | None      # Index in der Aufzählung (nur n <= ENUM_MAX)


def lex_reference(inst) -> Reference:
    """Das exakte lexikographische Optimum: n <= ENUM_MAX aufgezählt (ganzzahlig, wo möglich), darüber mit der Ungarischen Methode:
    ganze Zahlen mit M = B-Spanne + 1 (beweisbar exakt, solange alle Zahlen unter 2^53 bleiben), reelle Werte nur A (B hat dort nichts zu entscheiden)."""
    n = inst.n
    ints = inst.int_arrays()
    exact = ints is not None
    base = ints if exact else inst.arrays()
    if n <= C.ENUM_MAX:
        sums = enum_sums(base, n)
        k = lex_index(sums)
        return Reference(all_plans(n)[k].astype(int), tuple(s[k].item() for s in sums), exact, k)
    if exact:
        m = n * int(base[1].max() - base[1].min()) + 1
        plan = lap_plan(m * base[0].astype(float) + base[1].astype(float))
    else:
        plan = lap_plan(base[0])
    sums = plan_sums(base, plan)
    return Reference(plan, (sums[0].item(), sums[1].item(), None), exact, None)


def exact_values(inst, plan) -> tuple:
    """Werte (A, B, C) eines Plans in der exakten Darstellung der Instanz (ganze Zahlen in Zehnteln oder Gleitkomma)."""
    ints = inst.int_arrays()
    return tuple(s.item() for s in plan_sums(ints if ints is not None else inst.arrays(), plan))
