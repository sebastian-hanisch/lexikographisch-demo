"""Handwerte: jede Einheit des Kerns an einem Beispiel, das von Hand nachgerechnet ist (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import lxk_algorithm as A
import lxk_evaluation as ev
from lxk_scenario import make_instance

PLANS = {(0, 1, 2): (9, 10), (0, 2, 1): (9, 16), (1, 0, 2): (8, 14), (1, 2, 0): (9, 15), (2, 0, 1): (8, 18), (2, 1, 0): (9, 13)}


def _enum(hand):
    ints = hand.int_arrays()
    return ints, A.enum_sums(ints, 3)


def test_hand_example_all_six_plans(hand):
    ints, sums = _enum(hand)
    got = {tuple(int(x) for x in p): (int(sums[0][k]), int(sums[1][k])) for k, p in enumerate(A.all_plans(3))}
    assert got == PLANS


def test_hand_example_optimal_set_and_lexicographic_optimum(hand):
    ints, sums = _enum(hand)
    opt = A.optimal_set(sums[0])
    assert [tuple(int(x) for x in A.all_plans(3)[k]) for k in opt] == [(1, 0, 2), (2, 0, 1)]
    k = A.lex_index(sums)
    assert tuple(int(x) for x in A.all_plans(3)[k]) == (1, 0, 2) and (sums[0][k], sums[1][k]) == (8, 14)
    ref = A.lex_reference(hand)
    assert tuple(ref.plan) == (1, 0, 2) and ref.vals[:2] == (8, 14) and ref.exact and ref.index == k


def test_hand_example_smallest_weight_is_four_with_a_tie_at_exactly_four(hand):
    ints, sums = _enum(hand)
    ref = A.lex_index(sums)
    assert A.m_star(sums[0], sums[1], ref) == 4.0 and A.int_bound(sums[1]) == 9
    mats = hand.arrays()
    status = {m: A.status(tuple(s[A.weighted_plan_enum(mats[:2], (m, 1.0), 3)].item() for s in sums), (8, 14), True) for m in (1.0, 3.0, 4.0, 5.0, 9.0, 100.0)}
    # M = 3: Plan (0,1,2) gewinnt (37 gegen 38); M = 4: beide 46, die Aufzählung wählt den ersten Plan (0,1,2); ab M = 5 gilt das lexikographische Optimum
    assert status == {1.0: "A", 3.0: "A", 4.0: "A", 5.0: "lex", 9.0: "lex", 100.0: "lex"}
    assert [(3 * a + b) for a, b in ((8, 14), (9, 10))] == [38, 37] and [(4 * a + b) for a, b in ((8, 14), (9, 10))] == [46, 46]


def test_hand_example_sequential_tolerance(hand):
    ints, sums = _enum(hand)
    fa, fb = (s.astype(float) for s in sums[:2])
    for tau, expected, allowed in ((0.0, (8, 14), 2), (0.1, (8, 14), 2), (0.125, (9, 10), 6)):
        idx, count = A.tolerance_pick_enum(fa, fb, tau)
        assert (int(sums[0][idx]), int(sums[1][idx])) == expected and count == allowed, tau
    assert 1 - 10 / 14 == pytest.approx(4 / 14)                                     # Gewinn in B bei tau = 1/8: 28,6 %


def test_hand_example_solvers_agree_with_the_enumeration(hand):
    mats = hand.arrays()
    plan = A.lap_plan(9 * mats[0] + mats[1])
    assert tuple(int(x) for x in plan) == (1, 0, 2)
    assert tuple(int(x) for x in A.highs_plan(9 * mats[0] + mats[1])) == (1, 0, 2)
    plan = A.tolerance_pick_milp(mats[0], mats[1], 0.125)
    assert A.check_feasible(plan, 3) and tuple(int(x) for x in plan) == (0, 1, 2)
    assert tuple(int(x) for x in A.tolerance_pick_milp(mats[0], mats[1], 0.0)) == (1, 0, 2)


def test_float_facts_behind_the_window():
    assert float(2 ** 53) + 1 == float(2 ** 53) and 0.1 + 0.2 != 0.3
    assert 0.1 + 0.2 == pytest.approx(0.3) and 9 * 1e16 > 2 ** 53 and 9 * 1e15 < 2 ** 53


def test_hand_example_the_window_in_float64(hand):
    a = ev.analyse(ev.Settings(3, 0, 6))                                            # n = 3, ganze Zahlen, Seed 6: M* = 4
    assert a.m_star == 4.0 and a.m_bound == 12 and len(a.opt_set) == 2 and a.b_decides
    rows = ev.m_grid(a)
    ok = [r["k"] for r in rows if r["status"] == "lex"]
    assert min(ok) == 1 and 0 not in ok and rows[0]["status"] == "A"                # M = 1 kippt, M = 10 stimmt


def test_status_orders_the_levels_and_uses_a_relative_tolerance_for_reals():
    assert A.status((8, 14), (8, 14), True) == "lex" and A.status((9, 10), (8, 14), True) == "A" and A.status((8, 18), (8, 14), True) == "B"
    assert A.status((8, 14, 5), (8, 14, 3), True, levels=3) == "C" and A.status((8, 14, 5), (8, 14, 3), True) == "lex"
    assert A.status((10.0 + 1e-12, 5.0), (10.0, 5.0), False) == "lex" and A.status((10.1, 5.0), (10.0, 5.0), False) == "A"


def test_exact_values_use_the_integer_representation_of_decimal_families():
    inst = make_instance(4, 2, 7)                                                    # Dezimalzahlen 0,1 bis 0,9 in Zehnteln
    plan = (0, 1, 2, 3)
    vals = A.exact_values(inst, plan)
    assert all(isinstance(v, int) for v in vals) and vals[0] == int(round(sum(inst.mats[0][i][plan[i]] for i in range(4)) * 10))
    assert all(isinstance(v, float) for v in A.exact_values(make_instance(4, 3, 7), plan))


def test_check_feasible_and_enumeration_limits():
    assert A.check_feasible((2, 0, 1), 3) and not A.check_feasible((0, 0, 1), 3) and not A.check_feasible((0, 1), 3)
    with pytest.raises(ValueError):
        A.all_plans(10)
    assert len(A.all_plans(5)) == 120
