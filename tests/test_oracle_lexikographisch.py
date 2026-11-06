"""Orakel-Test lexikographisch-demo: der Kern gegen einen ANDEREN Rechenweg (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12).

Reines Python (itertools, Tupelvergleich, `fractions.Fraction`, keine numpy-Zeile im Referenzweg) gegen die Aufzählung (int64), die Ungarische Methode,
HiGHS und die Gleitkomma-Auswertung der Demo. Mini-Instanzen n = 3 bis 6, feste Seeds, ganze Zahlen und Zehntel."""

import itertools
from fractions import Fraction

import pytest

import lxk_algorithm as A
import lxk_evaluation as ev
from lxk_scenario import make_instance

scipy_opt = pytest.importorskip("scipy.optimize")
EXACT_FAMILIES = (0, 1, 2)                                                     # ganze Zahlen, Dezimal 1,0 bis 9,9, Dezimal 0,1 bis 0,9


def _all(inst, levels=3):
    """Alle Pläne mit ihren exakten Summen (ganze Zahlen, bei Dezimalfamilien in Zehnteln) in reinem Python."""
    n = inst.n
    return [(plan, tuple(sum(m[i][plan[i]] for i in range(n)) for m in inst.ints[:levels])) for plan in itertools.permutations(range(n))]


def _cases(sizes=(3, 4, 5, 6), seeds=range(6)):
    for n in sizes:
        for family in EXACT_FAMILIES:
            for seed in seeds:
                yield n, family, seed


def test_lexicographic_optimum_matches_tuple_comparison():
    for n, family, seed in _cases():
        inst = make_instance(n, family, seed)
        best = min(_all(inst), key=lambda t: (t[1], t[0]))
        ref = A.lex_reference(inst)
        assert tuple(int(x) for x in ref.plan) == best[0] and ref.vals == best[1], (n, family, seed)


def test_optimal_set_and_b_decision_match_the_definition():
    for n, family, seed in _cases():
        inst = make_instance(n, family, seed)
        plans = _all(inst)
        a_min = min(v[0] for _, v in plans)
        opt = [(p, v) for p, v in plans if v[0] == a_min]
        a = ev.analyse(ev.Settings(n, family, seed))
        assert len(a.opt_set) == len(opt) and a.b_decides == (len({v[1] for _, v in opt}) > 1)


def test_smallest_weight_matches_exact_rational_arithmetic():
    for n, family, seed in _cases(sizes=(3, 4, 5)):
        inst = make_instance(n, family, seed)
        plans = _all(inst, 2)
        ref = min(plans, key=lambda t: (t[1], t[0]))
        need = [Fraction(ref[1][1] - v[1], v[0] - ref[1][0]) for _, v in plans if v[0] > ref[1][0]]
        expected = max([Fraction(0)] + need)
        a = ev.analyse(ev.Settings(n, family, seed))
        assert a.m_star == pytest.approx(float(expected), rel=1e-12), (n, family, seed)


def test_exact_weighted_optimum_is_lexicographic_exactly_when_the_weight_reaches_m_star():
    """In exakter Bruch-Arithmetik: für M > M* gewinnt das lexikographische Optimum streng, für M < M* nicht."""
    for n, family, seed in _cases(sizes=(3, 4, 5), seeds=range(4)):
        inst = make_instance(n, family, seed)
        plans = _all(inst, 2)
        ref = min(plans, key=lambda t: (t[1], t[0]))
        a = ev.analyse(ev.Settings(n, family, seed))
        mstar = Fraction(a.m_star).limit_denominator(10 ** 6)
        for m, expect_lex in ((mstar + 1, True), (mstar + Fraction(1, 1000), True), (mstar - Fraction(1, 1000), False)):
            if m <= 0:
                continue
            best = min(plans, key=lambda t: (m * t[1][0] + t[1][1], t[0]))
            assert (best[1] == ref[1]) == expect_lex, (n, family, seed, m)


def test_hungarian_method_with_the_integer_bound_gives_the_lexicographic_optimum():
    for n, family, seed in _cases(sizes=(3, 4, 5, 6), seeds=range(4)):
        inst = make_instance(n, family, seed)
        ints = inst.int_arrays()
        sums = A.enum_sums(ints, n)
        m = A.int_bound(sums[1])
        plan = A.lap_plan(m * ints[0].astype(float) + ints[1].astype(float))
        vals = A.plan_sums(ints, plan)
        ref = A.lex_reference(inst)
        assert (int(vals[0]), int(vals[1])) == ref.vals[:2], (n, family, seed)


def test_highs_with_the_integer_bound_gives_the_lexicographic_optimum():
    for n, family, seed in _cases(sizes=(3, 4, 5), seeds=range(3)):
        inst = make_instance(n, family, seed)
        ints = inst.int_arrays()
        m = A.int_bound(A.enum_sums(ints, n)[1])
        plan = A.highs_plan(m * ints[0].astype(float) + ints[1].astype(float))
        vals = A.plan_sums(ints, plan)
        assert (int(vals[0]), int(vals[1])) == A.lex_reference(inst).vals[:2], (n, family, seed)


def test_scipy_assignment_matches_the_enumeration_for_a_weighted_sum():
    for n, family, seed in _cases(sizes=(4, 5, 6), seeds=range(3)):
        inst = make_instance(n, family, seed)
        mats = inst.arrays()
        for m in (3.0, 47.0):
            cost = m * mats[0] + mats[1]
            rows, cols = scipy_opt.linear_sum_assignment(cost)
            brute = min(sum(cost[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
            assert cost[rows, cols].sum() == pytest.approx(brute, rel=1e-12)
            idx = A.weighted_plan_enum(mats[:2], (m, 1.0), n)
            assert sum(cost[i][A.all_plans(n)[idx][i]] for i in range(n)) == pytest.approx(brute, rel=1e-12)


def test_sequential_tolerance_matches_pure_python_and_milp():
    for n, family, seed in _cases(sizes=(3, 4, 5), seeds=range(4)):
        inst = make_instance(n, family, seed)
        plans = _all(inst, 2)
        a_star = min(v[0] for _, v in plans)
        mats = inst.arrays()
        for tau in (0.0, 0.05, 0.2):
            allowed = [(p, v) for p, v in plans if v[0] <= a_star * (1 + tau) + 1e-9]
            b_min = min(v[1] for _, v in allowed)
            a = ev.analyse(ev.Settings(n, family, seed))
            idx, count = A.tolerance_pick_enum(a.float_sums[0], a.float_sums[1], tau)
            assert count == len(allowed) and a.exact_sums[1][idx] == b_min, (n, family, seed, tau)
            plan = A.tolerance_pick_milp(mats[0], mats[1], tau)
            assert A.plan_sums([inst.int_arrays()[1]], plan)[0] == b_min, (n, family, seed, tau)


def test_float_evaluation_of_the_weighted_sum_equals_the_exact_one_inside_the_window():
    """Für M zwischen M* und 1e12 stimmt die Gleitkomma-Wahl mit der exakten (Bruch-)Wahl überein."""
    for n, family, seed in _cases(sizes=(4, 5), seeds=range(5)):
        inst = make_instance(n, family, seed)
        plans = _all(inst, 2)
        ref = min(plans, key=lambda t: (t[1], t[0]))
        a = ev.analyse(ev.Settings(n, family, seed))
        for k in (3, 6, 9, 12):
            row = next(r for r in ev.m_grid(a) if r["k"] == k)
            assert row["vals"][:2] == ref[1] and row["status"] == "lex", (n, family, seed, k)
