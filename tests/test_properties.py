"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jede Instanz gelten müssen, an festen Seeds und allen vier Familien."""

import numpy as np
import pytest

import lxk_algorithm as A
import lxk_constants as C
import lxk_evaluation as ev
from lxk_scenario import make_instance

CASES = [(n, family, seed) for n in (3, 5, 7, 8) for family in range(4) for seed in range(4)]


def _analysis(n, family, seed):
    return ev.analyse(ev.Settings(n, family, seed))


@pytest.mark.parametrize("n,family,seed", CASES)
def test_lexicographic_optimum_is_a_optimal_has_the_smallest_b_among_them_and_is_pareto_optimal(n, family, seed):
    a = _analysis(n, family, seed)
    sa, sb = a.exact_sums[0], a.exact_sums[1]
    k = a.ref.index
    assert sa[k] == sa.min() and k in set(a.opt_set) and sb[k] == sb[a.opt_set].min()
    assert not any(sa[j] <= sa[k] and sb[j] <= sb[k] and (sa[j], sb[j]) != (sa[k], sb[k]) for j in range(len(sa)))


@pytest.mark.parametrize("n,family,seed", CASES)
def test_m_star_is_the_threshold_and_never_exceeds_the_integer_bound(n, family, seed):
    a = _analysis(n, family, seed)
    sa, sb = a.exact_sums[0].astype(float), a.exact_sums[1].astype(float)
    k = a.ref.index
    assert a.m_star >= 0.0
    for m, expect in ((a.m_star * 1.001 + 1e-9, True), (a.m_star * 0.999 - 1e-9, False)):
        if m <= 0:
            continue
        best = int(np.argmin(m * sa + sb))
        assert ((sa[best], sb[best]) == (sa[k], sb[k])) == expect, (m, a.m_star)
    if a.m_bound is not None:
        assert a.m_star <= a.m_bound - 1 + 1e-12                                       # M* <= B-Spanne


@pytest.mark.parametrize("n,family,seed", [(5, 0, 1), (7, 1, 2), (8, 3, 3), (6, 2, 4)])
def test_scaling_b_scales_m_star_and_the_optimal_set_is_unchanged(n, family, seed):
    inst = make_instance(n, family, seed)
    mats = inst.arrays()
    sa, sb = A.enum_sums(mats, n)[:2]
    ref = A.lex_index([sa, sb])
    base = A.m_star(sa, sb, ref)
    for c in (0.5, 3.0, 10.0):
        scaled = A.m_star(sa, sb * c, A.lex_index([sa, sb * c]))
        assert scaled == pytest.approx(base * c, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("n,family,seed", CASES)
def test_tolerance_zero_is_lexicographic_in_exact_numbers_and_the_allowed_set_grows(n, family, seed):
    a = _analysis(n, family, seed)
    rows = ev.tau_rows(a)
    allowed = [r["allowed"] for r in rows]
    assert allowed == sorted(allowed) and allowed[0] >= 1
    if a.inst.scale in (0, 1):                                                          # ganze Zahlen und reelle Werte: kein Rundungsrauschen
        assert rows[0]["same"]
        assert all(r["a_loss"] >= -1e-12 and r["b_gain"] >= -1e-12 for r in rows)
        assert all(rows[i]["vals"][1] >= rows[i + 1]["vals"][1] for i in range(len(rows) - 1))   # mehr Toleranz: B nie schlechter
    else:                                                                               # Zehntel: ab Toleranz 1e-9 ist der Gleichstand wieder da
        assert rows[2]["same"] or not a.b_decides or rows[2]["a_loss"] >= 0
        assert all(rows[i]["vals"][1] >= rows[i + 1]["vals"][1] for i in range(2, len(rows) - 1))


@pytest.mark.parametrize("n,family,seed", CASES)
def test_weight_grid_is_a_window_for_integer_families(n, family, seed):
    """Unten kippt die Rangfolge (A), oben nur bei ganzen Zahlen mit Gleichstand B; dazwischen stimmt es."""
    a = _analysis(n, family, seed)
    rows = ev.m_grid(a)
    ok = [r["k"] for r in rows if r["status"] == "lex"]
    assert ok and 12 in ok                                                              # M = 10^12 ist für n <= 8 immer richtig
    assert all(r["status"] in ("lex", "A", "B") for r in rows)
    below = [r for r in rows if r["k"] < min(ok)]
    assert all(r["status"] == "A" for r in below)
    for r in rows:
        if r["k"] > 12 and r["status"] != "lex":
            assert r["status"] == "B" and a.b_decides                                   # nur wo B gebraucht wird, kann B verloren gehen


@pytest.mark.parametrize("n,family,seed", [(5, 0, 1), (6, 0, 2), (4, 1, 3), (7, 2, 4)])
def test_permuting_rows_and_columns_of_all_matrices_keeps_the_lexicographic_values(n, family, seed):
    inst = make_instance(n, family, seed)
    ints = inst.int_arrays()
    rng = np.random.default_rng(seed + 70)
    r, c = rng.permutation(n), rng.permutation(n)
    s1 = A.enum_sums(ints, n)
    s2 = A.enum_sums([m[r][:, c] for m in ints], n)
    assert tuple(x[A.lex_index(s1)] for x in s1) == tuple(x[A.lex_index(s2)] for x in s2)


def test_float_window_top_follows_two_to_the_53():
    """M*A + B verliert B, sobald M*a_max über 2^53 liegt: bei ganzen Zahlen bis 9 ab etwa M = 10^15,95."""
    a = ev.analyse(ev.Settings(8, 0, 2))
    rows = {r["k"]: r["status"] for r in ev.m_grid(a)}
    assert rows[15] == "lex" and rows[16] == "B" and 9 * 10.0 ** 15 < C.FLOAT_LIMIT < 9 * 10.0 ** 16


def test_everything_is_deterministic():
    s = ev.Settings(8, 0, 11)
    a1, a2 = ev.analyse(s), ev.analyse(s)
    assert a1.ref.vals == a2.ref.vals and (a1.opt_set == a2.opt_set).all() and a1.m_star == a2.m_star
    assert make_instance(5, 0, 3) == make_instance(5, 0, 3) and make_instance(5, 0, 3) != make_instance(5, 0, 4)
    assert [r["status"] for r in ev.m_grid(a1)] == [r["status"] for r in ev.m_grid(a2)]
