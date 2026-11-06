"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import lxk_algorithm as A
import lxk_constants as C
import lxk_evaluation as ev
from lxk_scenario import make_instance


def test_a_single_person_has_the_only_plan():
    inst = make_instance(1, 0, 3)
    ints = inst.int_arrays()
    sums = A.enum_sums(ints, 1)
    assert len(A.all_plans(1)) == 1 and A.lex_index(sums) == 0 and A.m_star(sums[0], sums[1], 0) == 0.0
    assert A.lex_reference(inst).vals[0] == ints[0][0, 0]


def test_two_persons_have_at_most_two_plans_and_a_consistent_status():
    for seed in range(10):
        a = ev.analyse(ev.Settings(3, 0, seed))
        assert 1 <= len(a.opt_set) <= 6 and all(r["status"] in ("lex", "A", "B") for r in ev.m_grid(a))
    sums = A.enum_sums([np.array([[1, 2], [3, 1]]), np.array([[5, 1], [2, 4]])], 2)
    assert A.lex_index(sums) == 0 and list(A.optimal_set(sums[0])) == [0]


def test_identical_matrices_make_every_weight_lexicographic_and_the_optimal_set_the_whole_front():
    ints = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    sums = A.enum_sums([ints, ints], 3)
    ref = A.lex_index(sums)
    assert A.m_star(sums[0], sums[1], ref) == 0.0 and (sums[0] == sums[1]).all()
    mats = [ints.astype(float), ints.astype(float)]
    assert all(A.weighted_plan_enum(mats, (10.0 ** k, 1.0), 3) == 0 or sums[0][A.weighted_plan_enum(mats, (10.0 ** k, 1.0), 3)] == sums[0][ref] for k in range(19))


def test_all_entries_equal_make_every_plan_optimal_and_every_weight_lexicographic():
    c = np.full((4, 4), 5.0)
    sums = A.enum_sums([c, c, c], 4)
    assert len(A.optimal_set(sums[0])) == 24 and A.m_star(sums[0], sums[1], 0) == 0.0
    idx, allowed = A.tolerance_pick_enum(sums[0], sums[1], 0.0)
    assert idx == 0 and allowed == 24


def test_integers_with_many_ties_still_give_a_defined_statuses():
    for seed in range(40):
        a = ev.analyse(ev.Settings(6, 0, seed))
        assert a.opt_set.size >= 1 and a.float_ties == len(a.opt_set)                      # ganze Zahlen: float64 erkennt jeden Gleichstand
        assert all(r["status"] in ("lex", "A", "B") for r in ev.m_grid(a))


def test_seed_zero_and_the_largest_seed_work():
    for seed in (0, C.SEED_MAX):
        for family in range(4):
            a = ev.analyse(ev.Settings(5, family, seed))
            assert a.mode == "enum" and ev.tau_rows(a)[0]["tau"] == 0.0


def test_the_modes_switch_at_the_documented_sizes():
    assert ev.analyse(ev.Settings(C.ENUM_MAX, 0, 5)).mode == "enum" and ev.analyse(ev.Settings(C.ENUM_MAX + 1, 0, 5)).mode == "lap"
    assert ev.tau_rows(ev.analyse(ev.Settings(C.MILP_MAX, 0, 5))) is not None and ev.tau_rows(ev.analyse(ev.Settings(C.MILP_MAX + 1, 0, 5))) is None
    big = ev.analyse(ev.Settings(C.MILP_MAX + 1, 0, 5))
    with pytest.raises(ValueError):
        ev.w_grid(big, 1)
    with pytest.raises(ValueError):
        ev.chain_grid(ev.analyse(ev.Settings(C.ENUM_MAX + 1, 0, 5)))


def test_milp_tolerance_at_n_10_gives_a_valid_sequential_result():
    a = ev.analyse(ev.Settings(10, 0, 5))
    rows = ev.tau_rows(a)
    assert a.mode == "lap" and rows[0]["same"] and rows[0]["allowed"] is None and all(A.check_feasible(r["plan"], 10) for r in rows)
    assert all(rows[i]["vals"][1] >= rows[i + 1]["vals"][1] - 1e-9 for i in range(len(rows) - 1))


def test_large_instance_uses_the_hungarian_method_and_the_exact_reference():
    for family in (0, 1, 3):
        a = ev.analyse(ev.Settings(C.N_MAX, family, 3))
        assert a.mode == "lap" and a.opt_set is None and A.check_feasible(a.ref.plan, C.N_MAX)
        rows = ev.m_grid(a, (0, 3, 6, 18))
        assert rows[0]["status"] == "A" and rows[1]["status"] in ("lex", "A") and (family == 3 or rows[-1]["status"] in ("lex", "B"))
    ints50 = ev.analyse(ev.Settings(50, 0, 3))
    assert [r["status"] for r in ev.m_grid(ints50, (0, 3, 16, 18))] == ["A", "lex", "B", "B"]


def test_real_values_never_lose_b_because_b_decides_nothing():
    for seed in range(10):
        a = ev.analyse(ev.Settings(8, 3, seed))
        assert len(a.opt_set) == 1 and not a.b_decides
        assert all(r["status"] in ("lex", "A") for r in ev.m_grid(a))


def test_extreme_weights_are_defined():
    a = ev.analyse(ev.Settings(5, 0, 3))
    row = ev.m_grid(a, (0, 18))
    assert row[0]["plan"] is not None and A.check_feasible(row[1]["plan"], 5)
    zero = A.weighted_plan_enum(a.inst.arrays()[:2], (0.0, 1.0), 5)                       # M = 0: nur B
    assert a.exact_sums[1][zero] == a.exact_sums[1].min()


def test_sweep_rows_have_the_documented_shape():
    seeds = range(100000, 100003)
    r = ev.sweep_rows(0, n=6, seeds=seeds)
    assert r["seeds"] == 3 and len(r["hits_m"]) == len(C.M_EXPONENTS) and len(r["hits_w"]) == len(C.W_EXPONENTS) and len(r["hits_chain"]) == len(C.CHAIN_EXPONENTS) and len(r["hits_tau"]) == len(C.TAUS)
    assert all(0 <= h <= 3 for key in ("hits_m", "hits_w", "hits_chain", "hits_tau") for h in r[key]) and r["m_star"][0] <= r["m_star"][1] <= r["m_star"][2]
    h = ev.highs_rows(0, n=5, seeds=seeds)
    assert len(h["hits_w"]) == len(C.W_EXPONENTS) and len(h["hits_m"]) == len(C.M_EXPONENTS)
    t = ev.tie_rows(2, n=6, seeds=seeds)
    assert t["lost"] <= 3 and t["miss9"] == 0
    assert set(ev.large_rows(0, n=20, seeds=seeds)["hits"]) == {0, 3, 16, 18}
