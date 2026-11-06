"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und was jedes Preset zeigen soll."""

import lxk_constants as C
import lxk_evaluation as ev
import lxk_presets as P

(TIE, SMALL, NO_TIE, FLOAT_LOST, TOLERANCE, THREE, BIG_M, MANY) = list(C.PRESETS)


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and 0 <= preset["family"] < len(C.FAMILIES) and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[TIE]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def _analysis(name):
    return ev.analyse(ev.Settings(**C.PRESETS[name]))


def test_tie_preset_has_six_best_plans_and_a_b_decision():
    a = _analysis(TIE)
    assert a.mode == "enum" and len(a.opt_set) == 6 and a.b_decides and a.m_star == 2.0 and a.m_bound == 48
    rows = {r["k"]: r["status"] for r in ev.m_grid(a)}
    assert rows[0] == "A" and rows[1] == "lex" and rows[15] == "lex" and rows[16] == "B"


def test_small_preset_has_two_best_plans_and_m_star_four():
    a = _analysis(SMALL)
    assert a.inst.n == 3 and len(a.opt_set) == 2 and a.b_decides and a.m_star == 4.0 and a.m_bound == 12


def test_no_tie_preset_has_one_best_plan():
    a = _analysis(NO_TIE)
    assert len(a.opt_set) == 1 and not a.b_decides and a.inst.scale == 0


def test_float_lost_preset_loses_two_of_four_ties_and_decides_wrongly_at_tolerance_zero():
    a = _analysis(FLOAT_LOST)
    assert len(a.opt_set) == 4 and a.float_ties == 2 and not ev.tau_rows(a)[0]["same"] and ev.tau_rows(a)[2]["same"]


def test_tolerance_preset_trades_a_little_a_for_a_lot_of_b_at_one_percent():
    a = _analysis(TOLERANCE)
    row = ev.tau_rows(a)[C.TAUS.index(1e-2)]
    assert not row["same"] and row["a_loss"] < 0.01 and row["b_gain"] > 0.2 and ev.tau_rows(a)[C.TAUS.index(1e-3)]["same"]


def test_three_priorities_preset_loses_c_from_ten_to_the_eight():
    a = _analysis(THREE)
    rows = {r["k"]: r["status"] for r in ev.chain_grid(a)}
    assert all(rows[k] == "lex" for k in range(1, 8)) and rows[8] == "C"


def test_big_m_star_preset_needs_more_than_seventy_eight():
    a = _analysis(BIG_M)
    assert a.m_star == 78.0 and ev.m_grid(a)[1]["status"] == "A" and ev.m_grid(a)[2]["status"] == "lex"


def test_many_preset_is_not_enumerated_and_shows_the_same_windows():
    a = _analysis(MANY)
    assert a.mode == "lap" and a.opt_set is None
    assert [r["status"] for r in ev.m_grid(a, (0, 3, 16, 18))] == ["A", "lex", "B", "B"]
