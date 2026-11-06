"""Experimente der App: jedes Preset, jede Stufe, jeder Löser, jede Toleranz und jedes Nachrechnen laufen ohne Fehler."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import lxk_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at) -> str:
    return " ".join(m.value for kind in ("success", "info", "warning") for m in getattr(at, kind))


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_the_lexicographic_verdict(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, (name, [str(e) for e in at.exception])
    text = _texts(at)
    assert "beste Pl" in text or "Bei n = 50" in text or "genau einen" in text


@pytest.mark.parametrize("k", list(C.M_EXPONENTS))
def test_every_weight_stage_runs_and_is_named(k):
    at = _app()
    at.select_slider(key="m_exp").set_value(k).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any(f"M = {'1' if k == 0 else '10'}" in m.value for m in at.markdown)


@pytest.mark.parametrize("solver", [0, 1])
def test_every_solver_runs_and_is_named(solver):
    at = _app()
    at.radio(key="solver_select").set_value(solver).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert C.SOLVERS[solver].split(" (")[0] in _texts(at)


@pytest.mark.parametrize("tau", list(C.TAUS))
def test_every_tolerance_runs(tau):
    at = _app()
    at.select_slider(key="tau_select").set_value(tau).run()
    assert not at.exception, [str(e) for e in at.exception]


def test_the_sweeps_run_when_switched_on():
    at = _app()
    for key in ("sweep_main", "sweep_highs", "sweep_large", "sweep_ties"):
        at.toggle(key=key).set_value(True).run()
        assert not at.exception, (key, [str(e) for e in at.exception])
    text = _texts(at)
    assert "je 20 feste Instanzen" in text and "HiGHS (Voreinstellungen)" in text and "n = 50" in text


def test_the_sweep_for_reals_says_there_are_no_ties_to_lose():
    at = _app()
    at.sidebar.selectbox(key="family_select").set_value(3).run()
    at.toggle(key="sweep_ties").set_value(True).run()
    assert not at.exception and "keinen exakten Gleichstand, den man verlieren könnte" in _texts(at)


def test_sizes_switch_the_available_experiments():
    at = _app()
    at.sidebar.slider(key="n_slider").set_value(10).run()
    assert not at.exception and at.radio(key="solver_select").options and "Drei Prioritäten gibt es nur" in _texts(at)
    at.sidebar.slider(key="n_slider").set_value(11).run()
    assert not at.exception and "bis n = 10" in _texts(at) and len(at.radio(key="solver_select").options) == 1
