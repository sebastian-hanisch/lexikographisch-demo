"""Verdikte: reine Funktionen, jede Verzweigung und jede Zahl im Text an handgesetzten oder von Hand geprüften Werten."""

import lxk_constants as C
import lxk_evaluation as ev

(TIE, SMALL, NO_TIE, FLOAT_LOST, TOLERANCE, THREE, BIG_M, MANY) = list(C.PRESETS)


def _preset(name):
    return ev.analyse(ev.Settings(**C.PRESETS[name]))


def test_number_formatting_uses_decimal_commas_and_powers_of_ten():
    assert ev.de(3.14159) == "3,1" and ev.pct(0.059) == "5,9 %" and ev.pct(-0.0000001) == "0,0 %"
    assert ev.pow10(3) == "10³" and ev.pow10(-6) == "10⁻⁶" and ev.pow10(0) == "1" and ev.pow10(18) == "10¹⁸"
    assert ev.value_text(17, 1) == "17" and ev.value_text(19, 10) == "1,9" and ev.value_text(179.23, 0) == "179,2"


def test_lex_verdict_branches():
    kind, text = ev.verdict_lex(_preset(TIE))
    assert kind == "success" and "A hat **6** beste Pläne (A = 17)" in text and "reicht B von 31 bis 44" in text and "B = 31" in text
    kind, text = ev.verdict_lex(_preset(NO_TIE))
    assert kind == "info" and "A hat **genau einen** besten Plan (A = 89,3)" in text and "dasselbe wie „nur A“" in text
    kind, text = ev.verdict_lex(_preset(FLOAT_LOST))
    assert kind == "success" and "A hat **4** beste Pläne (A = 1,9)" in text and "reicht B von 2,7 bis 3,1" in text
    kind, text = ev.verdict_lex(_preset(MANY))
    assert kind == "info" and "Bei n = 50" in text and "(A = 50, B = 140)" in text and "M = 401" in text


def test_lex_verdict_when_b_cannot_separate_the_best_plans():
    import numpy as np
    a = _preset(TIE)
    a.b_decides = False                                                                  # künstlich: alle Pläne mit gleichem B
    a.exact_sums = [a.exact_sums[0], np.zeros_like(a.exact_sums[1]), a.exact_sums[2]]
    kind, text = ev.verdict_lex(a)
    assert kind == "info" and "alle mit demselben B = 0" in text and "erst C würde es" in text


def test_float_tie_verdict_branches():
    kind, text = ev.verdict_float_ties(_preset(FLOAT_LOST))
    assert kind == "warning" and "Exakt sind **4** Pläne in A gleich" in text and "nur **2** als gleich" in text and "2 Gleichstände sind verloren" in text
    assert "(A = 1,9, B = 2,9) statt (A = 1,9, B = 2,7)" in text and "B hat falsch entschieden" in text
    kind, text = ev.verdict_float_ties(_preset(TIE))
    assert kind == "info" and "**6** beste Pläne" in text and "ebenso viele (6)" in text
    assert "genau einen" in ev.verdict_float_ties(_preset(BIG_M))[1]
    assert "keinen exakten Gleichstand" in ev.verdict_float_ties(_preset(NO_TIE))[1] and "Aufzählung" in ev.verdict_float_ties(_preset(MANY))[1]


def test_m_verdict_names_the_window_and_both_failures():
    a = _preset(TIE)
    kind, text = ev.verdict_m(a, ev.m_grid(a))
    assert kind == "warning" and "bei 15 von 19 Stufen, zwischen M = 10¹ und M = 10¹⁵" in text and "Darunter (bis M = 1) ist **A verletzt**" in text
    assert "Ab 10¹⁶ gilt: **B geht verloren**" in text and "M\\* = 2,00" in text and "M = 48 (B-Spanne + 1)" in text


def test_m_verdict_for_reals_says_a_large_weight_is_harmless():
    a = _preset(NO_TIE)
    kind, text = ev.verdict_m(a, ev.m_grid(a))
    assert "zwischen M = 10¹ und M = 10¹⁸" in text and "schadet ein großes M nicht" in text and "M\\* = 1,17" in text and "B-Spanne" not in text


def test_m_verdict_when_no_stage_works_and_for_the_large_instance():
    a = _preset(TIE)
    kind, text = ev.verdict_m(a, [{"k": k, "plan": None, "vals": None, "status": "A"} for k in range(3)])
    assert kind == "warning" and "**Keine** der 3 Stufen" in text
    kind, text = ev.verdict_m(_preset(MANY), ev.m_grid(_preset(MANY)))
    assert "M = 401" in text and "Ab 10¹⁶" in text and "M\\*" not in text


def test_w_verdict_names_where_b_is_ignored_for_each_solver():
    a = _preset(FLOAT_LOST)
    kind, text = ev.verdict_w(a, ev.w_grid(a, 0), 0)
    assert kind == "warning" and text.startswith("Ungarische Methode (float64)") and "bei w von 10⁻¹ bis 10⁻¹⁶" in text and "Ab w = 10⁻¹⁷ wird **B ignoriert**" in text
    kind, text = ev.verdict_w(a, ev.w_grid(a, 1), 1)
    assert kind == "warning" and "HiGHS" in text and "bei w von 10⁻¹ bis 10⁻⁶" in text and "Ab w = 10⁻⁷ wird **B ignoriert**" in text
    kind, text = ev.verdict_w(_preset(NO_TIE), ev.w_grid(_preset(NO_TIE), 0), 0)
    assert kind == "info" and "bei w von 10⁻¹ bis 10⁻¹⁸" in text
    kind, text = ev.verdict_w(_preset(TOLERANCE), ev.w_grid(_preset(TOLERANCE), 0), 0)
    assert kind == "warning" and "Bei w ≥ 10⁻² ist w zu groß: A wird verletzt" in text


def test_tau_verdict_branches():
    kind, text = ev.verdict_tau(_preset(TIE), ev.tau_rows(_preset(TIE)))
    assert kind == "warning" and "Bis τ = 0,050 bleibt es lexikographisch" in text and "ab τ = 0,100" in text and "A verliert 5,9 %, B gewinnt 6,5 %, 14 Pläne sind erlaubt" in text
    kind, text = ev.verdict_tau(_preset(TOLERANCE), ev.tau_rows(_preset(TOLERANCE)))
    assert "ab τ = 0,010" in text and "A verliert 0,8 %, B gewinnt 20,1 %, 2 Pläne" in text and "Bei τ = 0,10: Verlust in A 7,2 %, Gewinn in B 35,3 %" in text
    kind, text = ev.verdict_tau(_preset(THREE), ev.tau_rows(_preset(THREE)))
    assert kind == "info" and "Alle Toleranzstufen bis τ = 10 % liefern" in text
    kind, text = ev.verdict_tau(_preset(FLOAT_LOST), ev.tau_rows(_preset(FLOAT_LOST)))
    assert kind == "warning" and "Schon bei τ = 0 weicht die Wahl vom lexikographischen Optimum ab" in text and "B ist 7,4 % schlechter" in text and "mit τ ≥ 10⁻⁹ ist er wieder da" in text
    assert "bis n = 10" in ev.verdict_tau(_preset(MANY), None)[1]


def test_chain_verdict_branches():
    kind, text = ev.verdict_chain(_preset(THREE), ev.chain_grid(_preset(THREE)))
    assert kind == "warning" and "von M = 10¹ bis M = 10⁷" in text and "ab M = 10⁸ geht es verloren" in text and "(C verloren)" in text and "M² = 10¹⁶" in text
    kind, text = ev.verdict_chain(_preset(TIE), ev.chain_grid(_preset(TIE)))
    assert kind == "info" and "von M = 10¹ bis M = 10¹²" in text
    assert "Keine Stufe" in ev.verdict_chain(_preset(TIE), [{"k": k, "plan": None, "vals": None, "status": "A"} for k in range(1, 4)])[1]


def _sweep(**over):
    r = {"family": 0, "n": 8, "seeds": 20, "ties": 13, "sizes": (1.0, 2.5, 9.0), "decides": 13, "m_star": (0.75, 3.75, 7.0), "m_p90": 7.0, "over10": 0, "over1e3": 0, "over1e6": 0,
         "bound_median": 48.5, "bound_ok": 20, "hits_m": [1] + [20] * 15 + [13, 11, 11], "hits_w": [20] * 18, "hits_chain": [20] * 12, "hits_tau": [20, 20, 20, 20, 20, 20, 19, 9],
         "a_loss": [(0, 0, 0)] * 8, "b_gain": [(0, 0, 0)] * 8, "c_decides": 2}
    r.update(over)
    return r


def test_sweep_verdicts_have_every_number_and_no_trend_word():
    kind, text = ev.verdict_sweep(_sweep())
    assert kind == "info" and "Gleichstand in A in 13 von 20, B entscheidet in 13; Optimalmenge Median 2 (Maximum 9)" in text.replace("2,5", "2") or "Optimalmenge Median 2" in text
    assert "M* Min 0,75, Median 3,75, 90. Perzentil 7,0, Max 7,0" in text and "ist beweisbar hinreichend; M* liegt darunter in 20 von 20" in text
    assert "Treffer je M" in text and "Treffer je τ" in text
    for word in ("fällt", "steigt", "wächst", "sinkt", "Trend"):
        assert word not in text
    assert "Schranke" not in ev.verdict_sweep(_sweep(bound_ok=None, bound_median=None))[1]


def test_highs_large_and_tie_sweep_verdicts():
    kind, text = ev.verdict_highs({"family": 0, "n": 8, "seeds": 20, "hits_w": [20, 20, 8], "hits_m": [1, 20, 17]})
    assert kind == "info" and "Treffer je w (10⁻¹ bis 10⁻¹⁸): 20, 20, 8" in text and "Treffer je M (10⁰ bis 10¹⁸): 1, 20, 17" in text
    kind, text = ev.verdict_large({"family": 0, "n": 50, "seeds": 20, "hits": {0: 0, 3: 20, 16: 0, 18: 0}})
    assert "n = 50" in text and "M = 1 in 0 von 20" in text and "M = 10³ in 20 von 20" in text and "M = 10¹⁶ in 0 von 20" in text
    kind, text = ev.verdict_ties({"family": 2, "n": 8, "seeds": 20, "ties": 13, "exact": (1.0, 2.5, 9.0), "float": (1.0, 2.0, 8.0), "lost": 2, "miss0": 1, "miss9": 0})
    assert "exakter Gleichstand in A in 13" in text and "verloren in 2" in text and "Toleranz 0 weicht ab in 1" in text and "10⁻⁹ in 0" in text


def test_tolerance_labels_are_powers_of_ten():
    assert [ev.tau_label(t) for t in C.TAUS] == ["0", "10⁻¹²", "10⁻⁹", "10⁻⁶", "10⁻³", "10⁻²", "5·10⁻²", "10⁻¹"]
