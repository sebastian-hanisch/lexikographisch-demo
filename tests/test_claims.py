"""Jede Zahl aus dem README wird hier nachgerechnet (DEMO-PLAYBOOK §4). Aufzählung, Ungarische Methode und feste Instanzerzeugung sind deterministisch;
die HiGHS-Zeilen sind Messwerte der benutzten Version (Voreinstellungen)."""

from pathlib import Path

import pytest

import lxk_constants as C
import lxk_evaluation as ev

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
(TIE, SMALL, NO_TIE, FLOAT_LOST, TOLERANCE, THREE, BIG_M, MANY) = list(C.PRESETS)


@pytest.fixture(scope="module")
def sw():
    return {f: ev.sweep_rows(f) for f in (0, 1, 3)}


@pytest.fixture(scope="module")
def highs():
    return ev.highs_rows(0)


def _claim(text):
    assert text in README, text


def _slash(values, digits=0):
    return " / ".join(ev.de(v, digits) for v in values)


def _preset(name):
    return ev.analyse(ev.Settings(**C.PRESETS[name]))


def test_sweep_instances_are_the_documented_ones():
    assert C.SWEEP_SEEDS == tuple(range(100000, 100020)) and C.SWEEP_N == 8 and C.M_EXPONENTS == tuple(range(19)) and C.W_EXPONENTS == tuple(range(1, 19))
    _claim("Seeds 100000 bis 100019")


def test_tie_claims(sw):
    assert [sw[f]["ties"] for f in (0, 1, 3)] == [13, 1, 0] and [sw[f]["decides"] for f in (0, 1, 3)] == [13, 1, 0]
    assert _slash([sw[f]["sizes"][1] for f in (0, 1, 3)], 1) == "2,5 / 1,0 / 1,0" and _slash([sw[f]["sizes"][2] for f in (0, 1, 3)]) == "9 / 2 / 1"
    for text in ("**13 / 1 / 0** von 20", "**2,5 / 1,0 / 1,0**", "**9 / 2 / 1**", "bei reellen Werten ist „lexikographisch“ dasselbe wie „nur A“"):
        _claim(text)


def test_m_star_claims(sw):
    ints, dec, real = sw[0], sw[1], sw[3]
    assert _slash([ints["m_star"][0], ints["m_star"][1]], 2) == "0,75 / 3,75" and ev.de(ints["m_p90"], 1) == "7,0" and ev.de(ints["m_star"][2], 1) == "7,0"
    assert _slash([dec["m_star"][0], dec["m_star"][1]], 2) == "0,97 / 8,11" and ev.de(dec["m_p90"], 1) == "23,5" and ev.de(dec["m_star"][2], 1) == "78,0"
    assert _slash([real["m_star"][0], real["m_star"][1]], 2) == "0,68 / 5,90" and ev.de(real["m_p90"], 1) == "29,9" and ev.de(real["m_star"][2], 1) == "96,7"
    assert [sw[f]["over10"] for f in (0, 1, 3)] == [0, 8, 7] and [sw[f]["over1e3"] for f in (0, 1, 3)] == [0, 0, 0]
    assert ints["bound_median"] == 48.5 and ints["bound_ok"] == 20
    for text in ("**0,75 / 3,75 / 7,0 / 7,0**", "**0,97 / 8,11 / 23,5 / 78,0**", "**0,68 / 5,90 / 29,9 / 96,7**", "**0 / 8 / 7** von 20", "**0 / 0 / 0**", "Median **48,5**", "in **20** von 20"):
        _claim(text)


def test_weight_window_claims(sw):
    hits = lambda f, ks: [sw[f]["hits_m"][k] for k in ks]
    assert [hits(f, (0, 1, 2)) for f in (0, 1, 3)] == [[1, 20, 20], [1, 12, 20], [2, 13, 20]]
    assert hits(0, (14, 15, 16, 17, 18)) == [20, 20, 13, 11, 11]
    assert all(h == 20 for f in (1, 3) for h in sw[f]["hits_m"][2:])                    # Dezimal und reell: ab M = 100 immer 20
    for text in ("**1 / 20 / 20**", "**1 / 12 / 20**", "**2 / 13 / 20**", "**20 / 20 / 13 / 11 / 11**"):
        _claim(text)


def test_large_instance_claims():
    assert ev.large_rows(0)["hits"] == {0: 0, 3: 20, 16: 0, 18: 0}
    _claim("`M` = 1: **0**; `M` = 10³: **20**; `M` = 10¹⁶: **0**; `M` = 10¹⁸: **0**")


def test_small_weight_claims(sw, highs):
    lap = sw[0]["hits_w"]                                                              # w = 10^-1 bis 10^-18
    assert all(h == 20 for h in lap[:15]) and lap[15:] == [16, 10, 10]
    _claim("bis w = 10⁻¹⁵ **20**, w = 10⁻¹⁶: **16**, w = 10⁻¹⁷ und 10⁻¹⁸: **10**")
    w = highs["hits_w"]
    assert all(h == 20 for h in w[:6]) and w[6] == 19 and all(h == 8 for h in w[7:])
    m = highs["hits_m"]
    assert m[0] == 1 and all(h == 20 for h in m[1:16]) and m[16:] == [17, 10, 12]
    _claim("w = 10⁻¹ bis 10⁻⁶: **20**, w = 10⁻⁷: **19**, **ab w = 10⁻⁸ bis 10⁻¹⁸: 8**")
    _claim("`M` = 1: **1**; 10¹ bis 10¹⁵: **20**; 10¹⁶: **17**; 10¹⁷: **10**; 10¹⁸: **12**")


def test_three_priorities_claims(sw):
    assert sw[0]["hits_chain"] == [20, 20, 20, 20, 20, 20, 20, 19, 19, 20, 18, 19] and sw[0]["c_decides"] == 2           # M = 10^1 bis 10^12
    _claim("`M` = 10¹ bis 10⁷: **20**; 10⁸: **19**; 10⁹: **19**; 10¹⁰: **20**; 10¹¹: **18**; 10¹²: **19**; C entscheidet unter den A- und B-Optimalen in **2** von 20 Instanzen")


def test_tolerance_claims(sw):
    assert sw[0]["hits_tau"] == [20, 20, 20, 20, 20, 20, 19, 9] and sw[1]["hits_tau"] == [20, 20, 20, 20, 20, 17, 9, 5] and sw[3]["hits_tau"] == [20, 20, 20, 20, 20, 18, 11, 5]
    loss, gain = sw[3]["a_loss"][7], sw[3]["b_gain"][7]
    assert (ev.pct(loss[1]), ev.pct(loss[2]), ev.pct(gain[1]), ev.pct(gain[2])) == ("5,4 %", "9,6 %", "4,5 %", "41,1 %")
    for text in ("**20 / 20 / 20 / 20 / 20 / 20 / 19 / 9**", "**20 / 20 / 20 / 20 / 20 / 17 / 9 / 5**", "**20 / 20 / 20 / 20 / 20 / 18 / 11 / 5**", "Verlust in A **5,4 % / 9,6 %**, Gewinn in B **4,5 % / 41,1 %**"):
        _claim(text)


def test_float_tie_claims():
    r = ev.tie_rows(2)
    assert (r["ties"], r["exact"][1], r["exact"][2], r["float"][1], r["float"][2], r["lost"], r["miss0"], r["miss9"]) == (13, 2.5, 9.0, 2.0, 8.0, 2, 1, 0)
    _claim("exakter Gleichstand in A in **13** von 20 (Optimalmenge Median 2,5, Maximum 9); in float64 erkannt Median **2,0** (Maximum **8**); verloren in **2** von 20; die Wahl mit Toleranz 0 weicht ab in **1** von 20, mit relativer Toleranz 10⁻⁹ in **0**")


def test_preset_claims():
    a = _preset(TIE)
    rows = ev.m_grid(a)
    assert (len(a.opt_set), ev.value_text(a.exact_sums[0][a.opt_set[0]], 1), a.m_star) == (6, "17", 2.0) and [r["status"] for r in rows].count("lex") == 15
    assert (a.exact_sums[1][a.opt_set].min(), a.exact_sums[1][a.opt_set].max()) == (31, 44)
    _claim("**6** beste Pläne (A = 17), B reicht von **31** bis **44**, `M*` = **2,00**, Fenster 10¹ bis 10¹⁵ (15 von 19 Stufen), ab 10¹⁶ geht B verloren")
    b = _preset(SMALL)
    assert (len(b.opt_set), b.m_star, b.m_bound, b.exact_sums[1][b.opt_set].min(), b.exact_sums[1][b.opt_set].max(), ev.value_text(b.exact_sums[0][b.opt_set[0]], 1)) == (2, 4.0, 12, 17, 23, "13")
    _claim("**2** beste Pläne (A = 13), B **17** gegen **23**, `M*` = **4,00**, Schranke **12**")
    c = _preset(FLOAT_LOST)
    assert (len(c.opt_set), c.float_ties) == (4, 2) and "B = 2,9) statt (A = 1,9, B = 2,7" in ev.verdict_float_ties(c)[1]
    assert "Ab w = 10⁻⁷ wird **B ignoriert**" in ev.verdict_w(c, ev.w_grid(c, 1), 1)[1]
    _claim("**4** exakte Gleichstände, in float64 **2**; die Wahl mit Toleranz 0 liefert B = **2,9** statt **2,7**; HiGHS ignoriert B ab w = 10⁻⁷")
    d = _preset(TOLERANCE)
    row = ev.tau_rows(d)[C.TAUS.index(1e-2)]
    assert (ev.pct(row["a_loss"]), ev.pct(row["b_gain"]), row["allowed"]) == ("0,8 %", "20,1 %", 2) and ev.tau_rows(d)[C.TAUS.index(1e-3)]["same"]
    _claim("Verlust in A **0,8 %**, Gewinn in B **20,1 %**, **2** Pläne erlaubt; τ = 0,001 ändert nichts")
    assert len(_preset(THREE).opt_set) == 8
    _claim("**8** beste Pläne in A; die Kette trifft von 10¹ bis 10⁷, ab **10⁸** geht C verloren")
    assert _preset(BIG_M).m_star == 78.0
    _claim("`M*` = **78,00**")
    m = _preset(MANY)
    assert (m.ref.vals[0], m.ref.vals[1]) == (50, 140) and [r["status"] for r in ev.m_grid(m, (0, 3, 16, 18))] == ["A", "lex", "B", "B"]
    _claim("lexikographisch A = **50**, B = **140**; `M` = 1 / 10³ / 10¹⁶ / 10¹⁸: Rangfolge gekippt / richtig / B verloren / B verloren")
