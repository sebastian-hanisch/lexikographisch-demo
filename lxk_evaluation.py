"""Kennzahlen, Verdikt-Logik und Sweeps. Verdikte sind reine Funktionen (Art, Text); jede Zahl im Text kommt aus der Rechnung.

Lexikographisch heißt: unter allen Plänen mit kleinstem A der mit kleinstem B (dann C). Ein Gewicht ersetzt das nur in einem Fenster; eine Toleranz auf A
macht aus dem Vorrang einen Kompromiss."""

import math
from dataclasses import dataclass

import numpy as np

import lxk_algorithm as A
import lxk_constants as C
from lxk_scenario import Instance, make_instance


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; kein „-0,0“."""
    text = f"{x:.{digits}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{digits}f}"
    return text.replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    """Anteil als Prozent: 0,05 -> „5,0 %“."""
    return de(x * 100.0, digits) + " %"


def pow10(k: int) -> str:
    """Zehnerpotenz als Text: 10^3 -> „10³“, 10^-6 -> „10⁻⁶“."""
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "10" + str(k).translate(sup) if k != 0 else "1"


def tau_label(t: float) -> str:
    """Toleranz als Text: 0 -> „0“, 1e-9 -> „10⁻⁹“, 0,05 -> „5·10⁻²“, 0,1 -> „10⁻¹“."""
    if t == 0:
        return "0"
    e = int(math.floor(math.log10(t) + 1e-9))
    m = t / 10.0 ** e
    return pow10(e) if abs(m - 1.0) < 1e-9 else f"{m:g}·{pow10(e)}"


def value_text(v, scale: int) -> str:
    """Wert in den Einheiten der Eingabe: ganze Zahlen ohne Nachkommastelle, Zehntel mit einer, reelle Werte mit einer."""
    if scale == 1:
        return str(int(v))
    return de(float(v) / scale, 1) if scale else de(float(v), 1)


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    family: int = C.DEFAULT_FAMILY
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    inst: Instance
    mode: str                       # "enum" (alle Pläne) oder "lap" (Ungarische Methode)
    ref: A.Reference                # exaktes lexikographisches Optimum
    exact_sums: list | None         # Summen aller Pläne in der exakten Darstellung (n <= ENUM_MAX)
    float_sums: list | None         # Summen aller Pläne in Gleitkomma
    opt_set: np.ndarray | None      # Indizes der A-optimalen Pläne (exakt)
    b_decides: bool | None          # unterscheidet B unter den A-optimalen Plänen?
    float_ties: int | None          # Zahl der Pläne, die in float64 die kleinste A-Summe haben
    m_star: float | None            # kleinstes M, das M*A + B lexikographisch macht
    m_bound: int | None             # B-Spanne + 1 (nur ganzzahlige Daten)


def analyse(settings: Settings) -> Analysis:
    inst = make_instance(settings.n, settings.family, settings.seed)
    ref = A.lex_reference(inst)
    if inst.n > C.ENUM_MAX:
        bound = None
        if inst.ints is not None:
            b = inst.int_arrays()[1]
            bound = inst.n * int(b.max() - b.min()) + 1
        return Analysis(inst, "lap", ref, None, None, None, None, None, None, bound)
    ints = inst.int_arrays()
    exact = A.enum_sums(ints if ints is not None else inst.arrays(), inst.n)
    flt = A.enum_sums(inst.arrays(), inst.n)
    opt = A.optimal_set(exact[0])
    decides = bool(exact[1][opt].min() < exact[1][opt].max())
    return Analysis(inst, "enum", ref, exact, flt, opt, decides, int((flt[0] == flt[0].min()).sum()), A.m_star(exact[0], exact[1], ref.index),
                    A.int_bound(exact[1]) if ints is not None else None)


# --- Gitter über Gewichte und Toleranzen ---------------------------------------------------------------------------------------------------

def _plan_by_weights(a: Analysis, coeffs, mats) -> np.ndarray:
    if a.mode == "enum":
        return A.all_plans(a.inst.n)[A.weighted_plan_enum(mats, coeffs, a.inst.n)].astype(int)
    return A.lap_plan(sum(c * m for c, m in zip(coeffs, mats)))


def m_grid(a: Analysis, exponents=C.M_EXPONENTS) -> list[dict]:
    """M*A + B für M = 10^k: welcher Plan, welche Werte, und ob er dem lexikographischen Optimum gleicht (aufgezählt: argmin in float64; sonst Ungarische Methode)."""
    mats = a.inst.arrays()
    rows = []
    for k in exponents:
        plan = _plan_by_weights(a, (10.0 ** k, 1.0), mats[:2])
        vals = A.exact_values(a.inst, plan)
        rows.append({"k": k, "plan": plan, "vals": vals, "status": A.status(vals, a.ref.vals, a.ref.exact)})
    return rows


def w_grid(a: Analysis, solver: int = 0, exponents=C.W_EXPONENTS) -> list[dict]:
    """A + w*B für w = 10^-k mit der Ungarischen Methode (solver 0) oder HiGHS (solver 1, nur n <= MILP_MAX)."""
    if solver == 1 and a.inst.n > C.MILP_MAX:
        raise ValueError(f"HiGHS rechnet hier nur bis n = {C.MILP_MAX}")
    mats = a.inst.arrays()
    rows = []
    for k in exponents:
        cost = mats[0] + 10.0 ** (-k) * mats[1]
        plan = A.lap_plan(cost) if solver == 0 else A.highs_plan(cost)
        if plan is None:
            rows.append({"k": k, "plan": None, "vals": None, "status": "none"})
            continue
        vals = A.exact_values(a.inst, plan)
        rows.append({"k": k, "plan": plan, "vals": vals, "status": A.status(vals, a.ref.vals, a.ref.exact)})
    return rows


def chain_grid(a: Analysis, exponents=C.CHAIN_EXPONENTS) -> list[dict]:
    """M^2*A + M*B + C für M = 10^k gegen das exakte dreistufige Optimum (nur aufgezählt)."""
    if a.mode != "enum":
        raise ValueError("Drei Prioritäten nur für n <= ENUM_MAX")
    mats = a.inst.arrays()
    rows = []
    for k in exponents:
        m = 10.0 ** k
        plan = _plan_by_weights(a, (m * m, m, 1.0), mats)
        vals = A.exact_values(a.inst, plan)
        rows.append({"k": k, "plan": plan, "vals": vals, "status": A.status(vals, a.ref.vals, a.ref.exact, levels=3)})
    return rows


def tau_rows(a: Analysis, taus=C.TAUS) -> list[dict] | None:
    """Sequentiell mit relativer Toleranz tau auf A: Plan, Werte, erlaubte Pläne, Verlust in A und Gewinn in B gegen das lexikographische Optimum
    (aufgezählt bis n = ENUM_MAX, HiGHS bis n = MILP_MAX; darüber None)."""
    if a.inst.n > C.MILP_MAX:
        return None
    mats = a.inst.arrays()
    ref_a, ref_b = (float(v) for v in a.ref.vals[:2])
    rows = []
    for tau in taus:
        if a.mode == "enum":
            idx, allowed = A.tolerance_pick_enum(a.float_sums[0], a.float_sums[1], tau)
            plan = A.all_plans(a.inst.n)[idx].astype(int)
        else:
            plan, allowed = A.tolerance_pick_milp(mats[0], mats[1], tau), None
        vals = A.exact_values(a.inst, plan)
        rows.append({"tau": tau, "plan": plan, "vals": vals, "allowed": allowed, "same": A.status(vals, a.ref.vals, a.ref.exact) == "lex",
                     "a_loss": float(vals[0]) / ref_a - 1.0 if ref_a else 0.0, "b_gain": 1.0 - float(vals[1]) / ref_b if ref_b else 0.0})
    return rows


# --- Sweeps über feste Instanzen -----------------------------------------------------------------------------------------------------------

def _spread(values, digits=None):
    a = np.asarray(values, dtype=float)
    return float(a.min()), float(np.median(a)), float(a.max())


def sweep_rows(family: int, n: int = C.SWEEP_N, seeds=C.SWEEP_SEEDS) -> dict:
    """Über die festen Instanzen (aufgezählt): Gleichstände in A, kleinstes M, Treffer je M, je w (Ungarische Methode), je Toleranz, je Kette."""
    ties, sizes, decides = 0, [], 0
    mstars, bounds_ok, bounds = [], 0, []
    hits_m = [0] * len(C.M_EXPONENTS)
    hits_w = [0] * len(C.W_EXPONENTS)
    hits_chain = [0] * len(C.CHAIN_EXPONENTS)
    hits_tau = [0] * len(C.TAUS)
    a_loss = [[] for _ in C.TAUS]
    b_gain = [[] for _ in C.TAUS]
    c_decides = 0
    for seed in seeds:
        a = analyse(Settings(n, family, seed))
        ties += len(a.opt_set) > 1
        sizes.append(len(a.opt_set))
        decides += bool(a.b_decides)
        mstars.append(a.m_star)
        if a.m_bound is not None:
            bounds.append(a.m_bound)
            bounds_ok += a.m_star <= a.m_bound
        for i, r in enumerate(m_grid(a)):
            hits_m[i] += r["status"] == "lex"
        for i, r in enumerate(w_grid(a, 0)):
            hits_w[i] += r["status"] == "lex"
        for i, r in enumerate(chain_grid(a)):
            hits_chain[i] += r["status"] == "lex"
        for i, r in enumerate(tau_rows(a)):
            hits_tau[i] += r["same"]
            a_loss[i].append(r["a_loss"]), b_gain[i].append(r["b_gain"])
        top = a.opt_set[a.exact_sums[1][a.opt_set] == a.exact_sums[1][a.opt_set].min()]
        c_decides += bool(a.exact_sums[2][top].min() < a.exact_sums[2][top].max())
    ms = np.array(mstars)
    return {"family": family, "n": n, "seeds": len(seeds), "ties": int(ties), "sizes": _spread(sizes), "decides": int(decides),
            "m_star": _spread(ms), "m_p90": float(np.percentile(ms, 90)), "over10": int((ms > 10).sum()), "over1e3": int((ms > 1e3).sum()),
            "over1e6": int((ms > 1e6).sum()), "bound_median": float(np.median(bounds)) if bounds else None, "bound_ok": int(bounds_ok) if bounds else None,
            "hits_m": hits_m, "hits_w": hits_w, "hits_chain": hits_chain, "hits_tau": hits_tau,
            "a_loss": [_spread(x) for x in a_loss], "b_gain": [_spread(x) for x in b_gain], "c_decides": int(c_decides)}


def highs_rows(family: int, n: int = C.SWEEP_N, seeds=C.SWEEP_SEEDS) -> dict:
    """HiGHS (Voreinstellungen) über die festen Instanzen: Treffer je w (A + w*B) und je M (M*A + B)."""
    hits_w = [0] * len(C.W_EXPONENTS)
    hits_m = [0] * len(C.M_EXPONENTS)
    mats_cache = []
    for seed in seeds:
        a = analyse(Settings(n, family, seed))
        mats = a.inst.arrays()
        for i, k in enumerate(C.W_EXPONENTS):
            plan = A.highs_plan(mats[0] + 10.0 ** (-k) * mats[1])
            hits_w[i] += plan is not None and A.status(A.exact_values(a.inst, plan), a.ref.vals, a.ref.exact) == "lex"
        for i, k in enumerate(C.M_EXPONENTS):
            plan = A.highs_plan(10.0 ** k * mats[0] + mats[1])
            hits_m[i] += plan is not None and A.status(A.exact_values(a.inst, plan), a.ref.vals, a.ref.exact) == "lex"
    return {"family": family, "n": n, "seeds": len(seeds), "hits_w": hits_w, "hits_m": hits_m}


def large_rows(family: int, n: int = 50, seeds=C.SWEEP_SEEDS, exponents=(0, 3, 16, 18)) -> dict:
    """Große Instanz (Ungarische Methode): Treffer von M*A + B für einzelne M gegen das exakte lexikographische Optimum."""
    hits = {k: 0 for k in exponents}
    for seed in seeds:
        a = analyse(Settings(n, family, seed))
        for r in m_grid(a, exponents):
            hits[r["k"]] += r["status"] == "lex"
    return {"family": family, "n": n, "seeds": len(seeds), "hits": hits}


def tie_rows(family: int, n: int = C.SWEEP_N, seeds=C.SWEEP_SEEDS) -> dict:
    """Gleichstände in A: exakt gegen float64 (Zehntel-Familien): Instanzen mit exaktem Gleichstand, Optimalmenge exakt und in float64, verlorene
    Gleichstände, Wahl mit Toleranz 0 und mit relativer Toleranz 1e-9 gegen das exakte lexikographische Optimum."""
    exact_sizes, float_sizes, lost, miss0, miss9 = [], [], 0, 0, 0
    for seed in seeds:
        a = analyse(Settings(n, family, seed))
        exact_sizes.append(len(a.opt_set)), float_sizes.append(a.float_ties)
        lost += a.float_ties < len(a.opt_set)
        fa, fb = a.float_sums[0], a.float_sums[1]
        for tau, name in ((0.0, "0"), (1e-9, "9")):
            idx, _ = A.tolerance_pick_enum(fa, fb, tau)
            vals = tuple(s[idx].item() for s in a.exact_sums)
            bad = A.status(vals, a.ref.vals, a.ref.exact) != "lex"
            if name == "0":
                miss0 += bad
            else:
                miss9 += bad
    return {"family": family, "n": n, "seeds": len(seeds), "ties": int(sum(s > 1 for s in exact_sizes)), "exact": _spread(exact_sizes), "float": _spread(float_sizes),
            "lost": int(lost), "miss0": int(miss0), "miss9": int(miss9)}


# --- Verdikte ------------------------------------------------------------------------------------------------------------------------------

def verdict_lex(a: Analysis) -> tuple[str, str]:
    """Kernfrage: entscheidet B überhaupt etwas?"""
    s = a.inst.scale
    if a.mode == "lap":
        how = (f"mit der Ungarischen Methode und M = {a.m_bound} (ganze Zahlen, beweisbar exakt)" if a.m_bound is not None and s == 1
               else "mit der Ungarischen Methode auf Zehnteln und M = B-Spanne + 1 (beweisbar exakt)" if s else "nur nach A (bei reellen Werten hat B nichts zu entscheiden)")
        return "info", (f"Bei n = {a.inst.n} sind es zu viele Pläne zum Aufzählen (mehr als {C.ENUM_MAX}!); das lexikographische Optimum (A = {value_text(a.ref.vals[0], s)}, "
                        f"B = {value_text(a.ref.vals[1], s)}) kommt {how}. Wie viele beste Pläne A hat, zeigt die Demo hier nicht.")
    k = len(a.opt_set)
    a_best = value_text(a.exact_sums[0][a.opt_set[0]], s)
    if k == 1:
        return "info", (f"A hat **genau einen** besten Plan (A = {a_best}): B hat nichts zu entscheiden, **lexikographisch ist hier dasselbe wie „nur A“**. "
                        f"Das ist bei reellen Werten der Normalfall, bei ganzen Zahlen die Ausnahme.")
    b_vals = a.exact_sums[1][a.opt_set]
    if a.b_decides:
        return "success", (f"A hat **{k}** beste Pläne (A = {a_best}); unter ihnen reicht B von {value_text(b_vals.min(), s)} bis {value_text(b_vals.max(), s)}, "
                           f"und **B wählt den Plan mit B = {value_text(b_vals.min(), s)}**. Nur wegen des exakten Gleichstands in A hat B etwas zu sagen.")
    return "info", f"A hat **{k}** beste Pläne (A = {a_best}), aber alle mit demselben B = {value_text(b_vals.min(), s)}: B entscheidet nichts, erst C würde es."


def verdict_float_ties(a: Analysis) -> tuple[str, str]:
    """Experiment Gleichstand in Gleitkomma: stimmt die Optimalmenge in float64 mit der exakten überein?"""
    if a.mode != "enum":
        return "info", f"Der Vergleich exakt gegen float64 braucht die Aufzählung (n <= {C.ENUM_MAX})."
    if a.inst.scale == 0:
        return "info", "Bei reellen Werten gibt es keinen exakten Gleichstand zu verlieren: die Summen sind fast sicher alle verschieden."
    exact, flt = len(a.opt_set), a.float_ties
    if flt < exact:
        idx, _ = A.tolerance_pick_enum(a.float_sums[0], a.float_sums[1], 0.0)
        vals = tuple(s[idx].item() for s in a.exact_sums)
        ok = A.status(vals, a.ref.vals, a.ref.exact) == "lex"
        return "warning", (f"Exakt sind **{exact}** Pläne in A gleich, in float64-Summen gelten nur **{flt}** als gleich: **{exact - flt} Gleichstände sind verloren**. "
                           + ("Die Wahl mit Toleranz 0 trifft hier trotzdem das lexikographische Optimum." if ok
                              else f"Die Wahl mit Toleranz 0 liefert (A = {value_text(vals[0], a.inst.scale)}, B = {value_text(vals[1], a.inst.scale)}) statt "
                                   f"(A = {value_text(a.ref.vals[0], a.inst.scale)}, B = {value_text(a.ref.vals[1], a.inst.scale)}): **B hat falsch entschieden.**")
                           + " Mit relativer Toleranz 10⁻⁹ ist der Gleichstand wieder da.")
    return "info", f"Exakt gibt es {'**genau einen** besten Plan' if exact == 1 else f'**{exact}** beste Pläne'} in A, in float64-Summen ebenso viele ({flt}): hier geht kein Gleichstand verloren."


def _window(rows: list[dict], exps_key="k") -> tuple[list, int | None, int | None]:
    ok = [r[exps_key] for r in rows if r["status"] == "lex"]
    return ok, (min(ok) if ok else None), (max(ok) if ok else None)


def verdict_m(a: Analysis, rows: list[dict]) -> tuple[str, str]:
    """Experiment M: das Fenster, in dem M*A + B das lexikographische Optimum findet."""
    ok, lo, hi = _window(rows)
    exps = [r["k"] for r in rows]
    star = f" Das kleinste M ist **M\\* = {de(a.m_star, 2)}** (exakt aus allen Plänen)." if a.m_star is not None else ""
    bound = f" Für ganze Zahlen genügt beweisbar M = {a.m_bound} (B-Spanne + 1)." if a.m_bound is not None else ""
    if not ok:
        return "warning", f"**Keine** der {len(rows)} Stufen (10^{exps[0]} bis 10^{exps[-1]}) trifft das lexikographische Optimum.{star}"
    below = [r for r in rows if r["k"] < lo]
    above = [r for r in rows if r["k"] > hi]
    parts = [f"M·A + B trifft das lexikographische Optimum bei {len(ok)} von {len(rows)} Stufen, zwischen M = {pow10(lo)} und M = {pow10(hi)}."]
    if below:
        parts.append(f"Darunter (bis M = {pow10(below[-1]['k'])}) ist **A verletzt**: die Rangfolge kippt.")
    if above:
        kinds = sorted({r["status"] for r in above})
        what = "B geht verloren" if kinds == ["B"] else "es geht etwas verloren (" + ", ".join(kinds) + ")"
        parts.append(f"Ab {pow10(above[0]['k'])} gilt: **{what}**: M·A erreicht float64-Größen, in denen B nicht mehr unterscheidbar ist (2^53 ≈ 9,0·10¹⁵).")
    elif a.m_star is not None and a.inst.scale == 0:
        parts.append("Bei reellen Werten schadet ein großes M nicht: B hat dort nichts zu entscheiden.")
    return ("warning" if below or above else "info"), " ".join(parts) + star + bound


def verdict_w(a: Analysis, rows: list[dict], solver: int) -> tuple[str, str]:
    """Experiment w: A + w*B mit dem gewählten Löser."""
    name = C.SOLVERS[solver]
    ok, lo, hi = _window(rows)
    if not ok:
        return "warning", f"{name}: **keine** der Stufen w = 10⁻¹ bis 10⁻¹⁸ trifft das lexikographische Optimum."
    first_bad = [r["k"] for r in rows if r["k"] > hi]
    tail = ""
    if first_bad:
        bad = sorted({r["status"] for r in rows if r["k"] > hi})
        what = "wird **B ignoriert**" if bad == ["B"] else f"geht etwas verloren ({', '.join(bad)})"
        tail = f" Ab w = {pow10(-first_bad[0])} {what}: sein Beitrag w·B liegt unter der Rechen- oder Toleranzgrenze des Lösers."
    lead = [r["k"] for r in rows if r["k"] < lo]
    head = f" Bei w ≥ {pow10(-lo)} ist w zu groß: A wird verletzt." if lead else ""
    return ("warning" if first_bad or lead else "info"), f"{name}: A + w·B trifft das lexikographische Optimum bei w von {pow10(-lo)} bis {pow10(-hi)}.{head}{tail}"


def verdict_tau(a: Analysis, rows: list[dict] | None) -> tuple[str, str]:
    """Experiment Toleranz: wo beginnt „fast lexikographisch“, etwas zu ändern?"""
    if rows is None:
        return "info", f"Die sequentielle Rechnung mit Toleranz gibt es bis n = {C.MILP_MAX} (aufgezählt oder mit HiGHS)."
    noise = not rows[0]["same"] and rows[0]["b_gain"] < 0                       # Gleichstand in float64 verloren: schon τ = 0 weicht ab
    lead = ""
    if noise:
        lead = (f"**Schon bei τ = 0 weicht die Wahl vom lexikographischen Optimum ab**: der Gleichstand in A ist in float64 verloren, B ist {pct(-rows[0]['b_gain'])} schlechter; "
                f"mit τ ≥ 10⁻⁹ ist er wieder da. ")
    diff = [r for r in rows if not r["same"] and r["b_gain"] >= 0 and r["tau"] > 0]
    same = [r for r in rows if r["same"] and r["tau"] > 0]
    if not diff:
        return ("warning" if noise else "info"), lead + (f"Alle Toleranzstufen bis τ = {de(rows[-1]['tau'] * 100, 0)} % liefern das lexikographische Optimum: der zweitbeste Wert von A liegt weiter als 10 % entfernt.")
    first = diff[0]
    kept = f"Bis τ = {de(same[-1]['tau'], 3)} bleibt es lexikographisch; " if same and same[-1]["tau"] < first["tau"] else ""
    tail = f". Bei τ = {de(rows[-1]['tau'], 2)}: Verlust in A {pct(rows[-1]['a_loss'])}, Gewinn in B {pct(rows[-1]['b_gain'])}" if rows[-1] is not first else ""
    return "warning", (lead + kept + f"**ab τ = {de(first['tau'], 3)} wählt die Toleranz einen anderen Plan**: A verliert {pct(first['a_loss'])}, B gewinnt {pct(first['b_gain'])}"
                       + (f", {first['allowed']} Pläne sind erlaubt" if first["allowed"] else "") + tail + ". Eine Toleranz macht aus dem Vorrang einen Kompromiss.")


def verdict_chain(a: Analysis, rows: list[dict]) -> tuple[str, str]:
    """Experiment drei Prioritäten: wo geht die letzte Ebene verloren?"""
    ok, lo, hi = _window(rows)
    if not ok:
        return "warning", "Keine Stufe von M²·A + M·B + C trifft das dreistufige lexikographische Optimum."
    bad_above = [r for r in rows if r["k"] > hi]
    first = next((r for r in rows if r["k"] > hi), None)
    if first is None:
        return "info", f"M²·A + M·B + C trifft das dreistufige Optimum bei allen Stufen von M = {pow10(lo)} bis M = {pow10(hi)}."
    lost = sorted({r["status"] for r in bad_above})
    names = {"A": "A verletzt", "B": "B verloren", "C": "C verloren"}
    return "warning", (f"M²·A + M·B + C trifft das dreistufige Optimum von M = {pow10(lo)} bis M = {pow10(hi)}; **ab M = {pow10(first['k'])} geht es verloren** "
                       f"({', '.join(names[x] for x in lost)}): M²·A erreicht schon dort die float64-Grenze 2^53 ≈ 9,0·10¹⁵ (mit M = 10⁸ ist M² = 10¹⁶).")


def verdict_sweep(r: dict) -> tuple[str, str]:
    """Über feste Instanzen: Gleichstände, M*, Treffer je M und je τ; keine Trendaussage."""
    fam = C.FAMILIES[r["family"]]
    ms = r["m_star"]
    parts = [f"Gleichstand in A in {r['ties']} von {r['seeds']}, B entscheidet in {r['decides']}; Optimalmenge Median {de(r['sizes'][1], 0)} (Maximum {de(r['sizes'][2], 0)})",
             f"M* Min {de(ms[0], 2)}, Median {de(ms[1], 2)}, 90. Perzentil {de(r['m_p90'], 1)}, Max {de(ms[2], 1)}; M* über 10 in {r['over10']}, über 1000 in {r['over1e3']}",
             "Treffer je M (10⁰, 10¹, 10², …): " + ", ".join(str(h) for h in r["hits_m"]),
             "Treffer je τ (0, 10⁻¹², 10⁻⁹, 10⁻⁶, 10⁻³, 10⁻², 5·10⁻², 0,1): " + ", ".join(str(h) for h in r["hits_tau"])]
    if r["bound_ok"] is not None:
        parts.insert(2, f"Schranke B-Spanne + 1 (Median {de(r['bound_median'], 0)}) hält in {r['bound_ok']} von {r['seeds']}")
    return "info", f"Familie {fam}, n = {r['n']}, je {r['seeds']} feste Instanzen: " + "; ".join(parts) + "."


def verdict_highs(r: dict) -> tuple[str, str]:
    return "info", (f"HiGHS (Voreinstellungen), Familie {C.FAMILIES[r['family']]}, n = {r['n']}, je {r['seeds']} feste Instanzen: Treffer je w (10⁻¹ bis 10⁻¹⁸): "
                    + ", ".join(str(h) for h in r["hits_w"]) + "; Treffer je M (10⁰ bis 10¹⁸): " + ", ".join(str(h) for h in r["hits_m"]) + ".")


def verdict_large(r: dict) -> tuple[str, str]:
    parts = ", ".join(f"M = {pow10(k)} in {h} von {r['seeds']}" for k, h in r["hits"].items())
    return "info", f"Ungarische Methode, n = {r['n']}, je {r['seeds']} feste Instanzen, Treffer gegen das exakte Optimum: {parts}."


def verdict_ties(r: dict) -> tuple[str, str]:
    return "info", (f"Familie {C.FAMILIES[r['family']]}, n = {r['n']}, je {r['seeds']} feste Instanzen: exakter Gleichstand in A in {r['ties']}; Optimalmenge exakt Median {de(r['exact'][1], 0)} (Maximum {de(r['exact'][2], 0)}), "
                    f"in float64 erkannt Median {de(r['float'][1], 0)} (Maximum {de(r['float'][2], 0)}); verloren in {r['lost']}; die Wahl mit Toleranz 0 weicht ab in {r['miss0']}, mit relativer Toleranz 10⁻⁹ in {r['miss9']}.")


def plan_text(plan, n: int) -> str:
    return ", ".join(f"P{i + 1}→A{int(j) + 1}" for i, j in enumerate(plan)) if n <= 12 else "siehe Rechnung"
