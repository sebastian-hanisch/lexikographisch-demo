"""Lexikographisch - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Reihe „Zielfunktionsdesign“: dieselbe Zuordnung wie in Stück 1 bis 3, jetzt mit Prioritäten: erst A (Kosten), unter den besten Plänen B (Zeit), dann C (Wünsche).
Man schreibt das gern als ein Ziel M·A + B mit großem Gewicht. Die Demo zeigt, wann das dasselbe ist, wann nicht, und was eine Toleranz daraus macht.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import lxk_algorithm as A
import lxk_constants as C
from lxk_evaluation import (
    Settings,
    analyse,
    chain_grid,
    de,
    highs_rows,
    large_rows,
    m_grid,
    plan_text,
    pow10,
    sweep_rows,
    tau_label,
    tau_rows,
    tie_rows,
    value_text,
    verdict_chain,
    verdict_float_ties,
    verdict_highs,
    verdict_large,
    verdict_lex,
    verdict_m,
    verdict_sweep,
    verdict_tau,
    verdict_ties,
    verdict_w,
    w_grid,
)
from lxk_layout import dunkles_design, kopf_css
from lxk_pdf_export import build_pdf
from lxk_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lxk_visualization import build_hits, build_plans, build_strip, build_tau

st.set_page_config(page_title="Lexikographisch – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="20 feste Instanzen werden gerechnet ...")
def _sweep(family):
    return sweep_rows(family)


@st.cache_data(show_spinner="20 feste Instanzen werden mit HiGHS gerechnet ...")
def _highs(family):
    return highs_rows(family)


@st.cache_data(show_spinner="20 feste Instanzen werden gerechnet ...")
def _large(family):
    return large_rows(family)


@st.cache_data(show_spinner="20 feste Instanzen werden gerechnet ...")
def _ties(family):
    return tie_rows(family)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lxk_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🥇 Lexikographisch")
st.markdown(
    """
**Ist „zuerst A, dann B“ dasselbe wie ein großes Gewicht?** Dieselbe Aufgabe wie in Stück 1 bis 3: **n Personen** übernehmen **n Aufgaben**, jede Paarung hat bis zu drei Belastungen:
**A** (Kosten), **B** (Zeit), **C** (Wünsche). **Lexikographisch** heißt: unter allen Plänen mit kleinstem A wähle den mit kleinstem B, unter diesen den mit kleinstem C. Man schreibt das gern als
**ein** Ziel **M·A + B** mit großem M. Die Demo prüft, wann das stimmt: Das Gewicht hat ein **Fenster**, in dem es das lexikographische Optimum findet; darunter kippt die Rangfolge, darüber verschwindet
B im Gleitkomma oder in der Toleranz des Lösers. Und eine **Toleranz** auf A macht aus dem Vorrang einen Kompromiss.

Weiter unten: **🔬 Das Gewicht M**, **🔬 Ein kleines Gewicht auf B und der Löser**, **🔬 Sequentiell mit Toleranz**, **🔬 Drei Prioritäten**, **🔬 Wie oft und wie viel?**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Vergleich von Vorrang und Gewicht", expanded=True):
    st.markdown(
        """
1. **Lexikographisch:** B entscheidet nur bei **exaktem Gleichstand** in A. Bei ganzen Zahlen kommt das vor (viele Pläne mit gleicher Summe), bei reellen Werten praktisch nie: dort ist „lexikographisch“ dasselbe wie „nur A“.
2. **Ein Gewicht:** M·A + B ist wieder eine Zuordnung mit den Kosten M·a + b je Paar. Sie hat dasselbe Optimum wie die lexikographische Rangfolge nur, wenn M groß genug ist (**M ≥ M\\***, das kleinste M, bei dem A nicht mehr verletzt wird) und **nicht zu groß**: in float64 (53 Bit Mantisse) unterscheidet M·A + B ab etwa 9·10¹⁵ nicht mehr jede ganze Zahl.
3. **Die Rechnung wie ein Löser:** Die Demo bildet die Kostenmatrix M·A + B und summiert die Einträge in float64, genau wie die Ungarische Methode es tut (bei Gleichstand der gerechneten Zahl gilt der erste Plan der Aufzählung).
4. **Eine Toleranz** τ erlaubt, dass A um den Anteil τ vom Besten abweicht, und wählt unter den erlaubten Plänen das kleinste B: **fast lexikographisch**. Das ist dieselbe Rechnung wie der ε-Constraint aus Stück 3.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Instanz**")
    n = st.slider("Personen und Aufgaben n", *bounds("n_slider"), key="n_slider",
                  help=f"Bis n = {C.ENUM_MAX} werden alle Pläne aufgezählt (exakt in ganzen Zahlen), bis n = {C.MILP_MAX} rechnet HiGHS mit; darüber die Ungarische Methode.")
    family = st.selectbox("Art der Belastungen", list(range(len(C.FAMILIES))), key="family_select", format_func=lambda i: C.FAMILIES[i],
                          help="Ganze Zahlen haben exakte Gleichstände; Dezimalzahlen sind in float64 nicht exakt; reelle Werte haben praktisch keine Gleichstände.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "family_select": int(family), "seed_input": int(seed)})

settings = Settings(int(n), int(family), int(seed))
a = _analysis(settings)
s = a.inst.scale

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: das lexikographische Optimum")
m1, m2, m3 = st.columns(3)
m1.metric("Beste Pläne in A", "–" if a.opt_set is None else str(len(a.opt_set)), delta=None if a.opt_set is not None else f"nur bis n = {C.ENUM_MAX}", delta_color="off")
m2.metric("B entscheidet?", "–" if a.b_decides is None else ("ja" if a.b_decides else "nein"))
m3.metric("Kleinstes Gewicht M*", "–" if a.m_star is None else de(a.m_star, 2), delta=None if a.m_bound is None else f"genügt: M = {a.m_bound}", delta_color="off")

ref_vals = [value_text(v, s) if v is not None else "–" for v in a.ref.vals]
st.markdown(f"**Lexikographisches Optimum:** A = {ref_vals[0]}, B = {ref_vals[1]}" + (f", C = {ref_vals[2]}" if a.ref.vals[2] is not None else "") + f"; Plan {plan_text(a.ref.plan, a.inst.n)}")

if a.mode == "enum":
    marks = {}
    b_only = A.lex_index([a.exact_sums[1], a.exact_sums[0], a.exact_sums[2]])
    marks["bestes B (nur B)"] = b_only
    st.plotly_chart(build_plans(a, marks), width="stretch", key="plans_chart")
    st.caption("Jeder Punkt ist einer der "
               f"{len(a.exact_sums[0])} Pläne. Blau: die besten Pläne in A (sie liegen auf einer senkrechten Linie); der Stern ist das lexikographische Optimum, die orange Raute das beste B ohne Rücksicht auf A.")
st.download_button("📄 Ergebnis als PDF", data=build_pdf(settings, a), file_name="lexikographisch.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Entscheidet B überhaupt etwas?")
kind, text = verdict_lex(a)
getattr(st, kind)(text)
if a.mode == "enum" and s:
    kind, text = verdict_float_ties(a)
    getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Das Gewicht M")
st.caption("M·A + B für M = 10⁰ bis 10¹⁸, gerechnet wie ein Löser (Kostenmatrix, Summe über die Einträge in float64). Jede Zelle ist eine Stufe: grün gleich dem lexikographischen Optimum, rot A verletzt, orange B verloren.")
rows_m = m_grid(a)
exp_choice = st.select_slider("Gewicht M = 10^k", options=list(C.M_EXPONENTS), value=6, format_func=pow10, key="m_exp")
row = rows_m[C.M_EXPONENTS.index(exp_choice)]
st.markdown(f"**M = {pow10(exp_choice)}:** Plan {plan_text(row['plan'], a.inst.n)}; A = {value_text(row['vals'][0], s)}, B = {value_text(row['vals'][1], s)}: {C.STATUS_NAMES[row['status']]}.")
st.plotly_chart(build_strip(rows_m, "M", star=a.m_star, bound=a.m_bound), width="stretch", key="m_chart")
st.caption("Die durchgezogene senkrechte Linie ist M\\* (das kleinste Gewicht, aus allen Plänen berechnet), die gestrichelte die Schranke „B-Spanne + 1“ für ganze Zahlen." if a.m_star is not None
           else "Das kleinste Gewicht M\\* braucht die Aufzählung (n <= 8).")
kind, text = verdict_m(a, rows_m)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Ein kleines Gewicht auf B und der Löser")
st.caption("Dieselbe Rangfolge als A + w·B mit w = 10⁻¹ bis 10⁻¹⁸ (so schreibt man „B nur als Tiebreak“). Welche Stufen stimmen, hängt vom Löser ab, nicht von der Mathematik.")
solver_options = [0, 1] if a.inst.n <= C.MILP_MAX else [0]
solver = st.radio("Löser", solver_options, format_func=lambda i: C.SOLVERS[i], horizontal=True, key="solver_select")
rows_w = w_grid(a, int(solver))
st.plotly_chart(build_strip(rows_w, "w", negative=True), width="stretch", key="w_chart")
if a.inst.n > C.MILP_MAX:
    st.caption(f"HiGHS rechnet hier nur bis n = {C.MILP_MAX}.")
kind, text = verdict_w(a, rows_w, int(solver))
getattr(st, kind)(text)
st.caption("Laut Dokumentation (Gurobi, Seite „Multiple Objectives“, gelesen) soll man Gewichte über 10⁶ oder unter 10⁻⁶ vermeiden, weil sie numerische Schwierigkeiten machen oder ignoriert werden können; laut HiGHS-Optionen gelten Toleranzen von 10⁻⁷ (Zulässigkeit) und 10⁻⁴ (relative Lücke). Die Messung oben gilt für die hier benutzte Version.")

st.markdown("---")

st.subheader("🔬 Sequentiell mit Toleranz")
st.caption("Erst A minimieren, dann B unter A ≤ A*·(1 + τ): „fast lexikographisch“. Die Toleranz τ entscheidet, wie viel A man für B opfert.")
rows_t = tau_rows(a)
if rows_t is None:
    st.info(f"Die sequentielle Rechnung mit Toleranz gibt es bis n = {C.MILP_MAX} (aufgezählt oder mit HiGHS).")
else:
    tau = st.select_slider("Relative Toleranz τ auf A", options=list(C.TAUS), value=1e-2, format_func=tau_label, key="tau_select")
    row_t = rows_t[C.TAUS.index(tau)]
    st.markdown(f"**τ = {tau_label(tau)}:** Plan {plan_text(row_t['plan'], a.inst.n)}; A = {value_text(row_t['vals'][0], s)}, B = {value_text(row_t['vals'][1], s)}" +
                (f"; erlaubt sind {row_t['allowed']} Pläne" if row_t["allowed"] else "") + f"; Verlust in A {de(row_t['a_loss'] * 100, 2)} %, Gewinn in B {de(row_t['b_gain'] * 100, 2)} %.")
    if a.mode == "enum":
        st.plotly_chart(build_plans(a, {f"gewählt (τ = {tau_label(tau)})": next(i for i in range(len(a.exact_sums[0])) if tuple(A.all_plans(a.inst.n)[i]) == tuple(row_t["plan"]))}, tau=float(tau)),
                        width="stretch", key="tau_plans_chart")
    st.plotly_chart(build_tau(rows_t), width="stretch", key="tau_chart")
    kind, text = verdict_tau(a, rows_t)
    getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Drei Prioritäten")
st.caption("A, dann B, dann C als M²·A + M·B + C mit M = 10¹ bis 10¹²: die Gewichte wachsen quadratisch, die float64-Grenze kommt früher.")
if a.mode != "enum":
    st.info(f"Drei Prioritäten gibt es nur für n <= {C.ENUM_MAX} (exakte Aufzählung).")
else:
    rows_c = chain_grid(a)
    st.plotly_chart(build_strip(rows_c, "M (Kette M²·A + M·B + C)"), width="stretch", key="chain_chart")
    kind, text = verdict_chain(a, rows_c)
    getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Wie oft und wie viel?")
st.caption(f"Je {len(C.SWEEP_SEEDS)} feste Instanzen (SWEEP_SEEDS) der gewählten Art, n = {C.SWEEP_N}, alle aufgezählt; die Zahlen sind Zählungen und Spannen über Instanzen, kein Satz.")
if st.toggle("Gleichstände, M* und Treffer je Stufe nachrechnen", value=False, key="sweep_main"):
    sw = _sweep(settings.family)
    kind, text = verdict_sweep(sw)
    getattr(st, kind)(text)
    st.plotly_chart(build_hits(sw["hits_m"], sw["seeds"], [pow10(k) for k in C.M_EXPONENTS], "Gewicht M (M·A + B)"), width="stretch", key="sweep_m_chart")
    st.plotly_chart(build_hits(sw["hits_w"], sw["seeds"], [pow10(-k) for k in C.W_EXPONENTS], "Gewicht w (A + w·B, Ungarische Methode)"), width="stretch", key="sweep_w_chart")
if st.toggle("HiGHS über die festen Instanzen (dauert einige Sekunden)", value=False, key="sweep_highs"):
    kind, text = verdict_highs(_highs(settings.family))
    getattr(st, kind)(text)
if st.toggle("n = 50 mit der Ungarischen Methode (M gegen das exakte Optimum)", value=False, key="sweep_large"):
    kind, text = verdict_large(_large(settings.family))
    getattr(st, kind)(text)
if st.toggle("Gleichstände exakt gegen float64 (nur Dezimalzahlen und ganze Zahlen)", value=False, key="sweep_ties"):
    if C.INT_SCALE[C.FAMILY_KEYS[settings.family]]:
        kind, text = verdict_ties(_ties(settings.family))
        getattr(st, kind)(text)
    else:
        st.info("Bei reellen Werten gibt es keinen exakten Gleichstand, den man verlieren könnte.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Summenkriterien einer Zuordnung** | Bei Wechselwirkungen, Nebenbedingungen oder nichtlinearen Kriterien (Maximum, Quadrat) sehen Fenster und Schranken anders aus; Stück 1 und 2 zeigen diese Kriterien | Stück 1 (Dasselbe Problem, drei Ziele), Stück 2 (Preis der Fairness) |
| **Die Fenster gelten für diese Löser und Versionen** | Gemessen sind die Ungarische Methode (scipy) und HiGHS (`scipy.optimize.milp`, Voreinstellungen); andere Löser (Gurobi, CP-SAT) und andere Versionen können andere Fenster haben | Dokumentation des jeweiligen Lösers |
| **Gleichstand heißt: exakt gleich** | In Gleitkommazahlen sind Dezimalzahlen nicht exakt; zwei gleiche Dezimalsummen können sich in der letzten Stelle unterscheiden | Stück 5 (Gleichstände) |
| **Die Toleranz ist relativ auf A** | Eine absolute Toleranz oder eine Toleranz je Stufe wählt anders; die Wahl von τ ist eine Entscheidung, kein Rechenergebnis | Stück 3 (ε-Constraint) |
| **Vier synthetische Familien, n = 8 und n = 50** | Wie oft B entscheidet und wie groß M\\* wird, hängt an Familie und Größe; echte Daten können sich anders verhalten | – |
| **Reelle Werte: B hat nichts zu entscheiden** | Die Demo zeigt das als Befund, nicht als Regel: bei gerundeten Messwerten gibt es sehr wohl Gleichstände | Stück 5 |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gegeben $n$ Personen, $n$ Aufgaben und drei Belastungen $a_{ij}, b_{ij}, c_{ij} \ge 0$. Ein Plan ist eine Permutation $\pi$; seine Summen $A(\pi) = \sum_i a_{i,\pi(i)}$, ebenso $B(\pi)$ und $C(\pi)$.

**Lexikographisch.** $\pi \prec \pi'$, wenn $A(\pi) < A(\pi')$ oder ($A(\pi) = A(\pi')$ und $B(\pi) < B(\pi')$) oder (beides gleich und $C(\pi) < C(\pi')$). Das lexikographische Optimum ist der kleinste Plan bezüglich $\prec$.

**Gewicht.** Für $M > 0$ löst die Ungarische Methode $\min_\pi M A(\pi) + B(\pi)$. Das kleinste $M^\*$, bei dem das Optimum das lexikographische ist: für das lexikographische Optimum $\pi^\*$ mit $(A^\*, B^\*)$ ist
$$M^\* = \max\left(0,\ \max_{\pi:\,A(\pi) > A^\*} \frac{B^\* - B(\pi)}{A(\pi) - A^\*}\right).$$
**Ganze Zahlen.** Sind alle Summen ganzzahlig, ist $A(\pi) - A^\* \ge 1$ für jeden Plan mit $A(\pi) > A^\*$, und $M = (\max B - \min B) + 1$ genügt: $M^\* \le \max B - \min B$. Reelle Werte haben keine solche Schranke für alle Instanzen.

**Gleitkomma.** float64 hat 53 Bit Mantisse: $\mathrm{fl}(2^{53} + 1) = 2^{53} \approx 9{,}007 \cdot 10^{15}$. Ist $M \cdot a_{ij} \ge 2^{53}$, unterscheidet die Kostenmatrix $M a_{ij} + b_{ij}$ das $b_{ij}$ nicht mehr. Bei drei Prioritäten kommt $M^2 a_{ij}$ ins Spiel: schon $M = 10^8$ erreicht $M^2 a = 10^{16} a$.

**Toleranz.** $\min_\pi B(\pi)$ unter $A(\pi) \le A^\* (1 + \tau)$; mit $\tau = 0$ ist das lexikographisch (bei exakten Zahlen), mit $\tau > 0$ ein ε-Constraint auf $A$ (Stück 3).

**Literatur.** Gurobi, Dokumentation „Multiple Objectives“ (gelesen): blended (Gewichte `ObjNWeight`) und hierarchical (Prioritäten `ObjNPriority`) mit `ObjNRelTol` (Voreinstellung 0) und `ObjNAbsTol` (1e-6), Warnung vor Gewichten über 1e6 und unter 1e-6. HiGHS, Optionen (gelesen): `mip_rel_gap` 1e-4, `mip_abs_gap` 1e-6, Zulässigkeitstoleranzen 1e-7. Diese Demo übernimmt keine Zahl daraus; alle Zahlen sind Messungen an den Instanzfamilien der Demo.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
