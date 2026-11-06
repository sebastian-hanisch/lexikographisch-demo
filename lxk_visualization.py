"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lxk_algorithm as A
import lxk_constants as C
from lxk_evaluation import pow10, tau_label

STATUS_CODE = {"lex": 0, "A": 1, "B": 2, "C": 3, "none": 4}
COLORSCALE = []
for _code, _key in enumerate(("lex", "A", "B", "C", "none")):
    COLORSCALE += [[_code / 5, C.STATUS_COLORS[_key]], [(_code + 1) / 5, C.STATUS_COLORS[_key]]]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_plans(a, marks: dict | None = None, tau: float | None = None):
    """Alle Pläne als Punkte (A gegen B, n <= 8); die A-optimalen Pläne blau, das lexikographische Optimum als Stern; `marks`: Name -> Index weiterer Pläne;
    `tau`: senkrechte Linie bei A*(1 + tau)."""
    s = a.inst.scale or 1
    fa, fb = (np.asarray(x, dtype=float) / s for x in a.exact_sums[:2])
    fig = go.Figure(go.Scattergl(x=fa, y=fb, mode="markers", marker=dict(size=4, color=C.COLOR_ALL), name="alle Pläne", hovertemplate="A %{x:.1f}, B %{y:.1f}<extra></extra>"))
    opt = a.opt_set
    fig.add_trace(go.Scatter(x=fa[opt], y=fb[opt], mode="markers", marker=dict(size=11, color=C.COLOR_OPT, line=dict(width=1, color="white")), name="beste Pläne in A",
                             hovertemplate="A %{x:.1f}, B %{y:.1f}<extra></extra>"))
    k = a.ref.index
    fig.add_trace(go.Scatter(x=[fa[k]], y=[fb[k]], mode="markers", marker=dict(symbol="star", size=20, color="#111", line=dict(width=1, color="white")), name="lexikographisches Optimum",
                             hovertemplate="A %{x:.1f}, B %{y:.1f}<extra></extra>"))
    for name, idx in (marks or {}).items():
        fig.add_trace(go.Scatter(x=[fa[idx]], y=[fb[idx]], mode="markers", marker=dict(symbol="diamond", size=14, color=C.STATUS_COLORS["B"], line=dict(width=1, color="white")), name=name,
                                 hovertemplate=name + ": A %{x:.1f}, B %{y:.1f}<extra></extra>"))
    if tau is not None:
        fig.add_vline(x=float(fa.min()) * (1.0 + tau), line=dict(color=C.STATUS_COLORS["A"], dash="dot", width=2))
    fig.update_xaxes(title_text="A (Kosten, Summe)")
    fig.update_yaxes(title_text="B (Zeit, Summe)")
    return _base(fig, height=400)


def build_strip(rows, key_label, star=None, bound=None, negative=False):
    """Eine Zeile Zellen, je Stufe eine: Farbe = Status (grün gleich dem lexikographischen Optimum, rot A verletzt, orange B verloren, violett C verloren).

    `negative`: Stufen sind Gewichte 10^-k (w auf B); `star`: M* (senkrechte Linie bei log10), `bound`: B-Spanne + 1 (gestrichelt)."""
    exps = [(-r["k"] if negative else r["k"]) for r in rows]
    z = [[STATUS_CODE[r["status"]] for r in rows]]
    text = [[f"{key_label} = {pow10(e)}: {C.STATUS_NAMES[r['status']]}" for e, r in zip(exps, rows)]]
    fig = go.Figure(go.Heatmap(z=z, x=exps, y=[""], text=text, hoverinfo="text", colorscale=COLORSCALE, zmin=0, zmax=5, showscale=False, xgap=2))
    for value, color, dash in ((star, "#111", "solid"), (bound, "#111", "dash")):
        if value is not None and value > 0 and not negative:
            fig.add_vline(x=float(np.log10(value)), line=dict(color=color, width=2, dash=dash))
    fig.update_xaxes(tickmode="array", tickvals=exps, ticktext=[pow10(e) for e in exps], title_text=key_label)
    fig.update_yaxes(showticklabels=False)
    for key in ("lex", "A", "B", "C"):
        if any(r["status"] == key for r in rows):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=C.STATUS_COLORS[key], symbol="square"), name=C.STATUS_NAMES[key]))
    fig.update_layout(legend=dict(orientation="h", y=-0.75))
    return _base(fig, height=250)


def build_tau(rows):
    """Verlust in A und Gewinn in B je Toleranz (Prozent gegen das lexikographische Optimum)."""
    xs = [f"τ = {tau_label(r['tau'])}" for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Verlust in A (Prozent)", "Gewinn in B (Prozent)"))
    colors = [C.STATUS_COLORS["lex"] if r["same"] else C.STATUS_COLORS["A"] for r in rows]
    fig.add_trace(go.Bar(x=xs, y=[r["a_loss"] * 100 for r in rows], marker_color=colors, showlegend=False, hovertemplate="%{x}: %{y:.2f} %<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=xs, y=[r["b_gain"] * 100 for r in rows], marker_color=colors, showlegend=False, hovertemplate="%{x}: %{y:.2f} %<extra></extra>"), row=1, col=2)
    fig.update_xaxes(type="category", tickangle=-45)
    return _base(fig, height=340)


def build_hits(hits, total, labels, title):
    """Treffer je Stufe über die festen Instanzen (Anzahl von `total`)."""
    fig = go.Figure(go.Bar(x=labels, y=hits, marker_color=C.STATUS_COLORS["lex"], hovertemplate="%{x}: %{y} von " + str(total) + "<extra></extra>"))
    fig.update_yaxes(range=[0, total], title_text=f"Treffer von {total}")
    fig.update_xaxes(type="category", title_text=title, tickangle=-45)
    return _base(fig, height=300)
