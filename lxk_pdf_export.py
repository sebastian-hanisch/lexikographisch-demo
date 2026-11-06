"""PDF-Export des aktuellen Ergebnisses (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import lxk_constants as C
from lxk_evaluation import Analysis, Settings, de, m_grid, pow10, value_text

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "Δ": "Delta", "↔": "<->", "„": '"', "“": '"', "²": "2", "∞": "unendlich", "τ": "tau",
            "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Lexikographisch"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{s.n} Personen, {s.n} Aufgaben, Familie: {C.FAMILIES[s.family]}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    sc = a.inst.scale
    ref = a.ref.vals
    text = f"Lexikographisches Optimum: A = {value_text(ref[0], sc)}, B = {value_text(ref[1], sc)}" + (f", C = {value_text(ref[2], sc)}" if ref[2] is not None else "") + "."
    if a.opt_set is not None:
        text += f" Beste Pläne in A: {len(a.opt_set)}; B entscheidet: {'ja' if a.b_decides else 'nein'}; kleinstes Gewicht M* = {de(a.m_star, 2)}."
    pdf.multi_cell(0, 5, _s(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    widths = (30, 30, 30, 90)
    for w, h in zip(widths, ("M", "A", "B", "Status")):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in m_grid(a):
        for w, v in zip(widths, (pow10(r["k"]), value_text(r["vals"][0], sc), value_text(r["vals"][1], sc), C.STATUS_NAMES[r["status"]])):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Gerechnet wie ein Löser: Kostenmatrix M*A + B, Summe über die Einträge in float64. Quelle: lexikographisch-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
