"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import qmc_constants as C
from qmc_evaluation import Analysis, Settings, de, de_e, dn, dz, faktor, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "Sigma", "ρ": "rho", "α": "alpha", "β": "beta", "θ": "theta", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta", "½": "1/2", "̂": "", "ϱ": "rho", "ᵢ": "_i", "∫": "Integral"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Randomisiertes Quasi-Monte-Carlo"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Integrand: {a.integ.label}, {a.integ.d} Dimensionen; N = {de(a.N, 0)} Punkte; {a.reps} Wiederholungen je Punktzahl, Strom {settings.seed}. Alles simuliert, Wahrheit und Varianz sind exakt."),
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    widths = (80, 60)
    pdf.cell(widths[0], 6, _s("Kennzahl"), border=1)
    pdf.cell(widths[1], 6, _s("Wert"), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    dims = a.integ.dims
    for label, val in (("Wahrheit (exakt)", dz(a.integ.truth)), ("Varianz sigma^2 (exakt)", de_e(a.integ.var)),
                       ("mittlere Dimension (exakt)", dn(dims["mean_dim"], 2) if dims else "nicht berechnet"), ("additiver Anteil (exakt)", f"{de(dims['additive'] * 100, 0)} %" if dims else "nicht berechnet"),
                       ("Faktor Zufall", "1 (Bezug)"), ("Faktor Latin Hypercube", faktor(a.factor_lhs)), ("Faktor Sobol + Verschiebung", faktor(a.factor_ds)), ("Faktor Sobol + Owen", faktor(a.factor_owen)),
                       ("RMSE-Verhältnis Zufall : Owen", dn(a.ratio, 1)), ("Steigung Zufall / Owen", f"{dn(a.slope['mc'], 2)} / {dn(a.slope['sobol_owen'], 2)}")):
        pdf.cell(widths[0], 6, _s(label), border=1)
        pdf.cell(widths[1], 6, _s(val), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: quasi-monte-carlo-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
