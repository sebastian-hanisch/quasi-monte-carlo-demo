"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import qmc_algorithm as A
import qmc_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(exponentformat="power")           # 10^-5 statt SI-Vorsilben („100µ“) auf den logarithmischen Achsen
    return lock_axes(fig)


def build_curve(a):
    """Fehler (RMSE) gegen die Punktzahl, doppelt logarithmisch: je steiler, desto besser; Zufall fällt mit Steigung −0,5."""
    fig = go.Figure()
    for m in ("mc", "lhs", "sobol_ds", "sobol_owen"):
        ys = [max(v, 1e-300) for v in a.rmse[m]]
        fig.add_trace(go.Scatter(x=list(a.ns), y=ys, mode="lines+markers", name=C.METHOD_LABELS[m], line=dict(color=C.METHOD_COLORS[m])))
    ref = a.rmse["mc"][0]
    fig.add_trace(go.Scatter(x=list(a.ns), y=[ref * (a.ns[0] / n) ** 0.5 for n in a.ns], mode="lines", name="Steigung −0,5", line=dict(color=C.COLOR_TRUTH, dash="dot", width=1)))
    fig.update_xaxes(title_text="Punkte N", type="log", tickvals=list(a.ns), ticktext=[str(n) for n in a.ns])
    fig.update_yaxes(title_text="Fehler (RMSE)", type="log")
    return _base(fig, 360)


def build_dimension(exp):
    """Gemessener Faktor der Owen-Sobol-Folge gegen die exakte mittlere Dimension; Begleiter und Netz-Integranden."""
    fig = go.Figure()
    for kind, name, color, symbol in (("exp", "Begleiter exp(Σ aᵢuᵢ)", C.COLOR_OWEN, "circle"), ("netz", "Netz-Integranden", C.COLOR_DS, "diamond")):
        rows = [r for r in exp["rows"] if r["kind"] == kind]
        fig.add_trace(go.Scatter(x=[r["dims"]["mean_dim"] for r in rows], y=[max(r["factor_owen"], 1e-5) for r in rows], mode="markers", name=name, text=[r["label"] for r in rows],
                                 hovertemplate="%{text}<br>mittlere Dimension %{x:.2f}<br>Faktor %{y:.4f}<extra></extra>", marker=dict(size=10, color=color, symbol=symbol)))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_BAD, dash="dot"), annotation_text="Zufall (Faktor 1)", annotation_position="top left")
    fig.update_xaxes(title_text="exakte mittlere Dimension")
    fig.update_yaxes(title_text="Faktor MSE / (σ²/N), Owen", type="log")
    return _base(fig, 360)


def build_ladder(exp):
    """Faktor je Punktmenge für die drei Integranden desselben Netzes."""
    fig = go.Figure()
    labels = {"sprung": "Sprung", "engpass": "Engpass (Knicke)", "glatt": "glatt (Polynom)"}
    for m in ("mc", "lhs", "sobol_ds", "sobol_owen"):
        fig.add_trace(go.Bar(x=[labels[r["key"]] for r in exp["rows"]], y=[max(r["factors"][m], 1e-5) for r in exp["rows"]], name=C.METHOD_LABELS[m], marker_color=C.METHOD_COLORS[m]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Faktor MSE / (σ²/N)", type="log")
    return _base(fig, 340)


def build_intervals(exp):
    """Überdeckung des 95-%-Intervalls je Verfahren."""
    rows = exp["rows"]
    colors = [C.COLOR_MC if r["name"] == "mc" else C.COLOR_BAD if r["name"] == "naiv" else C.COLOR_OWEN for r in rows]
    fig = go.Figure(go.Bar(x=[r["label"] for r in rows], y=[r["coverage"] for r in rows], marker_color=colors, text=[f"{r['coverage']:.3f}".replace(".", ",") for r in rows], textposition="outside"))
    fig.add_hline(y=0.95, line=dict(color=C.COLOR_TRUTH, dash="dot"), annotation_text="95 %", annotation_position="bottom right")
    fig.update_yaxes(title_text="Überdeckung", range=[0, 1.1])
    fig = _base(fig, 320)
    fig.update_layout(showlegend=False)
    return fig


def build_sizes(exp):
    sizes = exp["sizes"]
    fig = go.Figure()
    for key, label, color in (("factor_owen", "Sobol + Owen", C.COLOR_OWEN), ("factor_ds", "Sobol + Verschiebung", C.COLOR_DS)):
        fig.add_trace(go.Bar(x=[f"{r['N']} ({r['name']})" for r in sizes], y=[max(r[key], 1e-5) for r in sizes], name=label, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Faktor MSE / (σ²/N)", type="log")
    return _base(fig, 340)


def build_netz(netz):
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    ex, ey = [], []
    for u, v in netz.edges:
        ex += [xs[u], xs[v], None]
        ey += [ys[u], ys[v], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#666", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), showlegend=False, hoverinfo="skip"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    fig = _base(fig, 280)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig


def build_points(seed, N=64):
    """Die ersten beiden Koordinaten von N Punkten: Zufall, Latin Hypercube, Sobol mit Owen-Scrambling."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=[C.METHOD_LABELS[m] for m in ("mc", "lhs", "sobol_owen")], horizontal_spacing=0.05)
    for i, m in enumerate(("mc", "lhs", "sobol_owen"), start=1):
        u = A.points(m, N, 2, A.stream_seed(seed, 99, i))
        fig.add_trace(go.Scatter(x=u[:, 0], y=u[:, 1], mode="markers", marker=dict(size=6, color=C.METHOD_COLORS[m]), hoverinfo="skip", showlegend=False), row=1, col=i)
        fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1], row=1, col=i)
        fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1], row=1, col=i)
    fig = _base(fig, 300)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig
