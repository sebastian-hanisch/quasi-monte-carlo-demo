"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Gemessen wird immer derselbe Schätzer „Mittel von f über N Punkte“ für vier Punktmengen: Zufall (Monte Carlo), Latin Hypercube (Stück 5), Sobol-Folge mit zufälliger digitaler Verschiebung und Sobol-Folge mit
Owen-Scrambling. Kennzahl ist der Faktor = mittlerer quadratischer Fehler geteilt durch die exakte Monte-Carlo-Varianz σ²/N (unter 1: Gewinn). Wahrheit und σ² sind exakt; die effektive Dimension
(mittlere Dimension, additiver Anteil, Überlagerungsdimension) ebenfalls, wo die ANOVA berechenbar ist (Sprung, glatt, Begleiter).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np

import qmc_algorithm as A
import qmc_constants as C
from qmc_scenario import Netz, named_netz


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    """Wie `de`, aber „nicht bestimmbar“ für NaN und „unendlich“ für ∞."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, nd)


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


def dz(x: float) -> str:
    """Zahl über viele Zehnerpotenzen: ab 10 Millionen wissenschaftlich, sonst mit Tausenderpunkt und passend vielen Stellen."""
    if not np.isfinite(x):
        return dn(x)
    if abs(x) >= 1e7:
        return de_e(x)
    if abs(x) >= 100:
        return de(x, 0)
    return de(x, 1 if abs(x) >= 10 else 2)


def faktor(x: float) -> str:
    """Faktor MSE / (σ²/N): unter 0,0001 als solcher benannt, sonst mit passend vielen Stellen."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if x < 1e-4:
        return "unter 0,0001"
    if x < 0.01:
        return de(x, 4)
    if x < 1:
        return de(x, 3)
    return de(x, 2)


def kanten(n: int) -> str:
    return f"{n} Kante" if n == 1 else f"{n} Kanten"


@dataclass(frozen=True)
class Settings:
    integrand: str = C.DEFAULT_INTEGRAND
    netz: str = C.DEFAULT_NETZ
    p: float = C.DEFAULT_P
    d: int = C.DEFAULT_D
    weight: str = C.DEFAULT_WEIGHT
    nexp: int = C.DEFAULT_NEXP
    seed: int = C.DEFAULT_SEED

    @property
    def N(self) -> int:
        return 1 << self.nexp


@dataclass(frozen=True)
class Integrand:
    key: str
    label: str
    d: int
    truth: float
    var: float
    f: Callable = None                    # (N, d)-Punkte -> (N,)-Werte
    dims: dict | None = None              # exakte effektive Dimension (mean_dim, additive, sup90, sup99) oder None, wenn nicht berechnet


@dataclass(frozen=True)
class Analysis:
    integ: Integrand
    N: int
    ns: tuple                             # Punktzahlen der Fehlerkurve
    rmse: dict                            # Methode -> RMSE je N
    factor: dict                          # Methode -> Faktor je N
    slope: dict                           # Methode -> Steigung von log RMSE über log N
    factor_owen: float                    # beim gewählten N
    factor_ds: float
    factor_lhs: float
    lhs_pred: float                       # 1 - additiver Anteil (Vorhersage für Latin Hypercube), NaN ohne ANOVA
    ratio: float                          # RMSE Monte Carlo : RMSE Owen beim gewählten N
    reps: int


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def netz_table(netz: Netz):
    """Aufzählung aller Kantenmengen (höchstens C.LIVE_EDGE_LIMIT Kanten): Zusammenhangstabelle und Zählung N_k."""
    if netz.m > C.LIVE_EDGE_LIMIT:
        raise ValueError(f"{netz.m} Kanten: aufgezählt wird nur bis {C.LIVE_EDGE_LIMIT}")
    tab = A.conn_table(netz.n, netz.edges)
    return tab, A.conn_counts(tab, netz.m)


def _dims_of(shares) -> dict | None:
    return None if shares is None else A.dim_summary(np.asarray(shares))


def make_integrand(settings: Settings, netz: Netz | None = None, table=None, key: str | None = None) -> Integrand:
    """Der gewählte Integrand mit exakter Wahrheit, exakter Varianz und (wo möglich) exakter effektiver Dimension."""
    key = key or settings.integrand
    if key == "exp":
        a = A.exp_weights(settings.weight, settings.d)
        return Integrand("exp", f"Begleiter exp(Σ aᵢuᵢ), {C.WEIGHTS[settings.weight]}", settings.d, A.exp_exact(a), A.exp_var_exact(a), lambda u, a=a: A.f_exp(u, a),
                         _dims_of(A.exp_order_shares(a)))
    netz = netz if netz is not None else named_netz(settings.netz)
    tab, Nk = table if table is not None else netz_table(netz)
    m = netz.m
    if key == "sprung":
        p = settings.p
        R = A.r_exact(Nk, p)
        shares = A.order_shares(A.anova_coeffs(tab, m, p) ** 2, m)
        return Integrand("sprung", f"Sprung: {netz.name} hängt zusammen (p = {de(p, 2)})", m, R, R * (1 - R), lambda u, p=p, tab=tab: A.f_sprung(u, p, tab), _dims_of(shares))
    if key == "engpass":
        return Integrand("engpass", f"Engpass: {netz.name}", m, A.engpass_exact(Nk), A.engpass_var_exact(Nk), lambda u, tab=tab: A.f_engpass(u, tab), None)
    if key == "glatt":
        shares = A.order_shares(A.glatt_sigma2(tab, m), m)
        return Integrand("glatt", f"glatt: Zuverlässigkeit {netz.name} bei unsicheren p", m, A.r_exact(Nk, C.P_SMOOTH), A.glatt_var_exact(tab, m), lambda u, tab=tab: A.f_glatt(u, tab), _dims_of(shares))
    raise ValueError(f"unbekannter Integrand {key}")


# --- Messung ------------------------------------------------------------------------------------------------------------------------------

def _errors(integ: Integrand, method: str, N: int, reps: int, seed: int, tag: int) -> np.ndarray:
    mi = A.METHODS.index(method) if method in A.METHODS else 9
    return np.array([integ.f(A.points(method, N, integ.d, A.stream_seed(seed, tag, N, mi, r))).mean() - integ.truth for r in range(reps)])


def _factor(err: np.ndarray, var: float, N: int) -> float:
    return float(np.mean(err**2) / (var / N)) if var > 0 else float("nan")


def _slope(ns, rmse) -> float:
    if len(ns) < 3 or not all(r > 0 for r in rmse):
        return float("nan")
    return float(np.polyfit(np.log(ns), np.log(rmse), 1)[0])


def error_curve(integ: Integrand, nexp: int, reps: int, seed: int) -> dict:
    """Fehler gegen N = 2^MIN_CURVE_EXP ... 2^nexp für die vier Punktmengen."""
    ns = [1 << k for k in range(C.MIN_CURVE_EXP, nexp + 1)]
    rmse = {m: [] for m in A.METHODS}
    fac = {m: [] for m in A.METHODS}
    for N in ns:
        for m in A.METHODS:
            err = _errors(integ, m, N, reps, seed, 1)
            rmse[m].append(float(np.sqrt(np.mean(err**2))))
            fac[m].append(_factor(err, integ.var, N))
    return {"ns": ns, "rmse": rmse, "factor": fac, "slope": {m: _slope(ns, rmse[m]) for m in A.METHODS}}


def analyse(settings: Settings, netz: Netz | None = None, table=None, key: str | None = None) -> Analysis:
    """Exakte Wahrheit und Varianz, dazu die Fehlerkurve mit REPS Wiederholungen je Punktzahl."""
    integ = make_integrand(settings, netz, table, key)
    curve = error_curve(integ, settings.nexp, C.REPS, settings.seed)
    last = -1
    fo, fd, fl = curve["factor"]["sobol_owen"][last], curve["factor"]["sobol_ds"][last], curve["factor"]["lhs"][last]
    ratio = curve["rmse"]["mc"][last] / curve["rmse"]["sobol_owen"][last] if curve["rmse"]["sobol_owen"][last] > 0 else float("inf")
    pred = 1.0 - integ.dims["additive"] if integ.dims is not None else float("nan")
    integ = dataclasses.replace(integ, f=None)                       # die Auswertungsfunktion lässt sich nicht zwischenspeichern (pickle) und wird im Ergebnis nicht gebraucht
    return Analysis(integ, settings.N, tuple(curve["ns"]), {m: tuple(v) for m, v in curve["rmse"].items()}, {m: tuple(v) for m, v in curve["factor"].items()}, curve["slope"], fo, fd, fl, pred, ratio, C.REPS)


# --- Experiment 2: Gewinn gegen effektive Dimension -----------------------------------------------------------------------------------------

def dimension_rows_exact(settings: Settings, netz: Netz | None = None, table=None) -> list[dict]:
    """Ohne Simulation: exakte effektive Dimension der Begleiter (Dimension x Gewichte) und der Integranden des gewählten Netzes."""
    rows = []
    for d in C.D_CHOICES:
        for w in C.WEIGHTS:
            dims = A.dim_summary(A.exp_order_shares(A.exp_weights(w, d)))
            rows.append({"label": f"Begleiter d = {d}, {C.WEIGHTS[w]}", "kind": "exp", "d": d, "weight": w, "dims": dims})
    netz = netz if netz is not None else named_netz(settings.netz)
    table = table if table is not None else netz_table(netz)
    for key, p in (("sprung", 0.7), ("sprung", 0.9), ("sprung", 0.99), ("glatt", C.P_SMOOTH)):
        integ = make_integrand(Settings(integrand=key, netz=settings.netz, p=p if key == "sprung" else settings.p), netz, table, key)
        label = f"Sprung, p = {de(p, 2)}" if key == "sprung" else "glatt (unsichere p)"
        rows.append({"label": label, "kind": "netz", "d": integ.d, "key": key, "p": p, "dims": integ.dims})
    return rows


def dimension_experiment(settings: Settings, netz: Netz | None = None, table=None) -> dict:
    """Der exakten mittleren Dimension gegenübergestellt: der gemessene Faktor der Owen-Sobol-Folge und von Latin Hypercube bei N = settings.N (GRID_REPS Wiederholungen je Zeile)."""
    netz = netz if netz is not None else named_netz(settings.netz)
    table = table if table is not None else netz_table(netz)
    N = settings.N
    rows = []
    for row in dimension_rows_exact(settings, netz, table):
        if row["kind"] == "exp":
            integ = make_integrand(Settings(integrand="exp", d=row["d"], weight=row["weight"], nexp=settings.nexp, seed=settings.seed))
        else:
            integ = make_integrand(Settings(integrand=row["key"], netz=settings.netz, p=row["p"], seed=settings.seed), netz, table, row["key"])
        fo = _factor(_errors(integ, "sobol_owen", N, C.GRID_REPS, settings.seed, 2), integ.var, N)
        fl = _factor(_errors(integ, "lhs", N, C.GRID_REPS, settings.seed, 2), integ.var, N)
        dims = row["dims"]
        rows.append({**row, "factor_owen": fo, "factor_lhs": fl, "lhs_pred": 1.0 - dims["additive"]})
    return {"rows": rows, "N": N, "netz": netz.name}


# --- Experiment 3: Glattheitsleiter am selben Netz -------------------------------------------------------------------------------------------

def smoothness_ladder(settings: Settings, netz: Netz | None = None, table=None) -> dict:
    """Dasselbe Netz, drei Integranden von Sprung über Knick bis Polynom, alle vier Punktmengen bei N = settings.N."""
    netz = netz if netz is not None else named_netz(settings.netz)
    table = table if table is not None else netz_table(netz)
    N = settings.N
    rows = []
    for key in ("sprung", "engpass", "glatt"):
        integ = make_integrand(settings, netz, table, key)
        facs = {m: _factor(_errors(integ, m, N, C.GRID_REPS, settings.seed, 3), integ.var, N) for m in A.METHODS}
        rows.append({"key": key, "label": integ.label, "d": integ.d, "factors": facs, "dims": integ.dims})
    return {"rows": rows, "N": N, "netz": netz.name}


# --- Experiment 4: Fehlerbalken -------------------------------------------------------------------------------------------------------------

def interval_experiment(settings: Settings, integ: Integrand | None = None) -> dict:
    """Überdeckung des 95-%-Intervalls: Zufall, Owen mit naivem Standardfehler, Owen mit K = 4, 8, 16, 32 unabhängigen Randomisierungen (Blöcke zu N/K Punkten)."""
    integ = integ if integ is not None else make_integrand(settings)
    N, R, seed = settings.N, C.CI_REPS, settings.seed
    stats = {"mc": ([], []), "naiv": ([], [])}
    ks = [K for K in C.K_CHOICES if N // K >= 2]
    for K in ks:
        stats[K] = ([], [])
    for r in range(R):
        y = integ.f(A.points("mc", N, integ.d, A.stream_seed(seed, 4, r, 0)))
        est, hw = A.naive_interval(y)
        stats["mc"][0].append(est - integ.truth)
        stats["mc"][1].append(hw)
        y = integ.f(A.points("sobol_owen", N, integ.d, A.stream_seed(seed, 4, r, 1)))
        est, hw = A.naive_interval(y)
        stats["naiv"][0].append(est - integ.truth)
        stats["naiv"][1].append(hw)
        for K in ks:
            means = np.array([integ.f(A.points("sobol_owen", N // K, integ.d, A.stream_seed(seed, 4, r, 2, K, b))).mean() for b in range(K)])
            est, hw = A.block_interval(means)
            stats[K][0].append(est - integ.truth)
            stats[K][1].append(hw)
    rows = []
    for name, (errs, hws) in stats.items():
        errs, hws = np.array(errs), np.array(hws)
        rmse = float(np.sqrt(np.mean(errs**2)))
        label = "Zufall (Monte Carlo)" if name == "mc" else "Owen, naiver Standardfehler" if name == "naiv" else f"Owen, K = {name} Blöcke"
        rows.append({"name": name, "label": label, "coverage": float(np.mean(np.abs(errs) <= hws)), "rmse": rmse, "hw_ratio": float(np.mean(hws) / rmse) if rmse > 0 else float("nan")})
    single = next(r for r in rows if r["name"] == "naiv")["rmse"]
    mc = next(r for r in rows if r["name"] == "mc")["rmse"]
    for r in rows:
        r["vs_single"] = r["rmse"] / single if single > 0 else float("nan")
        r["vs_mc"] = r["rmse"] / mc if mc > 0 else float("nan")
    return {"rows": rows, "N": N, "label": integ.label, "reps": R}


# --- Experiment 5: Punktzahl und Reihenfolge ------------------------------------------------------------------------------------------------

def size_rows(settings: Settings, integ: Integrand | None = None) -> list[dict]:
    """Die ersten N Sobol-Punkte für N = 2^k, 2^k - 24, 2^k + 1, 1,25 · 2^k und 1,5 · 2^k: Faktor von Owen und von digitaler Verschiebung."""
    integ = integ if integ is not None else make_integrand(settings)
    base = settings.N
    sizes = [("Zweierpotenz", base), ("24 Punkte weniger", base - 24), ("ein Punkt mehr", base + 1), ("das 1,25-Fache", int(base * 1.25)), ("das 1,5-Fache", int(base * 1.5))]
    rows = []
    for name, N in sizes:
        if N < 2:
            continue
        fo = _factor(_errors(integ, "sobol_owen", N, C.GRID_REPS * 2, settings.seed, 5), integ.var, N)
        fd = _factor(_errors(integ, "sobol_ds", N, C.GRID_REPS * 2, settings.seed, 5), integ.var, N)
        rows.append({"name": name, "N": N, "factor_owen": fo, "factor_ds": fd})
    return rows


def order_rows(settings: Settings) -> list[dict]:
    """Reihenfolge der Koordinaten bei abfallenden Gewichten (a_i = 1/i, d = 64): wichtige zuerst gegen zuletzt, bei N = settings.N."""
    d = C.MAX_DIM
    a = A.exp_weights("harmonisch", d)
    rows = []
    for name, aa in (("wichtige Koordinaten zuerst", a), ("wichtige Koordinaten zuletzt", a[::-1].copy())):
        integ = Integrand("exp", name, d, A.exp_exact(aa), A.exp_var_exact(aa), lambda u, aa=aa: A.f_exp(u, aa), None)
        rows.append({"name": name, "factor_owen": _factor(_errors(integ, "sobol_owen", settings.N, C.GRID_REPS, settings.seed, 6), integ.var, settings.N),
                     "factor_ds": _factor(_errors(integ, "sobol_ds", settings.N, C.GRID_REPS, settings.seed, 6), integ.var, settings.N)})
    return rows


def size_order_experiment(settings: Settings, integ: Integrand | None = None) -> dict:
    return {"sizes": size_rows(settings, integ), "order": order_rows(settings), "N": settings.N}


def custom_result(settings: Settings, netz: Netz) -> dict:
    """Spielwiese: das eigene Netz mit dem gewählten Netz-Integranden (beim Begleiter: der Engpass)."""
    table = netz_table(netz)
    key = "engpass" if settings.integrand == "exp" else settings.integrand
    return {"analysis": analyse(settings, netz, table, key), "key": key}


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

def _dim_sentence(a: Analysis) -> str:
    dims = a.integ.dims
    if dims is None:
        return "Für den Engpass ist keine exakte ANOVA berechnet."
    return (f"Exakte effektive Dimension: mittlere Dimension {de(dims['mean_dim'], 2)}, 90 % der Varianz bis Ordnung {dims['sup90']}, additiver Anteil {de(dims['additive'] * 100, 0)} % "
            f"(so viel kann Latin Hypercube höchstens holen).")


def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Kernfrage: Wann schlägt die Sobol-Folge die Zufallsziehung?"""
    if not a.integ.var > 0:
        return "info", "Der Integrand ist hier konstant (Varianz 0): es gibt nichts zu schätzen."
    fo = a.factor_owen
    head = (f"{a.integ.label}, {a.integ.d} Dimensionen, N = {de(a.N, 0)} Punkte: Der mittlere quadratische Fehler der Sobol-Folge mit Owen-Scrambling hat gegenüber der Monte-Carlo-Varianz den Faktor {faktor(fo)}, "
            f"der RMSE ist also {dn(a.ratio, 1)}-mal kleiner als beim Zufall (digitale Verschiebung: Faktor {faktor(a.factor_ds)}; Latin Hypercube: Faktor {faktor(a.factor_lhs)}). ")
    if fo < 0.5:
        kind, tail = "success", "Die Sobol-Folge gewinnt deutlich" + (" – um Größenordnungen." if fo < 0.01 else ".")
    elif fo <= 1.3:
        kind, tail = "info", "Höchstens kleiner Gewinn: ein Faktor zwischen 0,5 und 1,3 ist vom Zufall schwer zu trennen (Messrauschen etwa ±15 % bei 100 Wiederholungen); unterhalb von etwa 0,8 ist der Gewinn klein, aber vorhanden."
    else:
        kind, tail = "warning", "Hier ist die Sobol-Folge schlechter als der Zufall."
    return kind, head + tail + " " + _dim_sentence(a)


def verdict_curve(a: Analysis) -> tuple[str, str]:
    s = a.slope
    if not np.isfinite(s["sobol_owen"]):
        return "info", "Für eine Steigung braucht die Kurve mindestens drei Punktzahlen (N ab 128)."
    text = (f"Steigung von log RMSE über log N: Zufall {de(s['mc'], 2)}, Latin Hypercube {de(s['lhs'], 2)}, Sobol mit Verschiebung {de(s['sobol_ds'], 2)}, Sobol mit Owen {de(s['sobol_owen'], 2)}. "
            f"Zufall fällt mit −0,5; ein steilerer Abfall ist der Gewinn der gleichmäßigen Punkte. ")
    if a.integ.dims is not None:
        text += (f"Latin Hypercube holt nur den additiven Anteil: vorhergesagter Faktor {faktor(a.lhs_pred)}, gemessen {faktor(a.factor_lhs)} (die Vorhersage gilt asymptotisch; bei großer Dimension und endlichem N weicht die Messung ab). ")
    text += "Die Steigung über das ganze Fenster mischt kleine und große N; die asymptotische Rate gilt erst bei großem N und kleiner effektiver Dimension."
    return "info", text


def verdict_dimension(exp: dict, settings: Settings) -> tuple[str, str]:
    rows = exp["rows"]
    low = [r for r in rows if r["kind"] == "exp" and r["d"] == 64 and r["weight"] == "harmonisch"]
    high = [r for r in rows if r["kind"] == "exp" and r["d"] == 32 and r["weight"] == "gleich"]
    text = f"Bei N = {de(exp['N'], 0)}: "
    if low and high:
        text += (f"64 Dimensionen mit abfallenden Gewichten (mittlere Dimension {de(low[0]['dims']['mean_dim'], 2)}) haben den Faktor {faktor(low[0]['factor_owen'])}, "
                 f"32 Dimensionen mit gleichen Gewichten (mittlere Dimension {de(high[0]['dims']['mean_dim'], 2)}) nur {faktor(high[0]['factor_owen'])}: die nominale Dimension allein sagt wenig, die effektive Dimension und die Glattheit bestimmen den Gewinn. ")
    netz_rows = [r for r in rows if r["kind"] == "netz"]
    if netz_rows:
        text += f"{exp['netz']}: " + "; ".join(f"{r['label']}: mittlere Dimension {de(r['dims']['mean_dim'], 2)}, Faktor {faktor(r['factor_owen'])}" for r in netz_rows) + ". "
    text += "Die mittlere Dimension allein erklärt den Gewinn nicht: bei ähnlicher mittlerer Dimension gewinnt ein Sprung weniger als ein glatter Integrand."
    return "info", text


def verdict_ladder(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    f = {r["key"]: r["factors"]["sobol_owen"] for r in rows}
    names = {"sprung": "Sprung", "engpass": "Engpass", "glatt": "glatt"}
    rang = sorted(rows, key=lambda r: r["factors"]["sobol_owen"])
    text = (f"{exp['netz']}, N = {de(exp['N'], 0)}, Faktor der Owen-Sobol-Folge: Sprung {faktor(f['sprung'])}, Engpass (stetig mit Knicken) {faktor(f['engpass'])}, glatt {faktor(f['glatt'])}. "
            f"Rangfolge vom größten Gewinn an: {' vor '.join(names[r['key']] for r in rang)}. Das Polynom gewinnt am meisten; Sprung und Knick liegen näher beieinander, ihr Abstand geht oft im Rauschen unter. "
            f"Latin Hypercube holt bei dem glatten Integranden {faktor(rows[2]['factors']['lhs'])}.")
    return ("info" if rang[0]["key"] == "glatt" else "warning"), text


def verdict_intervals(exp: dict) -> tuple[str, str]:
    rows = {r["name"]: r for r in exp["rows"]}
    naive = rows["naiv"]
    text = (f"{exp['label']}, N = {de(exp['N'], 0)}, {exp['reps']} Wiederholungen: Zufall {de(rows['mc']['coverage'], 3)} Überdeckung. Behandelt man die Owen-Punkte wie unabhängige Ziehungen, "
            f"ist die Halbbreite das {de(naive['hw_ratio'], 1)}-Fache des wahren RMSE und die Überdeckung {de(naive['coverage'], 3)}. ")
    blocks = [r for r in exp["rows"] if isinstance(r["name"], int)]
    if blocks:
        text += "Mit unabhängigen Randomisierungen: " + "; ".join(f"K = {r['name']}: Überdeckung {de(r['coverage'], 3)}, RMSE das {de(r['vs_single'], 1)}-Fache einer einzigen Menge" for r in blocks) + ". "
    text += "Der ehrliche Fehlerbalken kostet Genauigkeit."
    return "info", text


def verdict_sizes(exp: dict) -> tuple[str, str]:
    sizes = {r["name"]: r for r in exp["sizes"]}
    order = exp["order"]
    base, minus = sizes["Zweierpotenz"], sizes["24 Punkte weniger"]
    text = (f"Faktor der Owen-Sobol-Folge: bei N = {de(base['N'], 0)} {faktor(base['factor_owen'])}, bei N = {de(minus['N'], 0)} {faktor(minus['factor_owen'])}, "
            f"also das {de(minus['factor_owen'] / base['factor_owen'], 2)}-Fache; bei dem 1,5-Fachen {faktor(sizes['das 1,5-Fache']['factor_owen'])}. "
            f"Reihenfolge bei abfallenden Gewichten (d = 64, a_i = 1/i): wichtige zuerst {faktor(order[0]['factor_owen'])}, zuletzt {faktor(order[1]['factor_owen'])} (Owen), "
            f"{faktor(order[0]['factor_ds'])} gegen {faktor(order[1]['factor_ds'])} (digitale Verschiebung).")
    return "info", text
