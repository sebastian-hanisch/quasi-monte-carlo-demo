"""Unabhängige Orakel (ORACLE-PLAYBOOK §5): scipy für die Sobol-Folge und Verteilungen, networkx für den Zusammenhang, Brute-Force-Aufzählungen und Quadratur für Wahrheit, Varianz und ANOVA.

Hier steht, was der Kern nicht selbst beweisen darf: dass die eingebetteten Richtungszahlen dieselbe Folge wie scipy ergeben, dass Verschiebung und Owen-Scrambling die Netz-Eigenschaft und die Gleichverteilung wirklich erhalten,
dass die Schätzer erwartungstreu sind, und dass Wahrheit, Varianz und effektive Dimension der Integranden stimmen."""

import itertools

import networkx as nx
import numpy as np
import pytest
from scipy.stats import kstest, qmc

import qmc_algorithm as A
from qmc_scenario import grid_netz


# --- Sobol-Folge ----------------------------------------------------------------------------------------------------------------------

def test_eingebettete_richtungszahlen_geben_dieselbe_folge_wie_scipy():
    for d, N in ((64, 4096), (17, 1024), (1, 64), (2, 8)):
        ours = A.sobol_ints(N, d).astype(np.float64) / 2.0**32
        theirs = qmc.Sobol(d, scramble=False).random(N)
        assert (ours == theirs).all(), (d, N)


def _cell_counts(u: np.ndarray, shape: tuple[int, ...]) -> list[int]:
    """Zahl der Punkte in jedem dyadischen Kästchen mit a_j Teilungen in Koordinate j (Rechteck mit Inhalt 2^-sum(a)), sortiert."""
    idx = np.zeros(len(u), dtype=np.int64)
    for j, a in enumerate(shape):
        idx = idx * (1 << a) + np.floor(u[:, j] * (1 << a)).astype(np.int64)
    return sorted(np.bincount(idx, minlength=int(np.prod([1 << a for a in shape]))).tolist())


def _shapes(d: int, k: int):
    return [s for s in itertools.product(range(k + 1), repeat=d) if sum(s) == k]


@pytest.mark.parametrize("method", ["sobol_roh", "sobol_ds", "sobol_owen"])
def test_netz_eigenschaft_in_zwei_dimensionen_aus_der_definition(method):
    """N = 64 Punkte, Dimensionen 1 und 2: jedes Kästchen vom Inhalt 1/64 enthält genau einen Punkt (t = 0)."""
    u = A.points(method, 64, 2, 5)
    for shape in _shapes(2, 6):
        assert _cell_counts(u, shape) == [1] * 64, shape


def test_randomisierung_aendert_die_kaestchenzahlen_in_drei_dimensionen_nicht():
    """Für alle Formen mit Inhalt 1/64 in drei Dimensionen zählen die randomisierten Mengen dieselben Punkte je Kästchen wie die Folge (die Netz-Eigenschaft bleibt mit demselben t erhalten)."""
    raw = A.points("sobol_roh", 64, 3, 5)
    for method in ("sobol_ds", "sobol_owen"):
        u = A.points(method, 64, 3, 9)
        for shape in _shapes(3, 6):
            assert _cell_counts(u, shape) == _cell_counts(raw, shape), (method, shape)


@pytest.mark.parametrize("method", ["sobol_roh", "sobol_ds", "sobol_owen"])
def test_jede_einzelkoordinate_hat_einen_punkt_je_intervall(method):
    u = A.points(method, 1024, 17, 3)
    for j in range(17):
        assert _cell_counts(u[:, j:j + 1], (10,)) == [1] * 1024, j


@pytest.mark.parametrize("method", ["sobol_ds", "sobol_owen"])
def test_ein_einzelner_punkt_ist_gleichverteilt(method):
    """Der zweite Punkt, erste und dritte Koordinate, über viele Randomisierungen: Kolmogorov–Smirnov gegen die Gleichverteilung."""
    vals = np.array([A.points(method, 8, 3, s)[1] for s in range(1500)])
    for j in (0, 2):
        assert kstest(vals[:, j], "uniform").pvalue > 0.001


@pytest.mark.parametrize("method", ["sobol_ds", "sobol_owen"])
def test_schaetzer_ist_erwartungstreu(method):
    """∫ u1 u2 u3 = 1/8 mit nur acht Punkten: Mittel über viele Randomisierungen im Vier-Sigma-Band (der unrandomisierte Schätzer ist verzerrt)."""
    est = np.array([A.points(method, 8, 3, s).prod(axis=1).mean() for s in range(3000)])
    se = est.std(ddof=1) / np.sqrt(len(est))
    assert abs(est.mean() - 0.125) < 4 * se
    assert abs(A.points("sobol_roh", 8, 3, 0).prod(axis=1).mean() - 0.125) > 0.01


def test_owen_hat_kleinere_varianz_als_zufall_bei_glattem_integranden():
    est = {m: np.array([A.points(m, 64, 3, s).prod(axis=1).mean() for s in range(400)]) for m in ("mc", "sobol_owen")}
    assert est["sobol_owen"].var() < 0.2 * est["mc"].var()


# --- Netze und Integranden ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("netz", [grid_netz(2, 3), grid_netz(3, 3)], ids=["2x3", "3x3"])
def test_zusammenhangstabelle_gegen_networkx(netz):
    tab = A.conn_table(netz.n, netz.edges)
    for mask in range(0, 1 << netz.m, 7):
        g = nx.Graph()
        g.add_nodes_from(range(netz.n))
        g.add_edges_from(e for i, e in enumerate(netz.edges) if (mask >> i) & 1)
        assert bool(tab[mask]) == nx.is_connected(g), mask


def test_zusammenhangstabelle_der_hantel_gegen_networkx(hantel, tab_hantel):
    for mask in range(1 << hantel.m):
        g = nx.Graph()
        g.add_nodes_from(range(hantel.n))
        g.add_edges_from(e for i, e in enumerate(hantel.edges) if (mask >> i) & 1)
        assert bool(tab_hantel[mask]) == nx.is_connected(g)


def test_glatter_integrand_gegen_die_direkte_summe_ueber_alle_zustaende(gitter33, tab_gitter33):
    u = A.points("mc", 4, gitter33.m, 17)
    p = 0.7 + 0.4 * (u - 0.5)
    states = np.arange(1 << gitter33.m)
    direct = []
    for b in range(4):
        w = np.ones(len(states))
        for e in range(gitter33.m):
            w *= np.where((states >> e) & 1 == 1, p[b, e], 1 - p[b, e])
        direct.append(float((w * tab_gitter33).sum()))
    assert A.f_glatt(u, tab_gitter33).tolist() == pytest.approx(direct, abs=1e-12)


def test_wahrheit_und_varianz_der_netz_integranden_gegen_grosse_monte_carlo_laeufe(hantel, tab_hantel):
    Nk = A.conn_counts(tab_hantel, hantel.m)
    u = A.points("mc", 400_000, hantel.m, 77)
    for name, vals, truth, var in (("sprung", A.f_sprung(u, 0.9, tab_hantel), A.r_exact(Nk, 0.9), A.r_exact(Nk, 0.9) * (1 - A.r_exact(Nk, 0.9))),
                                   ("engpass", A.f_engpass(u, tab_hantel), A.engpass_exact(Nk), A.engpass_var_exact(Nk)),
                                   ("glatt", A.f_glatt(u[:100_000], tab_hantel), A.r_exact(Nk, 0.7), A.glatt_var_exact(tab_hantel, hantel.m))):
        se = np.sqrt(var / len(vals))
        assert abs(vals.mean() - truth) < 4.5 * se, name
        assert vals.var() == pytest.approx(var, rel=0.03), name


def test_exakte_varianz_des_begleiters_gegen_quadratur():
    """exp(u1 + 0,5 u2 + 0,25 u3): Wert und Varianz über Gauß–Legendre-Quadratur in drei Dimensionen."""
    a = np.array([1.0, 0.5, 0.25])
    x, w = np.polynomial.legendre.leggauss(24)
    x, w = (x + 1) / 2, w / 2
    X = np.meshgrid(x, x, x, indexing="ij")
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    f = np.exp(a[0] * X[0] + a[1] * X[1] + a[2] * X[2])
    mean = (W * f).sum()
    assert mean == pytest.approx(A.exp_exact(a), rel=1e-12)
    assert (W * f * f).sum() - mean**2 == pytest.approx(A.exp_var_exact(a), rel=1e-10)


# --- ANOVA -----------------------------------------------------------------------------------------------------------------------------

def _anova_brute_force(tab, m, p):
    """σ_S² über Var(E[X | x_S]) und Möbius-Inversion; alle Erwartungen als Summen über die 2^m Zustände (unabhängig vom Schmetterlingsschritt)."""
    states = np.arange(1 << m)
    w = np.ones(1 << m)
    for e in range(m):
        w *= np.where((states >> e) & 1 == 1, p, 1 - p)
    X = tab.astype(float)
    mean = float((w * X).sum())
    V = {}
    for S in range(1 << m):
        key = states & S
        g_sum = np.bincount(key, weights=w * X, minlength=1 << m)
        g_w = np.bincount(key, weights=w, minlength=1 << m)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(g_w > 0, g_sum / g_w, 0.0)
        V[S] = float((g_w * g * g).sum() - mean**2)
    sigma = {}
    for S in range(1 << m):
        total, T = 0.0, S
        while True:
            total += (-1) ** (bin(S).count("1") - bin(T).count("1")) * V[T]
            if T == 0:
                break
            T = (T - 1) & S
        sigma[S] = total
    return sigma


def test_anova_schmetterling_gegen_bedingte_erwartungen_auf_der_hantel(hantel, tab_hantel):
    sigma = _anova_brute_force(tab_hantel, hantel.m, 0.7)
    c = A.anova_coeffs(tab_hantel, hantel.m, 0.7)
    for S in range(1, 1 << hantel.m):
        assert c[S] ** 2 == pytest.approx(sigma[S], abs=1e-12), S


@pytest.mark.parametrize("p,erwartet", [(0.7, 0.597), (0.9, 0.736), (0.99, 0.946)])
def test_additiver_anteil_der_hantel_gleicht_stueck_5(hantel, tab_hantel, p, erwartet):
    sh = A.order_shares(A.anova_coeffs(tab_hantel, hantel.m, p) ** 2, hantel.m)
    assert sh[1] == pytest.approx(erwartet, abs=5e-4)


def test_additiver_anteil_des_gitters_3x4_gleicht_stueck_5():
    netz = grid_netz(3, 4)
    tab = A.conn_table(netz.n, netz.edges)
    for p, erwartet in ((0.7, 0.467), (0.9, 0.214), (0.99, 0.020)):
        sh = A.order_shares(A.anova_coeffs(tab, netz.m, p) ** 2, netz.m)
        assert sh[1] == pytest.approx(erwartet, abs=1e-3)


def test_ordnungsanteile_des_begleiters_gegen_quadratur_und_moebius():
    """Drei Dimensionen, a = (1, 0,5, 2): Var(E[f | u_S]) numerisch aus der Quadratur, dann Möbius-Inversion, dann Summe je Ordnung."""
    a = np.array([1.0, 0.5, 2.0])
    x, w = np.polynomial.legendre.leggauss(24)
    x, w = (x + 1) / 2, w / 2
    X = np.meshgrid(x, x, x, indexing="ij")
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    f = np.exp(sum(a[i] * X[i] for i in range(3)))
    mean = (W * f).sum()
    V = {}
    for S in range(8):
        axes = tuple(i for i in range(3) if not (S >> i) & 1)                    # über die Koordinaten außerhalb von S mitteln
        g = (f * W).sum(axis=axes, keepdims=True) / W.sum(axis=axes, keepdims=True)
        V[S] = float(((g * g) * W).sum() - mean**2)
    sigma = {S: sum((-1) ** (bin(S).count("1") - bin(T).count("1")) * V[T] for T in range(8) if T & S == T) for S in range(1, 8)}
    tot = sum(sigma.values())
    shares = np.array([0.0] + [sum(v for S, v in sigma.items() if bin(S).count("1") == k) / tot for k in (1, 2, 3)])
    assert A.exp_order_shares(a).tolist() == pytest.approx(shares.tolist(), abs=1e-10)


def test_latin_hypercube_faktor_gleicht_eins_minus_additiver_anteil():
    """Stück 5: Var(LHS-Mittel) ≈ (σ² − Σ Haupteffekte)/N. Am Begleiter exp mit 16 gleichen Gewichten, N = 1024, 150 Wiederholungen."""
    a = A.exp_weights("gleich", 16)
    err = np.array([A.f_exp(A.points("lhs", 1024, 16, A.stream_seed(5, r)), a).mean() - A.exp_exact(a) for r in range(150)])
    faktor = float(np.mean(err**2) / (A.exp_var_exact(a) / 1024))
    vorhersage = 1 - A.dim_summary(A.exp_order_shares(a))["additive"]
    assert faktor == pytest.approx(vorhersage, rel=0.25)
