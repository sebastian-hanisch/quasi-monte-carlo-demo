"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Aufzählung, geschlossene Formeln, ANOVA) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen und sind reproduzierbar; sie stehen hier mit Bändern, die das Rauschen
(100 bis 300 Wiederholungen; bei Faktoren um 0,0001 und bei schweren Rändern wie exp mit 32 gleichen Gewichten deutlich mehr als bei häufigen Ereignissen) und kleine Plattformunterschiede abdecken.
Faktoren werden im Verhältnis (mal/durch) verglichen, RMSE-Verhältnisse und Steigungen mit festen Abständen."""

import time

import numpy as np
import pytest

import qmc_algorithm as A
import qmc_evaluation as ev
from qmc_evaluation import Settings
from qmc_scenario import named_netz


def _im_band(wert, soll, mal=1.7):
    return soll / mal <= wert <= soll * mal


def _exp(d, w, nexp=10):
    return ev.analyse(Settings(integrand="exp", d=d, weight=w, nexp=nexp))


# --- exakte effektive Dimension -------------------------------------------------------------------------------------------------------------

def test_exakte_effektive_dimension_der_begleiter():
    soll = {(4, "gleich"): (1.12, 0.885, 2), (8, "gleich"): (1.30, 0.747, 2), (16, "gleich"): (1.69, 0.519, 3), (32, "gleich"): (2.64, 0.229, 4), (64, "gleich"): (4.88, 0.034, 8),
            (16, "quadratisch"): (1.03, 0.975, 1), (64, "harmonisch"): (1.04, 0.961, 1)}
    for (d, w), (md, add, s90) in soll.items():
        dm = A.dim_summary(A.exp_order_shares(A.exp_weights(w, d)))
        assert dm["mean_dim"] == pytest.approx(md, abs=0.006) and dm["additive"] == pytest.approx(add, abs=0.0006) and dm["sup90"] == s90, (d, w)


def test_exakte_effektive_dimension_der_netz_integranden():
    soll = {("hantel", 0.7): (1.71, 0.597, 3, 5), ("hantel", 0.9): (1.34, 0.736, 2, 3), ("hantel", 0.99): (1.06, 0.946, 1, 2),
            ("3x3", 0.7): (2.23, 0.523, 5, 8), ("3x3", 0.9): (2.07, 0.236, 3, 5), ("3x3", 0.99): (2.02, 0.021, 2, 3),
            ("3x4", 0.7): (2.48, 0.467, 5, 9), ("3x4", 0.9): (2.12, 0.214, 3, 5), ("3x4", 0.99): (2.02, 0.020, 2, 3)}
    for (key, p), (md, add, s90, s99) in soll.items():
        net = named_netz(key)
        tab, _ = ev.netz_table(net)
        dm = A.dim_summary(A.order_shares(A.anova_coeffs(tab, net.m, p) ** 2, net.m))
        assert dm["mean_dim"] == pytest.approx(md, abs=0.006) and dm["additive"] == pytest.approx(add, abs=0.0006) and (dm["sup90"], dm["sup99"]) == (s90, s99), (key, p)


def test_glatter_integrand_hat_mittlere_dimension_nahe_eins():
    for key in ("hantel", "3x3", "3x4"):
        net = named_netz(key)
        tab, _ = ev.netz_table(net)
        dm = A.dim_summary(A.order_shares(A.glatt_sigma2(tab, net.m), net.m))
        assert dm["mean_dim"] == pytest.approx(1.02, abs=0.006) and dm["additive"] > 0.97 and dm["sup90"] == 1


def test_wahrheit_der_netz_integranden_von_hand_geprueft():
    net = named_netz("hantel")
    tab, Nk = ev.netz_table(net)
    assert A.engpass_exact(Nk) == pytest.approx(0.285714, abs=1e-5) and A.r_exact(Nk, 0.7) == pytest.approx(0.430259, abs=1e-6)


# --- Fehlerkurve des Begleiters -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("d,w,faktor_soll,ratio_soll", [(8, "gleich", 0.0036, 17.3), (16, "gleich", 0.087, 3.4), (32, "gleich", 0.445, 1.3)])
def test_begleiter_gewinn_nimmt_mit_der_dimension_ab(d, w, faktor_soll, ratio_soll):
    a = _exp(d, w)
    assert _im_band(a.factor_owen, faktor_soll, 1.8) and a.ratio == pytest.approx(ratio_soll, rel=0.3)


def test_begleiter_in_kleiner_dimension_hat_faktor_unter_0_0001_bei_n_1024():
    for d, ratio in ((2, 633), (4, 153)):
        a = _exp(d, "gleich")
        assert a.factor_owen < 1e-4 and a.ratio == pytest.approx(ratio, rel=0.35), d


def test_abfallende_gewichte_gewinnen_auch_in_64_dimensionen():
    a = _exp(16, "quadratisch")
    b = _exp(64, "harmonisch")
    assert a.factor_owen < 1e-4 and a.ratio == pytest.approx(412, rel=0.35)
    assert b.factor_owen < 5e-4 and b.ratio == pytest.approx(78, rel=0.3)
    assert b.factor_owen < _exp(16, "gleich").factor_owen          # die nominale Dimension 64 gewinnt mehr als 16 mit gleichen Gewichten


def test_steigungen_der_fehlerkurve():
    soll = {(2, "gleich"): (-1.47, -1.00), (4, "gleich"): (-1.27, -1.05), (8, "gleich"): (-1.04, -1.02), (16, "gleich"): (-0.68, -0.68), (16, "quadratisch"): (-1.37, -1.00), (64, "harmonisch"): (-1.17, -1.03)}
    for (d, w), (owen, ds) in soll.items():
        a = _exp(d, w)
        assert a.slope["sobol_owen"] == pytest.approx(owen, abs=0.2) and a.slope["sobol_ds"] == pytest.approx(ds, abs=0.2), (d, w)
        if d >= 4:
            assert a.slope["mc"] == pytest.approx(-0.5, abs=0.1) and a.slope["lhs"] == pytest.approx(-0.5, abs=0.1)


def test_latin_hypercube_holt_den_additiven_anteil():
    for d, mess, vorher in ((4, 0.114, 0.115), (8, 0.225, 0.253)):
        a = _exp(d, "gleich")
        assert a.lhs_pred == pytest.approx(vorher, abs=0.001) and a.factor_lhs == pytest.approx(mess, rel=0.3)


def test_gleiche_gewichte_in_64_dimensionen_gewinnen_nicht_mehr_als_einen_kleinen_faktor():
    a = _exp(64, "gleich")
    assert 0.1 < a.factor_owen < 1.0 and a.integ.dims["mean_dim"] == pytest.approx(4.88, abs=0.006)


# --- Netz-Integranden -----------------------------------------------------------------------------------------------------------------------

def test_glattheitsleiter_bei_n_1024():
    soll = {"hantel": (0.062, 0.020), "3x3": (0.304, 0.137), "3x4": (0.230, 0.292)}
    for key, (sprung, engpass) in soll.items():
        rows = {r["key"]: r["factors"] for r in ev.smoothness_ladder(Settings(integrand="glatt", netz=key, nexp=10))["rows"]}
        assert _im_band(rows["sprung"]["sobol_owen"], sprung, 1.8) and _im_band(rows["engpass"]["sobol_owen"], engpass, 1.8), key
        assert rows["glatt"]["sobol_owen"] < rows["sprung"]["sobol_owen"] and rows["glatt"]["sobol_owen"] < rows["engpass"]["sobol_owen"]
        assert rows["glatt"]["lhs"] > rows["glatt"]["sobol_owen"]                          # Latin Hypercube holt nur den additiven Anteil
        assert rows["glatt"]["sobol_owen"] < (0.03 if key == "3x4" else 5e-4)


def test_netz_integranden_bei_n_4096():
    soll = {("hantel", "sprung", 0.9): 0.040, ("3x3", "sprung", 0.9): 0.222, ("3x4", "sprung", 0.9): 0.210, ("hantel", "engpass", 0.9): 0.019, ("3x3", "engpass", 0.9): 0.055, ("3x4", "engpass", 0.9): 0.139}
    for (key, ik, p), faktor in soll.items():
        a = ev.analyse(Settings(integrand=ik, netz=key, p=p, nexp=12))
        assert _im_band(a.factor_owen, faktor, 1.8), (key, ik)
        assert a.slope["sobol_owen"] < a.slope["mc"] + 0.0 and a.slope["mc"] == pytest.approx(-0.5, abs=0.1)
        assert -0.8 < a.slope["sobol_owen"] < -0.45


def test_seltene_ausfaelle_gewinnen_kaum():
    """Sprung mit p = 0,99: das Ereignis „unzusammenhängend“ hat 1 % Wahrscheinlichkeit je Kante; der Gewinn bleibt klein."""
    for key, faktor in (("3x3", 0.725), ("3x4", 0.672)):
        a = ev.analyse(Settings(integrand="sprung", netz=key, p=0.99, nexp=12))
        assert 0.4 < a.factor_owen < 1.1 and _im_band(a.factor_owen, faktor, 1.5), key


def test_glatter_netz_integrand_bei_n_4096():
    for key in ("hantel", "3x3"):
        a = ev.analyse(Settings(integrand="glatt", netz=key, nexp=12))
        assert a.factor_owen < 5e-4 and a.slope["sobol_owen"] < -1.0, key


# --- Fehlerbalken ---------------------------------------------------------------------------------------------------------------------------

def test_intervalle_am_begleiter_exp_d8():
    ex = ev.interval_experiment(Settings(integrand="exp", d=8, nexp=10))
    r = {x["name"]: x for x in ex["rows"]}
    assert r["mc"]["coverage"] == pytest.approx(0.95, abs=0.04) and r["naiv"]["coverage"] >= 0.995
    assert r["naiv"]["hw_ratio"] == pytest.approx(33.9, rel=0.3)
    for K, cov in ((4, 0.953), (8, 0.957), (16, 0.960), (32, 0.937)):
        assert r[K]["coverage"] == pytest.approx(cov, abs=0.045), K
    assert r[4]["vs_single"] == pytest.approx(2.0, rel=0.35) and r[8]["vs_single"] == pytest.approx(2.7, rel=0.35) and r[16]["vs_single"] == pytest.approx(4.7, rel=0.35) and r[32]["vs_single"] == pytest.approx(5.9, rel=0.35)
    assert r[4]["vs_mc"] == pytest.approx(0.12, rel=0.35) and r[8]["vs_mc"] == pytest.approx(0.16, rel=0.35) and r[16]["vs_mc"] == pytest.approx(0.29, rel=0.35) and r[32]["vs_mc"] == pytest.approx(0.36, rel=0.35)


def test_intervalle_am_gitter_3x3():
    ex = ev.interval_experiment(Settings(integrand="engpass", netz="3x3", nexp=10))
    r = {x["name"]: x for x in ex["rows"]}
    assert r["naiv"]["coverage"] >= 0.995 and r["naiv"]["hw_ratio"] == pytest.approx(6.2, rel=0.3)
    assert r[8]["coverage"] == pytest.approx(0.96, abs=0.045) and r[8]["vs_single"] == pytest.approx(1.6, rel=0.3)
    ex = ev.interval_experiment(Settings(integrand="sprung", netz="3x3", p=0.9, nexp=10))
    r = {x["name"]: x for x in ex["rows"]}
    assert r["naiv"]["coverage"] >= 0.995 and r["naiv"]["hw_ratio"] == pytest.approx(3.9, rel=0.3)
    assert r[8]["coverage"] == pytest.approx(0.95, abs=0.045) and r[8]["vs_single"] == pytest.approx(1.4, rel=0.3)


# --- Punktzahl und Reihenfolge --------------------------------------------------------------------------------------------------------------

def test_punktzahl_keine_zweierpotenz():
    rows = {r["name"]: r for r in ev.size_rows(Settings(integrand="exp", d=8, nexp=10))}
    assert _im_band(rows["Zweierpotenz"]["factor_owen"], 0.0031, 1.6) and _im_band(rows["24 Punkte weniger"]["factor_owen"], 0.0072, 1.6)
    assert rows["24 Punkte weniger"]["factor_owen"] / rows["Zweierpotenz"]["factor_owen"] == pytest.approx(2.3, rel=0.4)
    assert _im_band(rows["ein Punkt mehr"]["factor_owen"], 0.0040, 1.6) and _im_band(rows["das 1,5-Fache"]["factor_owen"], 0.0038, 1.6) and _im_band(rows["das 1,25-Fache"]["factor_owen"], 0.0060, 1.6)
    assert rows["das 1,5-Fache"]["N"] == 1536 and rows["24 Punkte weniger"]["N"] == 1000


def test_reihenfolge_der_koordinaten_bei_abfallenden_gewichten():
    zuerst, zuletzt = ev.order_rows(Settings(nexp=10))
    assert zuerst["factor_owen"] < 5e-4 and 4e-4 < zuletzt["factor_owen"] < 3e-3
    assert zuletzt["factor_owen"] > 3 * zuerst["factor_owen"] and zuletzt["factor_ds"] > zuerst["factor_ds"]


# --- Kein belegter Verlust ------------------------------------------------------------------------------------------------------------------

def _kern_faktor(f, truth, m, N=1024, R=200, method="sobol_owen"):
    err = np.array([f(A.points(method, N, m, A.stream_seed(5, m, R, r))).mean() - truth for r in range(R)])
    return float(np.mean(err**2) / (truth * (1 - truth) / N))


def test_sprungfamilien_in_hoher_dimension_verlieren_nicht_deutlich_gegen_zufall():
    """Reihe (alle m Kanten halten, p = 0,99) und Parallel (mindestens eine von m hält, p = 0,1) mit dem Kern: der Gewinn schrumpft mit m, schlägt aber im Rahmen des Rauschens nicht in Verlust um."""
    reihe = lambda u: (u.max(axis=1) < 0.99).astype(float)
    parallel = lambda u: (u.min(axis=1) < 0.1).astype(float)
    for m, soll in ((4, 0.03), (16, 0.08), (64, 0.29)):
        assert _im_band(_kern_faktor(reihe, 0.99**m, m), soll, 1.8), m
    for m, soll in ((4, 0.02), (16, 0.33)):
        assert _im_band(_kern_faktor(parallel, 1 - 0.9**m, m), soll, 1.8), m
    mc = _kern_faktor(parallel, 1 - 0.9**64, 64, method="mc")
    owen = _kern_faktor(parallel, 1 - 0.9**64, 64)
    assert 0.6 < mc < 1.5 and 0.6 < owen < 1.6                      # Gewinn verschwunden; Einzelwerte bis 1,2 liegen im Rauschen des Zufalls selbst


# --- Kosten ---------------------------------------------------------------------------------------------------------------------------------

def test_owen_punkte_erzeugen_kostet_ein_vielfaches_der_tabellen_auswertung():
    net = named_netz("3x4")
    tab, _ = ev.netz_table(net)
    u = A.points("mc", 4096, net.m, 1)

    def t(f, k=5):
        best = 1e9
        for _ in range(k):
            s = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - s)
        return best

    assert t(lambda: A.points("sobol_owen", 4096, net.m, 1)) > 10 * t(lambda: A.f_sprung(u, 0.9, tab))
    assert t(lambda: A.points("sobol_owen", 4096, net.m, 1)) > 5 * t(lambda: A.points("mc", 4096, net.m, 1))
