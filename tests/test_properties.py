"""Eigenschaften (ORACLE-PLAYBOOK §5): was für ALLE Eingaben gelten muss, geprüft an mehreren Instanzen und mit Bändern für das Rauschen."""

import numpy as np
import pytest

import qmc_algorithm as A
import qmc_constants as C
import qmc_evaluation as ev
from qmc_evaluation import Settings


@pytest.mark.parametrize("d", C.D_CHOICES)
@pytest.mark.parametrize("w", list(C.WEIGHTS))
def test_ordnungsanteile_sind_eine_verteilung(d, w):
    sh = A.exp_order_shares(A.exp_weights(w, d))
    assert sh[0] == 0 and sh.sum() == pytest.approx(1.0) and (sh >= 0).all()


def test_mittlere_dimension_waechst_mit_der_dimension_bei_gleichen_gewichten_und_bleibt_klein_bei_abfallenden():
    gleich = [A.dim_summary(A.exp_order_shares(A.exp_weights("gleich", d)))["mean_dim"] for d in C.D_CHOICES]
    harm = [A.dim_summary(A.exp_order_shares(A.exp_weights("harmonisch", d)))["mean_dim"] for d in C.D_CHOICES]
    assert all(b > a for a, b in zip(gleich, gleich[1:]))
    assert max(harm) < 1.1 and gleich[-1] > 4


def test_additiver_anteil_ist_ein_anteil_und_die_ueberlagerung_wird_erreicht(tab_hantel, hantel):
    for p in (0.5, 0.7, 0.9, 0.99):
        d = A.dim_summary(A.order_shares(A.anova_coeffs(tab_hantel, hantel.m, p) ** 2, hantel.m))
        assert 0 <= d["additive"] <= 1 and 1 <= d["mean_dim"] <= hantel.m and 1 <= d["sup90"] <= d["sup99"] <= hantel.m


def test_zufall_hat_faktor_eins():
    """Monte Carlo ist per Definition der Maßstab: der gemessene Faktor liegt über alle Punktzahlen im Mittel bei 1."""
    a = ev.analyse(Settings(integrand="exp", d=8, nexp=8))
    assert np.mean(a.factor["mc"]) == pytest.approx(1.0, abs=0.15)


def test_sobol_gewinnt_beim_glatten_begleiter_mit_niedriger_effektiver_dimension_und_der_gewinn_waechst_mit_n():
    a = ev.analyse(Settings(integrand="exp", d=4, nexp=9))
    assert a.factor_owen < 0.05 and a.factor["sobol_owen"][-1] < a.factor["sobol_owen"][0]
    assert a.slope["sobol_owen"] < -0.9 and a.slope["mc"] == pytest.approx(-0.5, abs=0.12)


def test_gewinn_ist_bei_gleichen_gewichten_in_hoher_dimension_kleiner_als_in_niedriger():
    lo = ev.analyse(Settings(integrand="exp", d=4, nexp=8))
    hi = ev.analyse(Settings(integrand="exp", d=32, nexp=8))
    assert hi.factor_owen > 10 * lo.factor_owen


def test_nominale_dimension_ist_nicht_entscheidend_abfallende_gewichte_in_64_dimensionen_schlagen_gleiche_in_16():
    harm64 = ev.analyse(Settings(integrand="exp", d=64, weight="harmonisch", nexp=8))
    gleich16 = ev.analyse(Settings(integrand="exp", d=16, weight="gleich", nexp=8))
    assert harm64.factor_owen < gleich16.factor_owen


def test_glattheitsleiter_am_selben_netz_ist_geordnet(hantel, tab_hantel):
    rows = ev.smoothness_ladder(Settings(integrand="glatt", netz="hantel", nexp=9))["rows"]
    f = {r["key"]: r["factors"]["sobol_owen"] for r in rows}
    assert f["glatt"] < f["engpass"] < f["sprung"] < 1.0


def test_naiver_standardfehler_ueberdeckt_zu_oft_und_bloecke_halten_das_niveau():
    ex = ev.interval_experiment(Settings(integrand="glatt", netz="hantel", nexp=8))
    rows = {r["name"]: r for r in ex["rows"]}
    assert rows["naiv"]["coverage"] >= 0.99 and rows["naiv"]["hw_ratio"] > 3
    assert 0.88 <= rows["mc"]["coverage"] <= 0.99
    for K in C.K_CHOICES:
        assert 0.88 <= rows[K]["coverage"] <= 1.0, K
    assert rows[32]["vs_single"] > rows[4]["vs_single"] > 1.0


def test_nicht_zweierpotenz_kostet_beim_glatten_integranden_genauigkeit():
    rows = {r["name"]: r for r in ev.size_rows(Settings(integrand="exp", d=8, nexp=8))}
    assert rows["24 Punkte weniger"]["factor_owen"] > 1.3 * rows["Zweierpotenz"]["factor_owen"]
    assert rows["das 1,5-Fache"]["factor_owen"] < 1.5 * rows["Zweierpotenz"]["factor_owen"]


def test_bei_abfallenden_gewichten_gehoeren_die_wichtigen_koordinaten_nach_vorn():
    rows = ev.order_rows(Settings(nexp=10))
    assert rows[0]["factor_owen"] < rows[1]["factor_owen"]


def test_der_schaetzer_ist_in_der_vollen_pipeline_erwartungstreu():
    integ = ev.make_integrand(Settings(integrand="exp", d=8))
    err = ev._errors(integ, "sobol_owen", 256, 200, 3, 1)
    assert abs(err.mean()) < 4 * err.std(ddof=1) / np.sqrt(len(err))


def test_gleiche_einstellung_gleiches_ergebnis():
    s = Settings(integrand="exp", d=4, nexp=7)
    a, b = ev.analyse(s), ev.analyse(s)
    assert a.rmse == b.rmse and a.factor_owen == b.factor_owen
    assert ev.analyse(Settings(integrand="exp", d=4, nexp=7, seed=s.seed + 1)).rmse != a.rmse
