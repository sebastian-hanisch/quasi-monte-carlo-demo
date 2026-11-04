"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import qmc_algorithm as A
import qmc_constants as C
import qmc_evaluation as ev
from qmc_evaluation import Settings
from qmc_scenario import Netz, named_netz, parse_edges


def test_eine_dimension_und_ein_punkt():
    assert A.sobol_ints(1, 1).tolist() == [[0]]
    for method in A.METHODS:
        u = A.points(method, 1, 1, 3)
        assert u.shape == (1, 1) and 0 <= u[0, 0] < 1


def test_zwei_punkte_sind_ein_netz_und_die_dimensionsgrenzen():
    u = A.points("sobol_owen", 2, 1, 4)
    assert sorted(np.floor(u[:, 0] * 2).astype(int).tolist()) == [0, 1]
    assert A.points("sobol_owen", 4, C.MAX_DIM, 1).shape == (4, C.MAX_DIM)
    for bad in (0, C.MAX_DIM + 1):
        with pytest.raises(ValueError):
            A.direction_ints(bad)
    with pytest.raises(ValueError):
        A.sobol_ints(0, 2)


def test_unbekannte_namen_werfen_fehler():
    with pytest.raises(ValueError):
        A.points("halton", 4, 2, 1)
    with pytest.raises(ValueError):
        A.exp_weights("kubisch", 4)
    with pytest.raises(ValueError):
        ev.make_integrand(Settings(), key="unbekannt")


def test_blockintervall_nur_fuer_die_tabellierten_k():
    with pytest.raises(ValueError):
        A.block_interval(np.arange(5.0))
    est, hw = A.block_interval(np.full(4, 2.0))
    assert est == 2.0 and hw == 0.0


def test_netz_mit_zu_vielen_kanten_wird_nicht_aufgezaehlt():
    edges = tuple((i, j) for i in range(7) for j in range(i + 1, 7))          # K7 hat 21 Kanten
    with pytest.raises(ValueError):
        ev.netz_table(Netz("K7", 7, edges, tuple((0.0, 0.0) for _ in range(7))))


def test_konstanter_integrand_hat_varianz_null_und_keinen_unsinn():
    """Ein Knoten mit einer Schleife: immer zusammenhängend, also Wert 1 und Varianz 0; keine Division durch null, Faktoren nicht bestimmbar."""
    netz = Netz("Einzelknoten", 1, ((0, 0),), ((0.0, 0.0),))
    s = Settings(integrand="sprung", netz="hantel", nexp=6)
    a = ev.analyse(s, netz, ev.netz_table(netz))
    assert a.integ.truth == 1.0 and a.integ.var == 0.0 and np.isnan(a.factor_owen)
    assert ev.verdict_main(a, s)[0] == "info" and "konstant" in ev.verdict_main(a, s)[1]


def test_engpass_einer_einzelnen_kante_ist_die_gleichverteilte_selbst():
    netz = Netz("Kante", 2, ((0, 1),), ((0.0, 0.0), (1.0, 0.0)))
    tab = A.conn_table(2, netz.edges)
    u = np.array([[0.25], [0.5], [0.875]])
    assert A.f_engpass(u, tab).tolist() == [0.25, 0.5, 0.875]
    Nk = A.conn_counts(tab, 1)
    assert A.engpass_exact(Nk) == pytest.approx(0.5) and A.engpass_var_exact(Nk) == pytest.approx(1 / 12)


def test_kleinste_punktzahl_aller_experimente():
    s = Settings(integrand="exp", d=4, nexp=C.NEXP_MIN)
    assert ev.interval_experiment(s)["rows"]                   # N = 64: K = 32 hat nur zwei Punkte je Block
    assert ev.size_rows(s) and ev.analyse(s).ns[0] == 32


def test_eingabepruefung_der_spielwiese():
    with pytest.raises(ValueError, match="Schleife"):
        parse_edges("0-0")
    with pytest.raises(ValueError, match="Keine Kanten"):
        parse_edges("\n# nur Kommentar\n")
    with pytest.raises(ValueError, match="Doppelte Kante"):
        parse_edges("0-1\n1-0")
    with pytest.raises(ValueError, match="erwartet zwei ganze Zahlen"):
        parse_edges("0-1-2")
    with pytest.raises(ValueError, match="höchstens 17"):
        parse_edges("\n".join(f"{i}-{j}" for i in range(7) for j in range(i + 1, 7)))
    with pytest.raises(ValueError, match="höchstens 14"):
        parse_edges("\n".join(f"{i}-{i + 1}" for i in range(15)))


def test_knoten_werden_von_null_an_umnummeriert():
    netz = parse_edges("10-20\n20,30\n# Kommentar\n30 10")
    assert netz.n == 3 and netz.m == 3 and set(netz.edges) == {(0, 1), (1, 2), (0, 2)}


def test_spielwiese_mit_eigenem_netz_und_beim_begleiter_der_engpass():
    netz = parse_edges("0-1\n1-2\n2-0\n3-4\n4-5\n5-3\n2-3")
    r = ev.custom_result(Settings(integrand="exp", nexp=6), netz)
    assert r["key"] == "engpass" and r["analysis"].integ.d == 7 and r["analysis"].integ.truth == pytest.approx(0.285714, abs=1e-5)
    assert ev.custom_result(Settings(integrand="glatt", nexp=6), netz)["key"] == "glatt"


def test_benannte_netze_haben_die_erwartete_kantenzahl():
    assert [named_netz(k).m for k in C.NETZE] == [7, 12, 17]


def test_formate():
    assert ev.de(1234.567, 1) == "1.234,6" and ev.de(-0.5, 2) == "−0,50" and ev.de_e(0.000042) == "4,2e-05"
    assert ev.dn(float("nan")) == "nicht bestimmbar" and ev.dn(float("inf")) == "unendlich"
    assert ev.dz(2.4e8) == "2,4e+08" and ev.dz(239700) == "239.700" and ev.dz(29.74) == "29,7"
    assert ev.faktor(0.00003) == "unter 0,0001" and ev.faktor(0.0029) == "0,0029" and ev.faktor(0.4231) == "0,423" and ev.faktor(1.2249) == "1,22" and ev.faktor(float("nan")) == "nicht bestimmbar"
    assert ev.kanten(1) == "1 Kante" and ev.kanten(2) == "2 Kanten"
