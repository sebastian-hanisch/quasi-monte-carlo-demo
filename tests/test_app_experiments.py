"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import qmc_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_dimension", "die nominale Dimension entscheidet nichts"),
    ("run_ladder", "Rangfolge vom größten Gewinn an"),
    ("run_intervals", "Der ehrliche Fehlerbalken kostet Genauigkeit"),
    ("run_sizes", "Reihenfolge bei abfallenden Gewichten"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_kernfrage_und_kurvenverdikt_stehen_ohne_klick_da():
    t = _texte(_app())
    assert "den Faktor" in t and "Die Sobol-Folge gewinnt deutlich" in t
    assert "Steigung von log RMSE über log N" in t


def test_exakte_tabelle_der_effektiven_dimension_steht_ohne_klick_da():
    at = _app()
    assert len(at.dataframe) >= 1


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_ladder").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="nexp_select").set_value(8).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Rangfolge vom größten Gewinn an" in _texte(at) and "N = 256" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "um Größenordnungen" in _texte(at)
    at.sidebar.button(key="preset_Sprung am Gitter").click().run(timeout=TIMEOUT)
    t = _texte(at)
    assert "Sprung: Gitter 3×4 hängt zusammen" in t and "um Größenordnungen" not in t
    at.sidebar.button(key="preset_Hohe Dimension, abfallende Gewichte").click().run(timeout=TIMEOUT)
    assert "Begleiter exp" in _texte(at) and "64 Dimensionen" in _texte(at)


def test_jeder_integrand_laeuft_durch():
    at = _app()
    for key in C.INTEGRANDS:
        at.sidebar.selectbox(key="integrand_select").set_value(key).run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])


def test_regler_sind_je_nach_integrand_gesperrt():
    at = _app()
    assert at.sidebar.select_slider(key="p_select").disabled and at.sidebar.select_slider(key="d_select").disabled and not at.sidebar.selectbox(key="netz_select").disabled
    at.sidebar.selectbox(key="integrand_select").set_value("exp").run(timeout=TIMEOUT)
    assert at.sidebar.selectbox(key="netz_select").disabled and not at.sidebar.select_slider(key="d_select").disabled
    at.sidebar.selectbox(key="integrand_select").set_value("sprung").run(timeout=TIMEOUT)
    assert not at.sidebar.select_slider(key="p_select").disabled


def test_spielwiese_gueltiges_netz_und_fehlermeldungen():
    at = _app()
    assert not at.error
    at.text_area(key="eigene_kanten").set_value("0-0").run(timeout=TIMEOUT)
    assert not at.exception and "Schleife" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_kanten").set_value("").run(timeout=TIMEOUT)
    assert not at.exception and "Keine Kanten" in " ".join(e.value for e in at.error)


def test_spielwiese_pfad_laeuft_durch():
    at = _app()
    at.text_area(key="eigene_kanten").set_value("0-1\n1-2").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="nexp_select").set_value(8).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["p"] = "0.95"
    at.query_params["d"] = "20"
    at.query_params["e"] = "13"
    at.query_params["i"] = "unbekannt"        # unbekannter Integrand: verworfen
    at.query_params["g"] = "9x9"              # unbekanntes Netz: verworfen
    at.query_params["w"] = "quadratisch"
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.select_slider(key="p_select").value == 0.99                            # 0,95 liegt näher an 0,99 als an 0,9
    assert at.sidebar.select_slider(key="d_select").value == 16                              # 20 liegt näher an 16 als an 32
    assert at.sidebar.select_slider(key="nexp_select").value == 12                           # oberhalb des Größten: das Größte
    assert at.sidebar.selectbox(key="integrand_select").value == C.DEFAULT_INTEGRAND
    assert at.sidebar.selectbox(key="netz_select").value == C.DEFAULT_NETZ
    assert at.sidebar.selectbox(key="weight_select").value == "quadratisch"
    assert at.sidebar.number_input(key="seed_input").value == 0
