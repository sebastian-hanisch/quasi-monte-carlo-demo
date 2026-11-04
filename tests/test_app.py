"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Regler an seine Grenzen (oder durch jede Auswahl) und prüft,
dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException` bei Reglern mit berechneten Grenzen."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import qmc_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def _werte(key):
    spec = P.SETTING_SPECS[key]
    return list(spec.options) if spec.options else [spec.lo, spec.hi]


_VORHER = {"p_select": "sprung", "d_select": "exp", "weight_select": "exp"}


def _regler(at, key):
    for art in ("select_slider", "selectbox", "number_input"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


@pytest.mark.parametrize("key", [k for k in P.SETTING_SPECS if k not in ("nexp_select", "seed_input")])
def test_every_option_of_every_setting_does_not_raise(key):
    for value in _werte(key):
        at = _fresh_app()
        if key in _VORHER:                                     # gesperrte Regler lassen sich erst nach dem passenden Integranden setzen
            at.sidebar.selectbox(key="integrand_select").set_value(_VORHER[key]).run(timeout=TIMEOUT)
        _regler(at, key).set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (key, value, [str(e) for e in at.exception])


@pytest.mark.parametrize("value", [6, 12])
def test_kleinste_und_groesste_punktzahl(value):
    at = _fresh_app()
    at.sidebar.select_slider(key="nexp_select").set_value(value).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_seed_grenzen():
    for value in _werte("seed_input"):
        at = _fresh_app()
        at.sidebar.number_input(key="seed_input").set_value(value).run(timeout=TIMEOUT)
        assert not at.exception
