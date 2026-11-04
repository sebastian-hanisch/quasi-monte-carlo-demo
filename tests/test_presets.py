"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import qmc_constants as C
import qmc_evaluation as ev
import qmc_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["integrand"] in C.INTEGRANDS and preset["netz"] in C.NETZE and preset["weight"] in C.WEIGHTS, name
        assert preset["p"] in C.P_CHOICES and preset["d"] in C.D_CHOICES and preset["nexp"] in C.NEXP_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    standard = C.PRESETS["Standardfall (Voreinstellung)"]
    for name, preset in C.PRESETS.items():
        if name != "Standardfall (Voreinstellung)":
            assert preset != standard, name


def test_t_quantile_gehoeren_zu_den_wahlbaren_k():
    assert set(C.T975) == set(C.K_CHOICES)


@pytest.mark.parametrize("name,kind,hoechstens", [("Standardfall (Voreinstellung)", "success", 0.001), ("Sprung am Gitter", "success", 0.5), ("Hantel: eine Brücke", "success", 0.2),
                                                   ("Hohe Dimension, abfallende Gewichte", "success", 0.001)])
def test_preset_zeigt_den_gewinn(name, kind, hoechstens):
    s = ev.Settings(**C.PRESETS[name])
    a = ev.analyse(s)
    assert ev.verdict_main(a, s)[0] == kind and a.factor_owen < hoechstens


def test_preset_gleiche_gewichte_in_hoher_dimension_gewinnt_nur_wenig():
    s = ev.Settings(**C.PRESETS["Hohe Dimension, gleiche Gewichte"])
    a = ev.analyse(s)
    assert 0.1 < a.factor_owen < 0.9 and a.integ.dims["mean_dim"] > 2.5


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(ev.Settings(**C.PRESETS[name]))
    assert a.integ.var > 0 and len(a.ns) == C.PRESETS[name]["nexp"] - C.MIN_CURVE_EXP + 1
