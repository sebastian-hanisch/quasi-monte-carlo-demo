"""Verdikte (reine Funktionen in qmc_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import qmc_evaluation as ev

S = ev.Settings()


def _dims(mean=2.07, add=0.236, sup=3):
    return {"mean_dim": mean, "additive": add, "sup90": sup, "sup99": sup + 2}


def _a(**kw) -> ev.Analysis:
    integ = ev.Integrand("sprung", "Sprung: Gitter 3×3 hängt zusammen (p = 0,90)", 12, 0.947, 0.05, None, _dims())
    ns = (32, 64, 128)
    basis = ev.Analysis(
        integ=integ, N=1024, ns=ns, rmse={"mc": (0.04, 0.03, 0.02), "lhs": (0.03, 0.02, 0.01), "sobol_ds": (0.02, 0.01, 0.006), "sobol_owen": (0.02, 0.01, 0.005)},
        factor={"mc": (1.0, 1.0, 1.0), "lhs": (0.8, 0.7, 0.6), "sobol_ds": (0.3, 0.2, 0.1), "sobol_owen": (0.3, 0.2, 0.1)}, slope={"mc": -0.5, "lhs": -0.52, "sobol_ds": -0.8, "sobol_owen": -0.84},
        factor_owen=0.217, factor_ds=0.25, factor_lhs=0.764, lhs_pred=0.764, ratio=2.1, reps=100)
    return dataclasses.replace(basis, **kw)


def test_gewinn_mit_zahlen_und_dimension():
    kind, text = ev.verdict_main(_a(), S)
    assert kind == "success"
    for frag in ("Sprung: Gitter 3×3 hängt zusammen (p = 0,90), 12 Dimensionen, N = 1.024 Punkte", "den Faktor 0,217", "2,1-mal kleiner", "digitale Verschiebung: Faktor 0,250", "Latin Hypercube: Faktor 0,764",
                 "Die Sobol-Folge gewinnt deutlich.", "mittlere Dimension 2,07", "bis Ordnung 3", "additiver Anteil 24 %"):
        assert frag in text, frag


def test_um_groessenordnungen_unterhalb_der_schwelle():
    kind, text = ev.verdict_main(_a(factor_owen=0.004, ratio=15.8), S)
    assert kind == "success" and "um Größenordnungen" in text and "den Faktor 0,0040" in text


def test_kein_gewinn_im_rauschen_und_verlust():
    kind, text = ev.verdict_main(_a(factor_owen=1.1, ratio=0.95), S)
    assert kind == "info" and "Höchstens kleiner Gewinn" in text and "±15 %" in text
    kind, text = ev.verdict_main(_a(factor_owen=1.5, ratio=0.8), S)
    assert kind == "warning" and "schlechter als der Zufall" in text


def test_schwellen_liegen_bei_einhalb_und_eins_komma_drei():
    assert ev.verdict_main(_a(factor_owen=0.499), S)[0] == "success" and ev.verdict_main(_a(factor_owen=0.5), S)[0] == "info"
    assert ev.verdict_main(_a(factor_owen=1.3), S)[0] == "info" and ev.verdict_main(_a(factor_owen=1.31), S)[0] == "warning"


def test_engpass_ohne_anova_wird_benannt():
    a = _a(integ=dataclasses.replace(_a().integ, dims=None, key="engpass"))
    assert "keine exakte ANOVA berechnet" in ev.verdict_main(a, S)[1]


def test_konstanter_integrand():
    a = _a(integ=dataclasses.replace(_a().integ, var=0.0))
    kind, text = ev.verdict_main(a, S)
    assert kind == "info" and "konstant" in text


def test_kurven_verdikt_nennt_steigungen_und_die_vorhersage_fuer_latin_hypercube():
    kind, text = ev.verdict_curve(_a())
    assert kind == "info" and "Zufall −0,50" in text and "Sobol mit Owen −0,84" in text and "vorhergesagter Faktor 0,764, gemessen 0,764" in text
    kind, text = ev.verdict_curve(_a(slope={"mc": float("nan"), "lhs": float("nan"), "sobol_ds": float("nan"), "sobol_owen": float("nan")}))
    assert "mindestens drei Punktzahlen" in text
    assert "vorhergesagter Faktor" not in ev.verdict_curve(_a(integ=dataclasses.replace(_a().integ, dims=None)))[1]


def test_dimensions_verdikt_vergleicht_die_beiden_leitfaelle():
    rows = [{"kind": "exp", "d": 64, "weight": "harmonisch", "dims": _dims(1.04, 0.96, 1), "factor_owen": 0.0002, "factor_lhs": 0.04, "lhs_pred": 0.04, "label": "a"},
            {"kind": "exp", "d": 32, "weight": "gleich", "dims": _dims(2.64, 0.23, 4), "factor_owen": 0.423, "factor_lhs": 0.7, "lhs_pred": 0.77, "label": "b"},
            {"kind": "netz", "d": 12, "dims": _dims(2.07, 0.236, 3), "factor_owen": 0.217, "factor_lhs": 0.8, "lhs_pred": 0.76, "label": "Sprung p = 0,9"}]
    kind, text = ev.verdict_dimension({"rows": rows, "N": 1024, "netz": "Gitter 3×3"}, S)
    assert kind == "info"
    for frag in ("Bei N = 1.024", "(mittlere Dimension 1,04) haben den Faktor 0,0002", "(mittlere Dimension 2,64) nur 0,423", "Gitter 3×3: Sprung p = 0,9: mittlere Dimension 2,07, Faktor 0,217"):
        assert frag in text, frag


def test_leiter_verdikt():
    rows = [{"key": k, "label": k, "d": 12, "dims": None, "factors": {"mc": 1.0, "lhs": l, "sobol_ds": f, "sobol_owen": f}} for k, f, l in (("sprung", 0.3, 0.5), ("engpass", 0.1, 0.4), ("glatt", 0.00003, 0.016))]
    kind, text = ev.verdict_ladder({"rows": rows, "N": 1024, "netz": "Gitter 3×3"})
    assert kind == "info" and "Sprung 0,300, Engpass (stetig mit Knicken) 0,100, glatt unter 0,0001" in text and "Latin Hypercube holt bei dem glatten Integranden 0,016" in text
    assert "Rangfolge vom größten Gewinn an: glatt vor Engpass vor Sprung" in text
    rows[2]["factors"]["sobol_owen"] = 0.9
    assert ev.verdict_ladder({"rows": rows, "N": 1024, "netz": "x"})[0] == "warning"


def test_intervall_verdikt():
    rows = [{"name": "mc", "label": "Zufall", "coverage": 0.947, "rmse": 1.0, "hw_ratio": 1.97, "vs_single": 5.0, "vs_mc": 1.0},
            {"name": "naiv", "label": "naiv", "coverage": 1.0, "rmse": 0.2, "hw_ratio": 31.5, "vs_single": 1.0, "vs_mc": 0.2},
            {"name": 8, "label": "K = 8", "coverage": 0.933, "rmse": 0.5, "hw_ratio": 2.2, "vs_single": 2.5, "vs_mc": 0.5}]
    kind, text = ev.verdict_intervals({"rows": rows, "N": 1024, "label": "Begleiter", "reps": 300})
    assert kind == "info"
    for frag in ("300 Wiederholungen", "Zufall 0,947 Überdeckung", "das 31,5-Fache des wahren RMSE und die Überdeckung 1,000", "K = 8: Überdeckung 0,933, RMSE das 2,5-Fache einer einzigen Menge", "kostet Genauigkeit"):
        assert frag in text, frag


def test_punktzahl_verdikt():
    sizes = [{"name": "Zweierpotenz", "N": 1024, "factor_owen": 0.0041, "factor_ds": 0.004}, {"name": "24 Punkte weniger", "N": 1000, "factor_owen": 0.0080, "factor_ds": 0.007},
             {"name": "das 1,5-Fache", "N": 1536, "factor_owen": 0.0043, "factor_ds": 0.004}]
    order = [{"name": "zuerst", "factor_owen": 0.00014, "factor_ds": 0.00125}, {"name": "zuletzt", "factor_owen": 0.00106, "factor_ds": 0.00178}]
    kind, text = ev.verdict_sizes({"sizes": sizes, "order": order, "N": 1024})
    assert kind == "info"
    for frag in ("bei N = 1.024 0,0041, bei N = 1.000 0,0080", "also das 1,95-Fache", "bei dem 1,5-Fachen 0,0043", "wichtige zuerst 0,0001, zuletzt 0,0011 (Owen)", "0,0013 gegen 0,0018 (digitale Verschiebung)"):
        assert frag in text, frag
