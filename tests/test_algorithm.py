"""Kern von Hand nachgerechnet (DEMO-PLAYBOOK §1b): jede Zahl hier steht auch auf Papier."""

import numpy as np
import pytest

import qmc_algorithm as A
import qmc_constants as C


def test_zufallsstrom_gleicht_der_ganzzahl_referenz():
    seed = A.stream_seed(7, 3)
    u = A.uniforms(seed, 5)
    ref = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 6)]
    assert u.tolist() == ref
    assert A.stream_seed(7, 3) != A.stream_seed(3, 7) and ((u >= 0) & (u < 1)).all()


def test_richtungszahlen_der_zweiten_koordinate_von_hand():
    """Polynom x + 1 (Bitmuster 3, Grad 1), m_1 = 1: m_k = 2 m_{k-1} xor m_{k-1}: 1, 3, 5, 15."""
    V = A.direction_ints(2)
    m = [int(V[k, 1]) >> (A.BITS - 1 - k) for k in range(4)]
    assert m == [1, 3, 5, 15]
    assert [int(V[k, 0]) for k in range(3)] == [1 << 31, 1 << 30, 1 << 29]     # van der Corput: v_k = 2^-k


def test_die_ersten_vier_sobol_punkte_in_zwei_dimensionen_von_hand():
    x = A.sobol_ints(4, 2).astype(np.float64) / 2**32
    assert x.tolist() == [[0.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]


def test_netz_eigenschaft_der_vier_punkte_von_hand():
    """Jedes der vier Rechtecke ½ × ½, jeder Streifen ¼ × 1 und 1 × ¼ enthält genau einen Punkt (t = 0)."""
    x = A.sobol_ints(4, 2).astype(np.float64) / 2**32
    for a1, a2 in ((2, 0), (1, 1), (0, 2)):
        cells = {(int(px * 2**a1), int(py * 2**a2)) for px, py in x}
        assert len(cells) == 4


def test_ungerade_punktzahlen_sind_die_ersten_punkte_der_folge():
    assert (A.sobol_ints(3, 2) == A.sobol_ints(8, 2)[:3]).all()


def test_schaetzung_ohne_randomisierung_von_hand():
    """∫ u1 u2 = ¼; die vier Punkte liefern (0 + ¼ + 3/16 + 3/16)/4 = 5/32 (verzerrt, weil der Nullpunkt dabei ist)."""
    x = A.sobol_ints(4, 2).astype(np.float64) / 2**32
    assert (x[:, 0] * x[:, 1]).mean() == pytest.approx(5 / 32)


def test_digitale_verschiebung_ist_ein_xor():
    x = A.sobol_ints(8, 3)
    s = A.digital_shift(x, 99)
    r = x[0] ^ s[0]                                    # der erste Punkt ist 0: x ^ r = r
    assert ((x ^ r[None, :]) == s).all() and (A.digital_shift(s, 99) == x).all()


def test_owen_scrambling_von_hand_bei_zwei_punkten():
    """Zwei Punkte der van-der-Corput-Folge (0 und ½): die Tiefe 1 kippt das oberste Bit beider Punkte gleich (Präfix leer), die Menge bleibt ein Netz: ein Punkt je Hälfte."""
    x = A.sobol_ints(2, 1)
    out = A.owen_scramble(x, 5, 1)
    assert sorted(out[:, 0].tolist()) == [0, 1] and out.max() < 2


def test_owen_erhaelt_die_netz_eigenschaft_in_vier_punkten():
    for seed in range(20):
        out = A.owen_scramble(A.sobol_ints(4, 2), seed, 2)
        assert out.max() < 4
        for a1, a2 in ((2, 0), (1, 1), (0, 2)):
            cells = {(int(px) >> (2 - a1), int(py) >> (2 - a2)) for px, py in out}
            assert len(cells) == 4


def test_tiefe_des_scramblings():
    assert [A.scramble_depth(N) for N in (1, 2, 3, 4, 5, 1024, 1025)] == [1, 1, 2, 2, 3, 10, 11]


def test_punkte_liegen_in_null_bis_eins_und_haben_die_richtige_form():
    for method in A.METHODS + ("sobol_roh",):
        u = A.points(method, 16, 5, 11)
        assert u.shape == (16, 5) and (u >= 0).all() and (u < 1).all()


def test_latin_hypercube_hat_eine_schicht_je_zeile():
    u = A.points("lhs", 32, 4, 5)
    for j in range(4):
        assert sorted((u[:, j] * 32).astype(int).tolist()) == list(range(32))


def test_zusammenhangstabelle_und_zaehlung_am_dreieck(tab_dreieck):
    assert A.conn_counts(tab_dreieck, 3) == [0, 0, 3, 1]
    assert tab_dreieck.tolist() == [False, False, False, True, False, True, True, True]


def test_sprung_am_dreieck_von_hand(tab_dreieck):
    u = np.array([[0.1, 0.2, 0.9], [0.6, 0.7, 0.9], [0.1, 0.9, 0.2]])
    assert A.f_sprung(u, 0.5, tab_dreieck).tolist() == [1.0, 0.0, 1.0]            # zwei haltende Kanten genügen


def test_engpass_am_dreieck_ist_die_zweitgroesste(tab_dreieck):
    u = np.array([[0.3, 0.9, 0.5], [0.2, 0.1, 0.8], [0.7, 0.7, 0.2]])
    assert A.f_engpass(u, tab_dreieck).tolist() == [0.5, 0.2, 0.7]


def test_engpass_wahrheit_am_dreieck_von_hand(tab_dreieck):
    """E B = (3 · 2! 1! + 1 · 3! 0!)/4! = ½, E B² = 2 (3 · 2! 2! + 1 · 3! 1!)/5! = 0,3, Var = 0,05 (zweite Ordnungsstatistik von drei Gleichverteilten: Beta(2, 2))."""
    Nk = A.conn_counts(tab_dreieck, 3)
    assert A.engpass_exact(Nk) == pytest.approx(0.5) and A.engpass_var_exact(Nk) == pytest.approx(0.05)


def test_zuverlaessigkeit_am_dreieck_von_hand(tab_dreieck):
    Nk = A.conn_counts(tab_dreieck, 3)
    assert A.r_exact(Nk, 0.7) == pytest.approx(3 * 0.49 * 0.3 + 0.343)


def test_glatter_integrand_in_der_mitte_ist_die_zuverlaessigkeit(tab_dreieck):
    """u = ½ ⇒ p_e = 0,7 für alle Kanten ⇒ R(0,7) = 0,784."""
    assert A.f_glatt(np.full((2, 3), 0.5), tab_dreieck).tolist() == pytest.approx([0.784, 0.784])


def test_anova_des_pfads_von_hand(tab_pfad):
    """Pfad, p = ½: X = x1 x2. c_leer = ¼, c_1 = c_2 = c_12 = ¼, Varianz 3/16; Ordnungsanteile 2/3 und 1/3, mittlere Dimension 4/3."""
    c = A.anova_coeffs(tab_pfad, 2, 0.5)
    assert c.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    sh = A.order_shares(c**2, 2)
    assert sh.tolist() == pytest.approx([0.0, 2 / 3, 1 / 3])
    d = A.dim_summary(sh)
    assert d["additive"] == pytest.approx(2 / 3) and d["mean_dim"] == pytest.approx(4 / 3) and d["sup90"] == 2 and d["sup99"] == 2


def test_begleiter_von_hand():
    """exp(u) in einer Dimension: Wert e − 1, Varianz (e² − 1)/2 − (e − 1)²; in einer Dimension liegt alles in Ordnung 1."""
    a = np.array([1.0])
    assert A.exp_exact(a) == pytest.approx(np.e - 1)
    assert A.exp_var_exact(a) == pytest.approx((np.e**2 - 1) / 2 - (np.e - 1) ** 2)
    assert A.exp_order_shares(a).tolist() == pytest.approx([0.0, 1.0])


def test_ordnungsanteile_des_begleiters_in_zwei_dimensionen():
    """r = Var g/(E g)² je Koordinate; Anteile: Ordnung 1 = 2 r (1 + r)... normiert auf 2 r + r²: (2 r)/(2 r + r²) und r²/(2 r + r²)."""
    a = np.ones(2)
    r = ((np.e**2 - 1) / 2 - (np.e - 1) ** 2) / (np.e - 1) ** 2
    sh = A.exp_order_shares(a)
    assert sh.tolist() == pytest.approx([0.0, 2 * r / (2 * r + r * r), r * r / (2 * r + r * r)])


def test_gewichte():
    assert A.exp_weights("gleich", 3).tolist() == [1.0, 1.0, 1.0]
    assert A.exp_weights("quadratisch", 3).tolist() == pytest.approx([2.0, 0.5, 2 / 9])
    assert A.exp_weights("harmonisch", 3).tolist() == pytest.approx([1.0, 0.5, 1 / 3])


def test_blockintervall_von_hand():
    """Blockmittel 1, 3, 5, 7: Mittel 4, Standardabweichung √(20/3), Halbbreite 3,182446 · √(20/3)/2."""
    est, hw = A.block_interval(np.array([1.0, 3.0, 5.0, 7.0]))
    assert est == pytest.approx(4.0) and hw == pytest.approx(3.182446 * np.sqrt(20 / 3) / 2)


def test_naives_intervall_von_hand():
    est, hw = A.naive_interval(np.array([1.0, 2.0, 3.0, 4.0]))
    assert est == pytest.approx(2.5) and hw == pytest.approx(C.Z95 * np.std([1, 2, 3, 4], ddof=1) / 2)


def test_t_quantile_der_tabelle():
    from scipy.stats import t
    for K, q in C.T975.items():
        assert q == pytest.approx(t.ppf(0.975, K - 1), abs=1e-6)
