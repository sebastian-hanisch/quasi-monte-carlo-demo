"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

NETZE = {"hantel": ("hantel", 0, 0), "3x3": ("gitter", 3, 3), "3x4": ("gitter", 3, 4)}
DEFAULT_NETZ = "3x3"
LIVE_EDGE_LIMIT = 17          # Aufzählung aller Kantenmengen live bis hierher (2^17 = 131 072)
CUSTOM_EDGE_LIMIT = 17
CUSTOM_NODE_LIMIT = 14

INTEGRANDS = {"sprung": "Sprung: das Netz hängt zusammen", "engpass": "Engpass: stetig mit Knicken", "glatt": "glatt: unsichere Haltewahrscheinlichkeiten", "exp": "Begleiter: exp(Σ aᵢuᵢ)"}
DEFAULT_INTEGRAND = "glatt"

P_CHOICES = (0.7, 0.9, 0.99)                                 # Haltewahrscheinlichkeit je Kante (nur Sprung)
P_MIN, P_MAX, DEFAULT_P = P_CHOICES[0], P_CHOICES[-1], 0.9
P_SMOOTH, DELTA_SMOOTH = 0.7, 0.4                            # glatter Integrand: p_e = P_SMOOTH + DELTA_SMOOTH (u_e - 1/2)
D_CHOICES = (2, 4, 8, 16, 32, 64)                            # Dimension des Begleiters
D_MIN, D_MAX, DEFAULT_D = D_CHOICES[0], D_CHOICES[-1], 16
WEIGHTS = {"gleich": "aᵢ = 1", "quadratisch": "aᵢ = 2/i²", "harmonisch": "aᵢ = 1/i"}
DEFAULT_WEIGHT = "gleich"
NEXP_CHOICES = (6, 7, 8, 9, 10, 11, 12)                      # N = 2^k Punkte
NEXP_MIN, NEXP_MAX, DEFAULT_NEXP = NEXP_CHOICES[0], NEXP_CHOICES[-1], 10
K_CHOICES = (4, 8, 16, 32)                                   # Zahlen unabhängiger Randomisierungen (Blöcke) für das Intervall in Experiment 4
T975 = {4: 3.182446, 8: 2.364624, 16: 2.131450, 32: 2.039513}   # t-Quantil 0,975 mit K-1 Freiheitsgraden
SEED_MAX, DEFAULT_SEED = 999999, 35

REPS = 100                                # Wiederholungen je Messung der Fehlerkurve (feste Ströme)
GRID_REPS = 60                            # Wiederholungen je Punkt der Dimensions- und Leiter-Messungen
CI_REPS = 300                             # Wiederholungen der Intervall-Messung
MIN_CURVE_EXP = 5                         # kleinstes N der Fehlerkurve: 2^5
BITS = 32                                 # Stellen der Sobol-Ganzzahlen
MAX_DIM = 64                              # eingebettete Richtungszahlen reichen bis hierher

Z95 = 1.959963984540054

COLOR_MC = "#8a8a8a"
COLOR_LHS = "#4c78a8"
COLOR_DS = "#E08E0B"
COLOR_OWEN = "#2F6B65"
COLOR_BAD = "#C0392B"
COLOR_TRUTH = "#222222"

METHOD_LABELS = {"mc": "Monte Carlo", "lhs": "Latin Hypercube", "sobol_ds": "Sobol + Verschiebung", "sobol_owen": "Sobol + Owen"}
METHOD_COLORS = {"mc": COLOR_MC, "lhs": COLOR_LHS, "sobol_ds": COLOR_DS, "sobol_owen": COLOR_OWEN}

PRESETS = {
    "Standardfall (Voreinstellung)": {"integrand": DEFAULT_INTEGRAND, "netz": DEFAULT_NETZ, "p": DEFAULT_P, "d": DEFAULT_D, "weight": DEFAULT_WEIGHT, "nexp": DEFAULT_NEXP, "seed": DEFAULT_SEED},
    "Sprung am Gitter": {"integrand": "sprung", "netz": "3x4", "p": 0.9, "d": DEFAULT_D, "weight": DEFAULT_WEIGHT, "nexp": DEFAULT_NEXP, "seed": DEFAULT_SEED},
    "Hantel: eine Brücke": {"integrand": "engpass", "netz": "hantel", "p": 0.9, "d": DEFAULT_D, "weight": DEFAULT_WEIGHT, "nexp": DEFAULT_NEXP, "seed": DEFAULT_SEED},
    "Hohe Dimension, abfallende Gewichte": {"integrand": "exp", "netz": DEFAULT_NETZ, "p": DEFAULT_P, "d": 64, "weight": "harmonisch", "nexp": DEFAULT_NEXP, "seed": DEFAULT_SEED},
    "Hohe Dimension, gleiche Gewichte": {"integrand": "exp", "netz": DEFAULT_NETZ, "p": DEFAULT_P, "d": 32, "weight": "gleich", "nexp": DEFAULT_NEXP, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×3, jede Kante hat eine unsichere Haltewahrscheinlichkeit um 0,7: der Integrand ist ein Polynom, 1024 Punkte. Die gescramblete Sobol-Folge ist um Größenordnungen genauer als Zufall.",
    "Sprung am Gitter": "Gitter 3×4, Ereignis „das Netz hängt zusammen“ bei p = 0,9: ein Sprung in 17 Dimensionen, der Gewinn ist mäßig.",
    "Hantel: eine Brücke": "Hantel mit stetigem, aber geknicktem Engpasswert: wenige wichtige Eingaben, der Gewinn liegt in der Mitte.",
    "Hohe Dimension, abfallende Gewichte": "64 Dimensionen, aber die späteren Koordinaten zählen kaum: die effektive Dimension ist klein und der Gewinn riesig.",
    "Hohe Dimension, gleiche Gewichte": "32 Dimensionen mit gleichen Gewichten: die effektive Dimension steigt, der Gewinn schrumpft auf einen kleinen Faktor.",
}
