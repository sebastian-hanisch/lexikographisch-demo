"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 3, 200, 8
ENUM_MAX = 8                                  # bis hierher werden alle n! Pläne aufgezählt (exakt in ganzen Zahlen)
MILP_MAX = 10                                 # bis hierher rechnet HiGHS (ganzzahlige Optimierung); darüber nur die Ungarische Methode
SEED_MAX, DEFAULT_SEED = 999999, 2
FAMILIES = ("Ganze Zahlen 1 bis 9", "Dezimalzahlen 1,0 bis 9,9", "Dezimalzahlen 0,1 bis 0,9", "Reell gleichverteilt 0 bis 100")
FAMILY_KEYS = ("int", "dec", "tenth", "real")
DEFAULT_FAMILY = 0
INT_SCALE = {"int": 1, "dec": 10, "tenth": 10, "real": 0}   # Faktor, mit dem die Werte ganzzahlig werden (0: nicht exakt darstellbar)

M_EXPONENTS = tuple(range(0, 19))             # Gewicht M = 10^k auf A (M*A + B)
W_EXPONENTS = tuple(range(1, 19))             # Gewicht w = 10^-k auf B (A + w*B)
CHAIN_EXPONENTS = tuple(range(1, 13))         # Kette M^2*A + M*B + C mit M = 10^k
TAUS = (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 5e-2, 1e-1)   # relative Toleranz auf A
TOL = 1e-9                                    # Vergleich reeller Werte: relativ
FLOAT_LIMIT = 2.0 ** 53                       # ab hier unterscheidet float64 nicht mehr jede ganze Zahl

SWEEP_SEEDS = tuple(range(100000, 100020))    # feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_N = 8                                   # Instanzgröße der Sweeps (alle aufgezählt)
SOLVERS = ("Ungarische Methode (float64)", "HiGHS (ganzzahlige Optimierung)")

STATUS_NAMES = {"lex": "gleich dem lexikographischen Optimum", "A": "A verletzt (Rangfolge gekippt)", "B": "B verloren", "C": "C verloren", "none": "keine Lösung"}
STATUS_COLORS = {"lex": "#2F6B65", "A": "#c0504d", "B": "#d8a13a", "C": "#7b5ea7", "none": "#999999"}
COLOR_ALL = "rgba(150,150,150,0.35)"
COLOR_OPT = "#4c78a8"

PRESETS = {
    "Gleichstand, B entscheidet (n = 8)": {"n": 8, "family": 0, "seed": 2},
    "Kleinstes Beispiel (n = 3, ganze Zahlen)": {"n": 3, "family": 0, "seed": 6},
    "Kein Gleichstand (n = 8, reell)": {"n": 8, "family": 3, "seed": 5},
    "Gleichstand geht in float64 verloren (Dezimal)": {"n": 8, "family": 2, "seed": 100014},
    "Toleranz macht den Vorrang zum Kompromiss": {"n": 8, "family": 3, "seed": 100013},
    "Drei Prioritäten (n = 8)": {"n": 8, "family": 0, "seed": 62},
    "Größtes M* (Dezimal 1,0 bis 9,9)": {"n": 8, "family": 1, "seed": 100019},
    "Viele Personen (n = 50)": {"n": 50, "family": 0, "seed": 3},
}
PRESET_HELP = {
    "Gleichstand, B entscheidet (n = 8)": "Kosten haben sechs beste Pläne; erst die Zeit wählt unter ihnen. Mit zu großem M verschwindet die Zeit im Gleitkomma.",
    "Kleinstes Beispiel (n = 3, ganze Zahlen)": "Drei Personen, sechs Pläne: zwei beste Pläne in A, das kleinste M ist 4.",
    "Kein Gleichstand (n = 8, reell)": "Bei reellen Werten hat A genau einen besten Plan: B hat nichts zu entscheiden, lexikographisch ist gleich nur A.",
    "Gleichstand geht in float64 verloren (Dezimal)": "Dezimalzahlen sind in float64 nicht exakt: zwei in A gleiche Pläne gelten nicht mehr als gleich, und B kann falsch entscheiden (hier ja; in den festen Instanzen 1 von 20).",
    "Toleranz macht den Vorrang zum Kompromiss": "Mit einer Toleranz von 1 % auf A gewinnt B rund ein Fünftel bei weniger als 1 % Verlust in A.",
    "Drei Prioritäten (n = 8)": "Auch C entscheidet noch; mit M²·A + M·B + C geht C ab M = 10⁸ verloren.",
    "Größtes M* (Dezimal 1,0 bis 9,9)": "Der Fall mit dem größten kleinsten Gewicht der festen Instanzen: M muss 78 überschreiten.",
    "Viele Personen (n = 50)": "Zu groß zum Aufzählen: die Ungarische Methode zeigt dieselben Fenster; M = 1 kippt die Rangfolge, ab M = 10¹⁶ geht B verloren.",
}
