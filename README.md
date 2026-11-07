# Lexikographisch – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-lexikographisch-demo.streamlit.app/)** -->

Stück 4 der Linie **Zielfunktionsdesign** („Das Ziel ist das Modell“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Arbeitsname; Name, Bereich und Themenfeld werden erst bei der Einbindung in die Website festgelegt). Es baut auf Stück 1 bis 3 auf (`zielvergleich-demo`, `preis-der-fairness-demo`, `zwei-kriterien-demo`): dieselbe Zuordnung, jetzt mit **Prioritäten**. Ein Plan hat bis zu drei Summen: **A** (Kosten), **B** (Zeit), **C** (Wünsche). **Lexikographisch** heißt: unter allen Plänen mit kleinstem A der mit kleinstem B, dann C. Man schreibt das gern als **ein** Ziel `M·A + B` mit großem `M`. Alles ist **simuliert**; keine Aussage über echte Personen, Betriebe oder Lasten.

## Warum dieses Problem

Ein Vorrang und ein großes Gewicht sehen gleich aus und sind es nur in einem **Fenster**. Die Fragen: Wann entscheidet B überhaupt etwas? Wie groß muss `M` mindestens sein? Wo verschwindet B im Gleitkomma oder bei einem Löser? Und was macht eine **Toleranz** („fast lexikographisch“) aus dem Vorrang?

## Modell

Gegeben Belastungen `a[i][j]`, `b[i][j]`, `c[i][j]` (Zeile Person, Spalte Aufgabe). Ein Plan ist eine Permutation `π`; `A(π) = Σ a[i][π(i)]`, ebenso `B(π)` und `C(π)`. **Lexikographisch:** `π` ist besser als `π'`, wenn `A(π) < A(π')`, oder bei gleichem `A` wenn `B(π) < B(π')`, oder bei gleichem `A` und `B` wenn `C(π) < C(π')`. **Gewicht:** `M·A + B` ist wieder eine Zuordnung (Kosten `M a + b` je Paar), gerechnet wie ein Löser: Kostenmatrix bilden, die Einträge in float64 summieren (bei Gleichstand der gerechneten Zahl gilt der erste Plan der Aufzählung). **`M*`** ist das kleinste `M`, bei dem `M·A + B` am lexikographischen Optimum minimal ist: `M* = max(0, max über alle Pläne mit A > A* von (B* − B) / (A − A*))`. Bei ganzen Zahlen gilt `A − A* ≥ 1` und daher `M* ≤ (B-Spanne)`; `M = (B-Spanne) + 1` genügt beweisbar. **Gleitkomma:** float64 hat 53 Bit Mantisse, `2^53 ≈ 9,007·10^15`; ist `M·a` größer, unterscheidet `M·a + b` das `b` nicht mehr. **Toleranz:** erst `A` minimieren, dann `B` unter `A ≤ A*·(1 + τ)`; das ist ein ε-Constraint auf `A` (Stück 3). Formeln stehen im 📐-Expander der App.

## Methodik

Vier Instanzfamilien mit drei unabhängigen Matrizen: **ganze Zahlen 1 bis 9** (exakte Gleichstände), **Dezimalzahlen 1,0 bis 9,9** (Schritt 0,1), **Dezimalzahlen 0,1 bis 0,9** (viele Gleichstände, die in Gleitkomma nicht exakt sind) und **reell gleichverteilt 0 bis 100** (praktisch keine Gleichstände). Ganze Zahlen und Dezimalzahlen sind zusätzlich als ganze Zahlen (Zehntel) gespeichert, damit „gleich“ exakt prüfbar ist. Bis n = 8 werden alle Pläne aufgezählt; darüber findet die Ungarische Methode (`scipy.optimize.linear_sum_assignment`) das exakte lexikographische Optimum mit `M` = (B-Spanne) + 1 auf den ganzzahligen Werten (reelle Werte: nur A). HiGHS (`scipy.optimize.milp`, Voreinstellungen) rechnet bis n = 10 mit. Die App zeigt: das lexikographische Optimum mit der Optimalmenge in A, das Gewicht `M` in 19 Stufen (10⁰ bis 10¹⁸) als Fensterleiste, ein kleines Gewicht `w` auf B (`A + w·B`, 10⁻¹ bis 10⁻¹⁸) mit zwei Lösern, die sequentielle Rechnung mit Toleranz, drei Prioritäten (`M²·A + M·B + C`) und Zählungen über 20 feste Instanzen. Verdikte entstehen live aus der Rechnung, jede Zahl im Text kommt aus dem Ergebnis.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus den Auswertungsfunktionen der App und sind in `tests/test_claims.py` nachgerechnet. „Je 20 feste Instanzen“ heißt `SWEEP_SEEDS` (Seeds 100000 bis 100019), nie ein einzelner Seed; die Zahlen sind **Zählungen und Spannen über Instanzen**, kein Satz und keine Aussage über andere Instanzen. Soweit nicht anders angegeben n = 8 (alle 40 320 Pläne aufgezählt). Familien: ganze Zahlen / Dezimal 1,0 bis 9,9 / reell.

| Frage | Befund |
|---|---|
| Entscheidet B überhaupt? Instanzen mit mindestens zwei besten Plänen in A | **13 / 1 / 0** von 20; Optimalmenge Median **2,5 / 1,0 / 1,0** (Maximum **9 / 2 / 1**); B unterscheidet unter ihnen in **13 / 1 / 0** von 20: **bei reellen Werten ist „lexikographisch“ dasselbe wie „nur A“** |
| Kleinstes Gewicht `M*` (Min / Median / 90. Perzentil / Max) | ganze Zahlen **0,75 / 3,75 / 7,0 / 7,0**; Dezimal **0,97 / 8,11 / 23,5 / 78,0**; reell **0,68 / 5,90 / 29,9 / 96,7**; `M*` über 10 in **0 / 8 / 7** von 20, über 1000 in **0 / 0 / 0**; die Schranke „B-Spanne + 1“ (ganze Zahlen, Median **48,5**) ist beweisbar hinreichend; in der Messung liegt `M*` darunter, in **20** von 20 (Maximum 7,0) |
| Zu kleines Gewicht: Treffer des lexikographischen Optimums bei `M` = 1 / 10 / 100 (von 20) | ganze Zahlen **1 / 20 / 20**, Dezimal **1 / 12 / 20**, reell **2 / 13 / 20**: die einfache Summe (`M` = 1) ist fast nie lexikographisch |
| Zu großes Gewicht in float64, ganze Zahlen: Treffer bei `M` = 10¹⁴ / 10¹⁵ / 10¹⁶ / 10¹⁷ / 10¹⁸ (von 20), gerechnet als float64-Summe über alle Pläne mit numpy | **20 / 20 / 13 / 11 / 11** (mit der Ungarischen Methode auf denselben Matrizen bei 10¹⁵ / 10¹⁶ / 10¹⁷ / 10¹⁸: 20 / 15 / 12 / 10; das Fenster endet in beiden Fällen bei 10¹⁵ bis 10¹⁶); bei Dezimal und reell **20** bei jeder Stufe bis 10¹⁸ (reell: B hat nichts zu entscheiden; Dezimal 1,0 bis 9,9: B entscheidet nur in 1 Instanz, und dort steht der richtige Plan zufällig vorn in der Aufzählung) |
| Ungarische Methode, n = 50, ganze Zahlen, gegen `M` = 401 (von 20) | `M` = 1: **0**; `M` = 10³: **20**; `M` = 10¹⁶: **0**; `M` = 10¹⁸: **0** |
| Kleines Gewicht `A + w·B` mit der Ungarischen Methode, ganze Zahlen (von 20) | bis w = 10⁻¹⁵ **20**, w = 10⁻¹⁶: **16**, w = 10⁻¹⁷ und 10⁻¹⁸: **10** |
| Kleines Gewicht `A + w·B` mit HiGHS (Voreinstellungen), ganze Zahlen (von 20) | w = 10⁻¹ bis 10⁻⁶: **20**, w = 10⁻⁷: **19**, **ab w = 10⁻⁸ bis 10⁻¹⁸: 8**: B wird ignoriert (Ursache nicht isoliert; die MIP-Lückenoptionen allein erklären es nicht) |
| `M·A + B` mit HiGHS, ganze Zahlen (von 20) | `M` = 1: **1**; 10¹ bis 10¹⁵: **20**; 10¹⁶: **17**; 10¹⁷: **10**; 10¹⁸: **12** |
| Drei Prioritäten `M²·A + M·B + C`, ganze Zahlen (von 20) | `M` = 10¹ bis 10⁷: **20**; 10⁸: **19**; 10⁹: **19**; 10¹⁰: **20**; 10¹¹: **18**; 10¹²: **19**; C entscheidet unter den A- und B-Optimalen in **2** von 20 Instanzen |
| Sequentiell mit relativer Toleranz τ = 0 / 10⁻¹² / 10⁻⁹ / 10⁻⁶ / 10⁻³ / 10⁻² / 5·10⁻² / 0,1: gleich dem exakten Lex-Optimum (von 20) | ganze Zahlen **20 / 20 / 20 / 20 / 20 / 20 / 19 / 9**; Dezimal **20 / 20 / 20 / 20 / 20 / 17 / 9 / 5**; reell **20 / 20 / 20 / 20 / 20 / 18 / 11 / 5** |
| Die Toleranz als Kompromiss, reell, τ = 0,1 (Median / Maximum über Instanzen) | Verlust in A **5,4 % / 9,6 %**, Gewinn in B **4,5 % / 41,1 %** |
| Gleichstand in Dezimalzahlen 0,1 bis 0,9 | exakter Gleichstand in A in **13** von 20 (Optimalmenge Median 2,5, Maximum 9); in float64 erkannt Median **2,0** (Maximum **8**); verloren in **2** von 20; die Wahl mit Toleranz 0 weicht ab in **1** von 20, mit relativer Toleranz 10⁻⁹ in **0** |
| Preset „Gleichstand, B entscheidet“ (n = 8, ganze Zahlen, Seed 2) | **6** beste Pläne (A = 17), B reicht von **31** bis **44**, `M*` = **2,00**, Fenster 10¹ bis 10¹⁵ (15 von 19 Stufen), ab 10¹⁶ geht B verloren |
| Preset „Kleinstes Beispiel“ (n = 3, Seed 6) | **2** beste Pläne (A = 13), B **17** gegen **23**, `M*` = **4,00**, Schranke **12** |
| Preset „Gleichstand geht in float64 verloren“ (Dezimal 0,1 bis 0,9, Seed 100014) | **4** exakte Gleichstände, in float64 **2**; die Wahl mit Toleranz 0 liefert B = **2,9** statt **2,7**; HiGHS ignoriert B ab w = 10⁻⁷ |
| Preset „Toleranz macht den Vorrang zum Kompromiss“ (reell, Seed 100013), τ = 0,01 | Verlust in A **0,8 %**, Gewinn in B **20,1 %**, **2** Pläne erlaubt; τ = 0,001 ändert nichts |
| Preset „Drei Prioritäten“ (Seed 62) | **8** beste Pläne in A; die Kette trifft von 10¹ bis 10⁷, ab **10⁸** geht C verloren |
| Preset „Größtes M\*“ (Dezimal 1,0 bis 9,9, Seed 100019) | `M*` = **78,00** |
| Preset „Viele Personen“ (n = 50, ganze Zahlen, Seed 3) | lexikographisch A = **50**, B = **140**; `M` = 1 / 10³ / 10¹⁶ / 10¹⁸: Rangfolge gekippt / richtig / B verloren / B verloren |

**Korrektur der Planungshypothese:** Vor der Messung stand „zu kleines `M` kippt die Rangfolge, zu großes verliert B“. Beides stimmt, aber **zu großes `M` schadet nur dort, wo B gebraucht wird** (exakte Gleichstände, hier ganze Zahlen). Die gemessenen Fenster sind **breit** (bei ganzen Zahlen von 10¹ bis 10¹⁵) und hängen vom Löser und von der Schreibweise ab (`A + w·B` gegen `M·A + B`). Die Gurobi-Dokumentation rät, Gewichte über 10⁶ und unter 10⁻⁶ zu vermeiden; das ist vorsichtiger als die gemessenen Fenster, aber für ein Gewicht unter 10⁻⁷ (HiGHS) berechtigt.

## Ehrliche Grenzen

- **Vier Familien, n = 8 und n = 50:** Wie oft B entscheidet und wie groß `M*` wird, hängt an Familie und Größe; echte Daten können sich anders verhalten.
- **Zwei Löser, eine Version:** Die Fenster gelten für die Ungarische Methode (scipy) und HiGHS über `scipy.optimize.milp` mit den Voreinstellungen in der benutzten Version; andere Löser (Gurobi, CP-SAT) und andere Versionen können andere Fenster haben. Die Zahlen sind Messwerte, keine Eigenschaft des Lösers.
- **Die Aufzählung entscheidet Gleichstände nach der Reihenfolge:** Das aufgezählte Gewichtsoptimum wählt bei Gleichstand der gerechneten Zahl den ersten Plan; ein anderer Löser wählt anders.
- **Reelle Werte:** B hat dort nichts zu entscheiden; das ist ein Befund dieser Familie, bei gerundeten Messwerten gibt es sehr wohl Gleichstände.
- **Die Toleranz ist relativ auf A:** Eine absolute Toleranz oder eine Toleranz je Stufe wählt anders; τ ist eine Entscheidung.
- **Dokumentationsaussagen:** Gurobi (Seite „Multiple Objectives“) und HiGHS (Optionen) sind im Wortlaut gelesen; die Demo übernimmt keine Zahl daraus.

## Verifikation

Insgesamt 393 Tests (`pytest tests/`), alle grün.

- **Kern:** Handbeispiel (n = 3, sechs Pläne) von Hand: Optimalmenge, lexikographisches Optimum, `M*` = 4 mit Gleichstand bei genau 4, Schranke 9, sequentielle Toleranz; lexikographisches Optimum, `M*`, Ungarische Methode, HiGHS und Toleranz gegen reines Python (Tupelvergleich, `fractions.Fraction`, itertools) und `scipy.optimize.linear_sum_assignment`; die Gleitkomma-Wahl gleich der exakten im Fenster; die Gleitkomma-Fakten (`2^53 + 1`, `0,1 + 0,2`).
- **Eigenschaften:** das lexikographische Optimum ist A-optimal, hat unter den besten Plänen das kleinste B und ist Pareto-optimal; `M*` ist die Schwelle (knapp darüber lexikographisch, knapp darunter nicht) und nie über der ganzzahligen Schranke; Skalierung von B skaliert `M*`; Toleranz 0 gleich lexikographisch bei exakten Zahlen, die erlaubte Menge wächst mit τ; das Gewichtsgitter ist ein Fenster (unten A, oben nur bei Bedarf B); Zeilen- und Spaltenpermutation; Determinismus.
- **Randfälle:** n = 1, n = 2, gleiche Matrizen, alle Einträge gleich, Seed 0 und größter Seed, Umschaltgrenzen n = 8 / 9 / 10 / 11, HiGHS bei n = 10, n = 200, reelle Werte ohne Verlust von B, extreme Gewichte.
- **Verdikte:** reine Funktionen, jede Verzweigung und jede Zahl im Text an Hand- oder Presetwerten (`tests/test_verdicts.py`), kein Trendwort in den Zählungen.
- **Presets:** Vollständigkeit, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung, und dass jedes Preset zeigt, was sein Name sagt.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, n = 200), jedes Preset, jede Gewichtsstufe, jeder Löser, jede Toleranz, jedes Nachrechnen, Footer-Test, Layout.
- **Claims:** jede Zahl dieser README in `tests/test_claims.py`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `lxk_algorithm.py` | Kern: Aufzählung, exakte Summen, lexikographisches Optimum, `M*`, Gewichtsrechnung, Ungarische Methode, HiGHS, Toleranz |
| `lxk_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `lxk_evaluation.py` | Gitter, Sweeps, Verdikte |
| `lxk_layout.py` | Fester Kopf (Stylesheet) |
| `lxk_pdf_export.py` | PDF-Export des Ergebnisses |
| `lxk_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `lxk_scenario.py` | Instanzen (drei Matrizen, vier Familien, ganzzahlige Darstellung) |
| `lxk_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Handbeispiel, Gegenproben, Eigenschaften, Randfälle, Verdikte, Presets, Claims, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe, Orakel |

## Bewusst nicht umgesetzt

- **Keine Gleichstände beim Maximum:** Wann das zweite Ziel entscheidet, wenn das erste ein Maximum ist (Optimalmengen, Preis der Fairness je Regel), ist Stück 5.
- **Keine Tabellen für andere Löser:** Gurobi, CP-SAT und andere Löser sind nicht gemessen, nur ihre Dokumentation ist zitiert.
- **Keine absoluten Toleranzen und keine Toleranz je Stufe:** nur die relative Toleranz auf A.
- **Keine Gewichte erheben, kein Swing, keine SMAA:** Wie man Gewichte findet, behandelt die Mehrkriterien-Ranking-Linie.
- **Kein Zweikriterien-Nachbau:** Pareto-Front, Hülle und ε-Constraint sind Stück 3; hier nur der Hinweis, dass die Toleranz ein ε-Constraint ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
