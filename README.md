# Randomisiertes Quasi-Monte-Carlo: wann gewinnt die Sobol-Folge? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-quasi-monte-carlo-demo.streamlit.app/)** -->

Neuntes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [latin-hypercube-demo](https://github.com/sebastian-hanisch/latin-hypercube-demo) (Stück 5, gleichmäßig abdecken je Eingabe): Hier füllt eine **Sobol-Folge** den Würfel auch in den Wechselwirkungen niedriger Ordnung gleichmäßig, mit **zufälligem Scrambling**, damit der Schätzer erwartungstreu bleibt und Fehlerbalken möglich sind. Vehikel sind die Netz-Integranden der Reihe (Zusammenhang eines Gitternetzes oder einer Hantel), nach Glattheit in drei Stufen, dazu ein Begleiter exp(Σ aᵢuᵢ) mit Dimensionsregler bis 64; Wahrheit, Varianz und effektive Dimension sind exakt.

## Warum dieses Problem

Quasi-Monte-Carlo gilt als Verbesserung des Zufalls: gleichmäßigere Punkte, schnellerer Fehlerabfall. Wann stimmt das, und was kostet es? Die Kernfrage: **Wann schlägt eine gescramblete Sobol-Folge die Zufallsziehung, und warum nicht immer?** Dazu die zwei Fallen, die in der Praxis stören: Der naive Fehlerbalken stimmt nicht, und die Punktzahl muss eine Zweierpotenz sein.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat oder beim Bau dazukam:

- **„In hoher Dimension kann es schlechter sein als Zufall“ ist nicht reproduziert.** Der Gewinn verschwindet mit wachsender Dimension (Parallel, m = 64: Faktor um 1), schlägt aber im Rahmen des Rauschens nicht in Verlust um. Für Integranden mit Randsingularitäten ist ein Verlust aus der Literatur bekannt; hier nicht geprüft.
- **Die nominale Dimension allein sagt wenig, die effektive Dimension und die Glattheit bestimmen den Gewinn.** 64 Dimensionen mit abfallenden Gewichten gewinnen mehr als 16 mit gleichen.
- **Die Glattheitsleiter ist keine strenge Rangfolge.** Glatt gewinnt in allen drei Netzen am meisten, aber Sprung und Knick liegen nah beieinander: beim Gitter 3×4 und N = 1024 ist der Engpass nicht besser als der Sprung (0,29 gegen 0,23).
- **Der Regler K (Zahl der Randomisierungen) entfiel.** Experiment 4 rechnet alle vier Werte K = 4, 8, 16, 32 zugleich; ein Regler hätte nur eine Zeile hervorgehoben.
- **Der glatte Integrand reicht bis 17 Kanten**, nicht nur bis 12: die Zustandssumme wird als Matrixprodukt aus zwei Hälften gerechnet (Kosten wie N · 2^(m/2) · 2^(m/2) in schneller Matrixrechnung).
- **Owen-Scrambling nur bis Tiefe ⌈log₂ N⌉:** ab dort sind die Präfixe der Punkte einer Koordinate paarweise verschieden, die tieferen Kippbits unabhängig gleichverteilt; sie werden durch Zufallszahlen ersetzt (gleiche Verteilung, weniger Rechnung).

## Modell

- **Integrale:** I = ∫ f(u) du über den Würfel [0,1]^d. Der Schätzer ist das Mittel über N Punkte; Kennzahl ist der **Faktor** = mittlerer quadratischer Fehler geteilt durch die exakte Monte-Carlo-Varianz σ²/N (unter 1: Gewinn). Das **RMSE-Verhältnis** Zufall : Owen ist 1/√Faktor.
- **Integranden am Netz** (jede Kante ↔ eine Koordinate): *Sprung* X(u) = 1, wenn das Netz mit den Kanten {u_e < p} zusammenhängt; *Engpass* B(u) = größtes t, bei dem es mit {u_e ≥ t} zusammenhängt (stetig, stückweise linear); *glatt* f(u) = R(p₁(u₁), …, p_m(u_m)) mit p_e = 0,7 + 0,4 (u_e − ½), die Zuverlässigkeit bei unsicheren Haltewahrscheinlichkeiten, ein Polynom in u mit E f = R(0,7).
- **Begleiter:** exp(Σ aᵢuᵢ), Gewichte aᵢ = 1, 2/i² oder 1/i; Dimension 2 bis 64.
- **Sobol-Folge:** Joe-Kuo-Richtungszahlen (64 Dimensionen eingebettet), Gray-Code, 32 Bit. Randomisierung: zufällige digitale Verschiebung (XOR) oder verschachteltes Owen-Scrambling (Hash statt gespeichertem Baum).
- **Effektive Dimension (exakt):** ANOVA nach Ordnung der Wechselwirkung; mittlere Dimension Σ|S|σ²_S/σ², Überlagerungsdimension (kleinste Ordnung mit 90 % der Varianz), additiver Anteil (Ordnung 1). Mathematik im 📐-Expander der App.

## Methodik

Zufallszahlen aus festen SplitMix64-Strömen (nie ein einzelner Seed). Wahrheit, Varianz und ANOVA exakt (Aufzählung aller 2^m Kantenmengen bzw. geschlossene Formeln). Fehlerkurven mit 100 Wiederholungen je Punktzahl (N = 32 bis 2^k), Dimensions-, Leiter- und Punktzahl-Messungen mit 60 bis 120, Intervalle mit 300.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. „Faktor“ = mittlerer quadratischer Fehler / (σ²/N) der Owen-Sobol-Folge, N = 1024, wenn nicht anders angegeben; „RMSE-Verhältnis“ = Zufall : Owen.

| Frage | Befund |
|---|---|
| Begleiter, gleiche Gewichte, Dimension 2 / 4 / 8 / 16 / 32 | Faktor **unter 0,0001 / unter 0,0001 / 0,0036 / 0,087 / 0,45**; RMSE-Verhältnis **633 / 153 / 17 / 3,4 / 1,3** |
| Begleiter, abfallende Gewichte | aᵢ = 2/i², d = 16: Faktor **unter 0,0001** (RMSE-Verhältnis 412); aᵢ = 1/i, **d = 64**: Faktor **0,0002** (78): mehr als 16 Dimensionen mit gleichen Gewichten (0,087) |
| Exakte effektive Dimension des Begleiters (mittlere Dimension; additiver Anteil) | gleiche Gewichte d = 4 / 8 / 16 / 32 / 64: **1,12 / 1,30 / 1,69 / 2,64 / 4,88** (0,885 / 0,747 / 0,519 / 0,229 / 0,034); 2/i², d = 16: **1,03** (0,975); 1/i, d = 64: **1,04** (0,961) |
| Steigung von log RMSE über log N (Owen / Verschiebung), N = 32 … 1024 | d = 2: **−1,47 / −1,00**; d = 4: −1,27 / −1,05; d = 8: −1,04 / −1,02; d = 16: **−0,68 / −0,68**; 2/i² (d = 16): −1,37 / −1,00; 1/i (d = 64): −1,17 / −1,03; Zufall und Latin Hypercube **−0,5** |
| Latin Hypercube (Stück 5) am Begleiter | Faktor ≈ 1 − additiver Anteil, so für d = 4 und 8 bei N = 1024 gemessen: d = 4 **0,114** gegen vorhergesagt 0,115; d = 8: 0,225 gegen 0,253 (bei d ≥ 16 weicht die Messung ab) |
| Gleiche Gewichte in 64 Dimensionen | mittlere Dimension 4,88; Faktor der Owen-Folge zwischen 0,3 und 0,5 (kleiner Gewinn, RMSE-Verhältnis etwa 1,4 bis 1,8; die Messung rauscht wegen der schweren Ränder von exp stark) |
| Exakte ANOVA des Sprungs, p = 0,7 / 0,9 / 0,99 (mittlere Dimension; 90 % bis Ordnung) | Hantel **1,71 / 1,34 / 1,06** (3 / 2 / 1); 3×3 2,23 / 2,07 / 2,02 (5 / 3 / 2); 3×4 **2,48 / 2,12 / 2,02** (5 / 3 / 2); additiver Anteil Hantel 0,597 / 0,736 / 0,946, 3×4 0,467 / 0,214 / 0,020 (wie Stück 5) |
| Exakte ANOVA des glatten Integranden | mittlere Dimension **1,02**, additiver Anteil über 0,97 und 90 % der Varianz schon in Ordnung 1, bei allen drei Netzen |
| **Glattheitsleiter** am selben Netz, Faktor Sprung / Engpass / glatt, N = 1024 | Hantel **0,062 / 0,020 / unter 0,0001**; Gitter 3×3 **0,30 / 0,14 / unter 0,0001**; Gitter 3×4 0,23 / 0,29 / 0,005 (Engpass nicht besser als Sprung) |
| Netz-Integranden bei N = 4096, Faktor, Sprung (p = 0,9) / Engpass | Hantel **0,040 / 0,019**; 3×3 **0,22 / 0,055**; 3×4 **0,21 / 0,14**; die Steigungen liegen für Sprung und Engpass zwischen −0,5 und −0,75, für das Polynom bei Hantel und 3×3 unter −1 (3×4: −0,8) |
| Seltene Ausfälle (Sprung p = 0,99), N = 4096 | Faktor 3×3 **0,73**, 3×4 **0,67**: der Gewinn bleibt klein (Stück 5: erst ab N·q ≈ 1 überhaupt Gewinn) |
| **Fehlerbalken**, Begleiter exp d = 8, N = 1024 (300 Wiederholungen) | Zufall Überdeckung **0,95**; Owen mit naivem Standardfehler: Überdeckung **1,000**, Halbbreite **34-mal** der wahre RMSE; K = 4 / 8 / 16 / 32 Blöcke: Überdeckung **0,953 / 0,957 / 0,960 / 0,937**, RMSE das **2,0 / 2,7 / 4,7 / 5,9**-Fache einer einzigen Menge (aber **0,12 / 0,16 / 0,29 / 0,36** des Zufalls) |
| dasselbe, Gitter 3×3, N = 1024 | Engpass: naiv **1,000** (6,2-mal), K = 8: Überdeckung **0,96**, RMSE das 1,6-Fache einer einzigen Menge; Sprung p = 0,9: naiv **1,000** (3,9-mal), K = 8: **0,95**, das 1,4-Fache |
| **Punktzahl**, Begleiter exp d = 8, Faktor Owen | N = 1024: **0,0031**; **N = 1000: 0,0072** (das 2,3-Fache); N = 1025: 0,0040; N = 1280: 0,0060; **N = 1536: 0,0038** (Faktor in der Größenordnung von 1024; absolut ist der Fehler kleiner, weil mehr Punkte) |
| **Reihenfolge der Koordinaten**, aᵢ = 1/i, d = 64, N = 1024, Faktor Owen | wichtige Koordinaten zuerst **0,0002**, zuletzt **0,0012** (digitale Verschiebung 0,0011 gegen 0,0021); an der Hantel ist kein Unterschied messbar, an den Gittern höchstens ein kleiner (bis etwa 20 %; Kontrollmessung) |
| **Hohe Dimension, kein belegter Verlust** (Kern, N = 1024, symmetrische Ziele) | Reihe (alle m Kanten halten, p = 0,99), m = 4 / 16 / 64: Faktor **0,03 / 0,08 / 0,29**; Parallel (eine von m hält, p = 0,1): **0,02 / 0,33 / um 1,0** (0,9 bis 1,2 je nach Strom), m = 64 im Rauschen des Zufalls, der ebenfalls um 1 liegt |
| **Kosten** (3×4, N = 4096) | Owen-Punkte erzeugen etwa **38 ms**, Sobol + Verschiebung 4,4 ms, Zufall 1,4 ms, Auswertung des Sprungs per Tabelle 0,22 ms (Wanduhr auf einem Windows-Rechner; getestet wird nur das Verhältnis: Owen-Punkte erzeugen dauert mehr als das Zehnfache der Tabellenauswertung und mehr als das Fünffache der Zufallspunkte): das Scrambling kostet bei billiger Auswertung mehr als es spart |

## Ehrliche Grenzen

- **Kein Verlust gegen den Zufall gemessen, aber nicht ausgeschlossen:** gemessen sind der Begleiter, drei Netze und drei symmetrische Sprungfamilien bis m = 64; für andere Integranden (Randsingularitäten) ist ein Verlust bekannt.
- **Der Gewinn hängt an effektiver Dimension und Glattheit zugleich;** die mittlere Dimension allein erklärt ihn nicht (3×4, p = 0,99: mittlere Dimension 2,02, Faktor erst 0,67).
- **Messungen mit schweren Rändern oder seltenen Ereignissen rauschen:** exp mit 32 oder 64 gleichen Gewichten, Sprung bei p = 0,99, Parallel mit m = 64 (etwa ein Fehlschlag je Menge). Dort sind Faktoren um 1 vom Zufall nicht zu trennen.
- **Kosten sind Punktzahlen (Modellauswertungen).** Das Scrambling selbst ist bei billiger Auswertung (Tabelle) teurer als die Auswertung; der Gewinn gilt für teure Modelle.
- **Eigene Sobol-Folge nur bis 64 Dimensionen,** keine Halton-Folge, keine Gittermengen, keine Netze höherer Ordnung, keine Dimensionsreduktion (Brownian Bridge, PCA).
- **Steigungen gelten für das Fenster N = 32 bis 2^k,** nicht asymptotisch; die theoretische Rate bei Owen-Scrambling und glatten Integranden (schneller als N^(−1)) zeigt sich nur bei kleiner Dimension.
- **Literaturangaben im Mathematik-Expander** stammen aus dem Gedächtnis und sind nicht eingesehen.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Zufallsstrom gegen Ganzzahl-Referenz; Richtungszahlen der zweiten Koordinate von Hand (1, 3, 5, 15); die ersten vier Sobol-Punkte von Hand; Netz-Eigenschaft der vier Punkte von Hand; Schätzung ohne Randomisierung (5/32); XOR der Verschiebung; Owen bei zwei und vier Punkten; Tiefe; Zusammenhangstabelle, Sprung und Engpass am Dreieck von Hand; Engpass-Wahrheit am Dreieck (½, 0,05); ANOVA des Pfads von Hand (¼, Anteile 2/3 und 1/3, mittlere Dimension 4/3); Begleiter; Blockintervall.
- **Orakel** (`tests/test_oracle_qmc.py`): **eingebettete Richtungszahlen ganzzahlig gleich `scipy.stats.qmc.Sobol`** (d = 64, N = 4096); **Netz-Eigenschaft aus der Definition** (jedes Kästchen vom Inhalt 1/64 in zwei Dimensionen genau ein Punkt) für Folge, Verschiebung und Owen, in drei Dimensionen **gleiche Kästchenzahlen nach der Randomisierung wie davor**; jede Einzelkoordinate mit einem Punkt je Intervall (17 Koordinaten, N = 1024); Gleichverteilung eines Einzelpunkts (Kolmogorov–Smirnov); **Erwartungstreue** (z-Band); Zusammenhangstabelle gegen `networkx`; glatter Integrand gegen die direkte Summe über alle Zustände; Wahrheit und Varianz der drei Netz-Integranden gegen 400 000 Zufallspunkte; Begleiter gegen Gauß–Legendre-Quadratur; **ANOVA-Schmetterling gegen bedingte Erwartungen und Möbius-Inversion** auf der Hantel; additive Anteile gegen Stück 5; Ordnungsanteile des Begleiters gegen Quadratur; Latin-Hypercube-Faktor gegen 1 − additiver Anteil.
- **Eigenschaften** (`tests/test_properties.py`): Ordnungsanteile sind eine Verteilung, mittlere Dimension wächst mit der Dimension, Zufall hat Faktor 1, Gewinn wächst mit N, nominale Dimension nicht entscheidend, Leiter geordnet, naiver Standardfehler überdeckt über 0,99 und Blöcke halten das Niveau, Nicht-Zweierpotenz kostet, wichtige Koordinaten nach vorn, Pipeline erwartungstreu, Reproduzierbarkeit.
- **Randfälle** (`tests/test_edge_cases.py`): d = 1, N = 1, 2, Dimensionsgrenzen, unbekannte Namen, konstanter Integrand (Varianz 0), Engpass einer Kante, kleinste Punktzahl in allen Experimenten, Eingabeprüfung der Spielwiese.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jede Auswahl jedes Reglers, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims** (`tests/test_claims.py`).

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `qmc_algorithm.py` | Kern: Strom, Sobol-Folge (64 Dimensionen), Verschiebung, Owen-Scrambling, Integranden, ANOVA, Intervalle |
| `qmc_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `qmc_evaluation.py` | Kennzahlen, Verdikt-Logik, Experimente über feste Ströme |
| `qmc_layout.py` | Fester Kopf (Stylesheet) |
| `qmc_pdf_export.py` | PDF der aktuellen Einstellung |
| `qmc_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufallsstrom-Knopf |
| `qmc_scenario.py` | Gitter, Hantel und eigene Kantenlisten |
| `qmc_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Gleichmäßig je Eingabe (Latin Hypercube, Schichtung) | [latin-hypercube-demo](https://github.com/sebastian-hanisch/latin-hypercube-demo) (Stück 5) |
| Halton-Folge, Gittermengen (Lattice), Netze höherer Ordnung | nicht gebaut |
| Dimensionsreduktion (Brownian Bridge, PCA) | nicht gebaut |
| Koksma–Hlawka, Diskrepanz-Rechnung | nur genannt |
| Welche Eingabe treibt die Varianz (Sobol-Indizes) | Stück 10 der Linie |
| Bayessche Quadratur | gelber Kandidat der Linie, nicht gebaut |

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

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
