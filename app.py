"""Randomisiertes Quasi-Monte-Carlo: wann gewinnt die Sobol-Folge? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 9 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Eine Sobol-Folge füllt den Würfel gleichmäßiger als Zufallspunkte; mit zufälligem Scrambling bleibt der Schätzer erwartungstreu und
liefert Fehlerbalken. Wie groß der Gewinn ist, hängt an der effektiven Dimension und an der Glattheit des Integranden, nicht an der nominalen Dimension. Vehikel sind die Netz-Integranden der
Reihe (Sprung, Engpass, glatt) und ein Begleiter exp(Σ aᵢuᵢ); Wahrheit, Varianz und effektive Dimension sind exakt.

Lauffähig mit: streamlit run app.py
"""

import dataclasses

import streamlit as st

import qmc_algorithm as A
import qmc_constants as C
import qmc_evaluation as ev
from qmc_evaluation import Settings, de, de_e, dn, dz, faktor
from qmc_layout import dunkles_design, kopf_css
from qmc_pdf_export import build_pdf
from qmc_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from qmc_scenario import named_netz, parse_edges
from qmc_visualization import build_curve, build_dimension, build_intervals, build_ladder, build_netz, build_points, build_sizes

st.set_page_config(page_title="Randomisiertes Quasi-Monte-Carlo: wann gewinnt die Sobol-Folge? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _table(netz_key):
    return ev.netz_table(named_netz(netz_key))


@st.cache_data(show_spinner="Fehlerkurve messen …")
def _analysis(settings, table):
    return ev.analyse(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Dimension und Gewinn messen …")
def _dimension(settings, table):
    return ev.dimension_experiment(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Glattheitsleiter messen …")
def _ladder(settings, table):
    return ev.smoothness_ladder(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Intervalle prüfen …")
def _intervals(settings, table):
    return ev.interval_experiment(settings, ev.make_integrand(settings, named_netz(settings.netz), table))


@st.cache_data(show_spinner="Punktzahl und Reihenfolge messen …")
def _sizes(settings, table):
    return ev.size_order_experiment(settings, ev.make_integrand(settings, named_netz(settings.netz), table))


@st.cache_data(show_spinner=False)
def _exact_rows(settings, table):
    return ev.dimension_rows_exact(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _custom(text, settings):
    netz = parse_edges(text)
    return netz, ev.custom_result(settings, netz)


def _netz_label(key):
    netz = named_netz(key)
    return f"{netz.name} ({netz.m} Kanten)"


def _p_label(p):
    return f"{p:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in qmc_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Randomisiertes Quasi-Monte-Carlo: wann gewinnt die Sobol-Folge?")
st.markdown(
    """
Zufallspunkte streuen ungleichmäßig; eine **Sobol-Folge** füllt den Würfel gleichmäßig. Mit **zufälligem Scrambling** (Owen) oder einer **digitalen Verschiebung** bleibt der Schätzer erwartungstreu und
man bekommt Fehlerbalken. Wie viel das bringt, hängt aber nicht an der Zahl der Eingaben, sondern an der **effektiven Dimension** (wie viele Eingaben wirklich zusammenwirken) und an der **Glattheit**
des Integranden: ein Polynom gewinnt um Größenordnungen; Knick und Sprung gewinnen meist mäßig bis deutlich, ihre Reihenfolge hängt vom Netz ab, und bei vielen wirksamen Eingaben schrumpft der Gewinn. Dazu kommen zwei Fallen: der **naive
Fehlerbalken** (die Punkte wie unabhängig behandelt) ist bei großem Gewinn viel zu groß (bis 34-fach), bei kleinem Gewinn nur wenig, und eine Punktzahl, die **keine Zweierpotenz** ist, kostet Genauigkeit.

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert randomisiertes Quasi-Monte-Carlo", expanded=True):
    st.markdown(
        """
1. **Sobol-Folge:** Punkte aus Richtungszahlen und dem Gray-Code des Zählers. Die ersten N = 2ᵏ Punkte bilden ein **Netz**: in jeder Einzelkoordinate liegt genau ein Punkt in jedem Intervall der Länge 2⁻ᵏ,
   und auch in vielen mehrdimensionalen Kästchen liegt die richtige Zahl.
2. **Randomisieren:** Bei der **digitalen Verschiebung** wird jede Koordinate mit einer Zufallszahl »verknüpft« (XOR der Bits). Beim **Owen-Scrambling** wird jedes Bit mit einem Zufallsbit gekippt, das nur von den höheren Bits
   abhängt. Beides erhält die Netz-Eigenschaft und macht jeden Punkt gleichverteilt, der Schätzer bleibt erwartungstreu.
3. **Schätzen:** Mittel von f über die N Punkte. Der Fehler hängt davon ab, wie sich f auf Wechselwirkungen niedriger Ordnung verteilt (**ANOVA**): wenige wirksame Eingaben und ein glatter Verlauf ⇒ großer Gewinn.
4. **Fehlerbalken:** K unabhängige Randomisierungen ⇒ K Mittelwerte ⇒ t-Intervall. Die N Punkte einer Menge sind nicht unabhängig; der naive Standardfehler passt nicht.
        """
    )
    st.plotly_chart(build_points(C.DEFAULT_SEED), width="stretch", key="points_chart")
    st.caption("Die ersten beiden Koordinaten von 64 Punkten: Zufall (Klumpen und Lücken), Latin Hypercube (je Zeile und Spalte einer) und Sobol mit Owen-Scrambling (gleichmäßig in vielen Kästchen).")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Integrand**")
    integrand = st.selectbox("Integrand", options("integrand_select"), format_func=C.INTEGRANDS.get, key="integrand_select",
                             help="Sprung: das Netz hängt zusammen (0 oder 1). Engpass: das größte t, bei dem es noch zusammenhängt (stetig, mit Knicken). Glatt: Zuverlässigkeit bei unsicheren Haltewahrscheinlichkeiten (ein Polynom). Begleiter: exp(Σ aᵢuᵢ) mit wählbarer Dimension.")
    netz_key = st.selectbox("Netz", options("netz_select"), format_func=_netz_label, key="netz_select", disabled=integrand == "exp",
                            help="Gitter oder die Hantel (zwei Dreiecke, durch eine Brücke verbunden). Jede Kante hängt von einer Gleichverteilten ab, die Dimension ist die Kantenzahl.")
    p = st.select_slider("Haltewahrscheinlichkeit p je Kante", options("p_select"), key="p_select", format_func=_p_label, disabled=integrand != "sprung", help="Nur für den Sprung: jede Kante hält, wenn ihre Gleichverteilte unter p liegt.")
    d = st.select_slider("Dimension d des Begleiters", options("d_select"), key="d_select", disabled=integrand != "exp", help="Zahl der Koordinaten von exp(Σ aᵢuᵢ).")
    weight = st.selectbox("Gewichte aᵢ", options("weight_select"), format_func=C.WEIGHTS.get, key="weight_select", disabled=integrand != "exp",
                          help="Gleiche Gewichte: alle Koordinaten zählen. Abfallende Gewichte: spätere Koordinaten zählen kaum, die effektive Dimension bleibt klein.")
    st.markdown("**Punkte**")
    nexp = st.select_slider("Punkte N = 2ᵏ", options("nexp_select"), key="nexp_select", format_func=lambda k: f"2^{k} = {2 ** k}", help="Zahl der Punkte je Menge; die Fehlerkurve beginnt bei 32.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"integrand_select": integrand, "netz_select": netz_key, "p_select": float(p), "d_select": int(d), "weight_select": weight, "nexp_select": int(nexp), "seed_input": int(seed)})

settings = Settings(str(integrand), str(netz_key), float(p), int(d), str(weight), int(nexp), int(seed))
table = _table(settings.netz)
core = settings                                                             # Auswertungen hängen von allen Einstellungen ab
a = _analysis(core, table)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"Integrand: {a.integ.label}; {a.integ.d} Dimensionen; {de(a.N, 0)} Punkte; {a.reps} Wiederholungen je Punktzahl.")
if a.integ.var > 0:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wahrheit (exakt)", dz(a.integ.truth), help="Exakter Wert des Integrals (Aufzählung bzw. geschlossene Formel).")
    m2.metric("Faktor Sobol + Owen", faktor(a.factor_owen), delta="Zufall = 1", delta_color="off", help="Mittlerer quadratischer Fehler geteilt durch die Monte-Carlo-Varianz σ²/N. Unter 1: Gewinn.")
    m3.metric("RMSE Zufall : Owen", f"{dn(a.ratio, 1)}×", help="Wie viel mal kleiner der Fehler der Owen-Sobol-Folge ist.")
    dims = a.integ.dims
    m4.metric("Mittlere Dimension (exakt)", dn(dims["mean_dim"], 2) if dims else "–", delta=f"additiv {de(dims['additive'] * 100, 0)} %" if dims else None, delta_color="off",
              help="Mittlere Zahl wechselwirkender Eingaben nach der ANOVA; der additive Anteil ist, was Latin Hypercube höchstens holen kann. Für den Engpass nicht berechnet.")
    st.plotly_chart(build_curve(a), width="stretch", key="main_chart")
    st.caption("Fehler (RMSE) gegen die Punktzahl, doppelt logarithmisch. Die gepunktete Linie hat die Steigung −0,5 des Zufalls; je steiler eine Kurve, desto besser.")
else:
    st.info("Der Integrand ist hier konstant: es gibt nichts zu schätzen.")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="quasi-monte-carlo.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann schlägt eine Sobol-Folge die Zufallsziehung?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Fehler gegen die Punktzahl")
st.caption("Die Kurve im Bild oben: vier Punktmengen, je 100 Wiederholungen mit festen Strömen. Die Steigung ist der Maßstab; Zufall fällt mit −0,5.")
kind, text = ev.verdict_curve(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Der Gewinn hängt an der effektiven Dimension, nicht an der nominalen")
st.caption("Exakt (kein Zufall): die ANOVA nach Ordnung der Wechselwirkung für den Begleiter bei jeder Dimension und jedem Gewichtsabfall und für die Netz-Integranden. "
           f"Der Knopf misst dazu den Gewinn der Owen-Sobol-Folge bei N = {de(settings.N, 0)} ({C.GRID_REPS} Wiederholungen je Zeile).")
exact_rows = _exact_rows(core, table)
st.dataframe(
    {"Integrand": [r["label"] for r in exact_rows], "Dimension": [r["d"] for r in exact_rows], "mittlere Dimension": [de(r["dims"]["mean_dim"], 2) for r in exact_rows],
     "90 % bis Ordnung": [r["dims"]["sup90"] for r in exact_rows], "additiver Anteil": [f"{de(r['dims']['additive'] * 100, 0)} %" for r in exact_rows]},
    hide_index=True, width="stretch", key="dim_exact_table",
)
if st.button("▶ Gewinn messen", key="run_dimension"):
    st.session_state["dimension_done"] = True
if st.session_state.get("dimension_done"):
    ex2 = _dimension(dataclasses.replace(core, integrand="glatt", p=C.DEFAULT_P, d=C.DEFAULT_D, weight=C.DEFAULT_WEIGHT), table)
    st.plotly_chart(build_dimension(ex2), width="stretch", key="dim_chart")
    st.dataframe(
        {"Integrand": [r["label"] for r in ex2["rows"]], "mittlere Dimension": [de(r["dims"]["mean_dim"], 2) for r in ex2["rows"]], "Faktor Sobol + Owen": [faktor(r["factor_owen"]) for r in ex2["rows"]],
         "Faktor Latin Hypercube": [faktor(r["factor_lhs"]) for r in ex2["rows"]], "Latin Hypercube vorhergesagt (1 − additiv)": [faktor(r["lhs_pred"]) for r in ex2["rows"]]},
        hide_index=True, width="stretch", key="dim_table",
    )
    kind, text = ev.verdict_dimension(ex2, settings)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Die Glattheitsleiter am selben Netz")
st.caption(f"Dieselben Kanten, drei Integranden: der Sprung »hängt zusammen«, der stetige Engpass mit Knicken und die glatte Zuverlässigkeit bei unsicheren Haltewahrscheinlichkeiten. {C.GRID_REPS} Wiederholungen, N = {de(settings.N, 0)}.")
if st.button("▶ Messung starten", key="run_ladder"):
    st.session_state["ladder_done"] = True
if st.session_state.get("ladder_done"):
    ex3 = _ladder(dataclasses.replace(core, integrand="glatt", d=C.DEFAULT_D, weight=C.DEFAULT_WEIGHT), table)
    st.plotly_chart(build_ladder(ex3), width="stretch", key="ladder_chart")
    kind, text = ev.verdict_ladder(ex3)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Der ehrliche Fehlerbalken")
st.caption(f"Überdeckung des 95-%-Intervalls für den gewählten Integranden: Zufall, Owen mit naivem Standardfehler (die N Punkte wie unabhängig behandelt) und Owen mit K = 4, 8, 16, 32 unabhängigen Randomisierungen (jeweils N/K Punkte). "
           f"{C.CI_REPS} Wiederholungen. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_intervals"):
    st.session_state["intervals_done"] = True
if st.session_state.get("intervals_done") and a.integ.var > 0:
    ex4 = _intervals(core, table)
    st.plotly_chart(build_intervals(ex4), width="stretch", key="ci_chart")
    st.dataframe(
        {"Verfahren": [r["label"] for r in ex4["rows"]], "Überdeckung": [de(r["coverage"], 3) for r in ex4["rows"]], "Halbbreite / RMSE": [dn(r["hw_ratio"], 2) for r in ex4["rows"]],
         "RMSE gegen eine Menge mit N Punkten": [dn(r["vs_single"], 2) for r in ex4["rows"]], "RMSE gegen Zufall": [dn(r["vs_mc"], 3) for r in ex4["rows"]]},
        hide_index=True, width="stretch", key="ci_table",
    )
    kind, text = ev.verdict_intervals(ex4)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Zweierpotenz und Reihenfolge der Koordinaten")
st.caption(f"Die ersten N Sobol-Punkte für N = {de(settings.N, 0)} und vier andere Punktzahlen; dazu der Begleiter mit abfallenden Gewichten (d = 64) mit den wichtigen Koordinaten zuerst oder zuletzt. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_sizes"):
    st.session_state["sizes_done"] = True
if st.session_state.get("sizes_done") and a.integ.var > 0:
    ex5 = _sizes(core, table)
    st.plotly_chart(build_sizes(ex5), width="stretch", key="sizes_chart")
    st.dataframe(
        {"Punktzahl": [f"{r['N']} ({r['name']})" for r in ex5["sizes"]], "Faktor Owen": [faktor(r["factor_owen"]) for r in ex5["sizes"]], "Faktor Verschiebung": [faktor(r["factor_ds"]) for r in ex5["sizes"]]},
        hide_index=True, width="stretch", key="sizes_table",
    )
    st.dataframe(
        {"Reihenfolge": [r["name"] for r in ex5["order"]], "Faktor Owen": [faktor(r["factor_owen"]) for r in ex5["order"]], "Faktor Verschiebung": [faktor(r["factor_ds"]) for r in ex5["order"]]},
        hide_index=True, width="stretch", key="order_table",
    )
    kind, text = ev.verdict_sizes(ex5)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz"):
    st.caption(
        f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; Integrand (beim Begleiter: der Engpass), p, N und Strom stammen aus der Seitenleiste. "
        "Es zeigt die Fehlerkurve und die exakte effektive Dimension."
    )
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-0\n3-4\n4-5\n5-3\n2-3", height=140, key="eigene_kanten")
    try:
        cnetz, cres = _custom(text_in, core)
    except ValueError as err:
        st.error(str(err))
    else:
        ca = cres["analysis"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Wahrheit (exakt)", dz(ca.integ.truth))
        s2.metric("Faktor Sobol + Owen", faktor(ca.factor_owen))
        s3.metric("Mittlere Dimension", dn(ca.integ.dims["mean_dim"], 2) if ca.integ.dims else "–")
        st.plotly_chart(build_netz(cnetz), width="stretch", key="custom_net")
        st.plotly_chart(build_curve(ca), width="stretch", key="custom_curve")
        kind, text = ev.verdict_main(ca, core)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Integrand ist glatt** | Bei Knicken und vor allem Sprüngen bleibt meist nur ein mäßiger Gewinn und die Steigung liegt zwischen der des Zufalls (−0,5) und −0,75 (Experimente 1 und 3). | – |
| **Die effektive Dimension ist klein** | Wirken viele Eingaben zusammen, schrumpft der Gewinn auf einen kleinen Faktor (Experiment 2); die nominale Dimension allein sagt wenig. | – |
| **Die Punktzahl ist eine Zweierpotenz** | Mit 1000 statt 1024 Punkten geht die Netz-Eigenschaft teilweise verloren (Experiment 5). | – |
| **Fehlerbalken aus Randomisierungen** | Der naive Standardfehler ist bei großem Gewinn viel zu groß (bis 34-fach), bei kleinem Gewinn nur wenig; K Blöcke decken richtig ab, kosten aber Genauigkeit (Experiment 4). | – |
| **Wichtige Koordinaten stehen vorn** | Bei abfallenden Gewichten kostet eine ungünstige Reihenfolge etwa das 6- bis 8-Fache im mittleren quadratischen Fehler (Experiment 5); an den Netzen ist der Unterschied klein (bis etwa 20 %, nicht in der App gemessen). | – |
| **Die Auswertung ist teuer** | Das Scrambling kostet bei billiger Auswertung mehr Zeit als es spart; der Gewinn gilt in Modellauswertungen. | – |
| **Nur Haupteffekte gleichmäßig abdecken** | Das leistet Latin Hypercube; die Sobol-Folge deckt zusätzlich Wechselwirkungen niedriger Ordnung ab. | `latin-hypercube-demo` (Stück 5) |
| **Welche Eingabe treibt die Varianz** | Das ist eine eigene Frage (Sobol-Indizes). | Stück 10 der Linie |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze und Integranden sind **erfunden**; Wahrheit, Varianz und effektive Dimension sind exakt (Aufzählung aller Kantenmengen bzw. geschlossene Formeln). Die Kanten sind unabhängig. Die Zufallszahlen
kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Aufgezählt wird live bis 17 Kanten; eingebettet sind die Richtungszahlen für 64 Dimensionen (Joe und Kuo). Es gibt Sobol-Folgen mit Verschiebung und Owen-Scrambling, aber keine Halton-Folge,
keine Gittermengen und keine Netze höherer Ordnung. Die Steigungen der Fehlerkurve gelten für das Fenster N = 32 bis 2ᵏ, nicht asymptotisch. Gemessen wird bis N = 4096.

**Bekannte Probleme und Unsicherheiten.** Messungen bei seltenen Ereignissen (Sprung mit p = 0,99) rauschen stark; ein Faktor um 1 liegt dort im Rauschen. Dass die Sobol-Folge in hoher Dimension schlechter als der Zufall wird,
ist hier **nicht** gemessen worden (der Gewinn verschwindet nur); für Integranden mit Randsingularitäten ist ein Verlust aus der Literatur bekannt, hier nicht geprüft. Die Literaturangaben im Mathematik-Expander stammen aus dem
Gedächtnis und sind nicht eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gesucht ist $I = \int_{[0,1]^d} f(u)\,du$. Der Monte-Carlo-Schätzer $\hat I = \frac1N\sum_i f(u_i)$ mit unabhängigen $u_i$ hat die Varianz $\sigma^2/N$; als Kennzahl dient der **Faktor** $\mathrm{MSE}/(\sigma^2/N)$.

**Sobol-Folge.** Zur Koordinate $j$ gehören Richtungszahlen $v_{1},v_2,\dots$ (aus einem primitiven Polynom und Anfangswerten); der $i$-te Punkt ist das XOR der $v_k$ über die gesetzten Bits des Gray-Codes von $i$.
Die ersten $2^k$ Punkte bilden ein digitales $(t,k,d)$-Netz in Basis 2: In jeder Koordinate liegt genau ein Punkt in jedem Intervall der Länge $2^{-k}$.

**Randomisierung.** Digitale Verschiebung: $x \mapsto x \oplus r$ mit zufälligem $r$ je Koordinate. Verschachteltes Owen-Scrambling: Bit $j$ von $x$ wird mit einem Zufallsbit $\beta(x_1,\dots,x_{j-1})$ gekippt. Beide erhalten die Netz-Eigenschaft
und machen jeden Punkt gleichverteilt, also $\mathbb E\hat I = I$. Ab der Tiefe $k$ sind die Präfixe der Punkte einer Koordinate paarweise verschieden, die tieferen Bits sind unabhängig gleichverteilt (hier durch Zufallszahlen ersetzt).

**Effektive Dimension.** ANOVA: $f=\sum_{S}f_S$, $\sigma_S^2=\operatorname{Var}f_S$. Mittlere Dimension $\sum_S |S|\sigma_S^2/\sigma^2$; Überlagerungsdimension: kleinste Ordnung mit 90 % der Varianz. Für den Begleiter $\exp(\sum a_iu_i)$ ist $f$ ein Produkt;
die Ordnung der Wechselwirkung ist Poisson-Binomial mit $r_i/(1+r_i)$, $r_i=\operatorname{Var}g_i/(\mathbb E g_i)^2$. Für die Netz-Integranden hängt $f$ von $u_e$ nur über $x_e=\mathbf 1[u_e<p]$ ab; in der Basis $(x_e-p)/\sqrt{p(1-p)}$ liefert ein Schmetterlingsschritt je Kante alle $2^m$ Koeffizienten.
Der glatte Integrand $f(u)=R(p_1(u_1),\dots,p_m(u_m))$ mit $p_e=\bar p+\delta(u_e-\tfrac12)$ ist multilinear in den $p_e$, also ein Polynom in $u$ mit $\mathbb E f=R(\bar p)$ und $\operatorname{Var} f=\sum_{S\ne\emptyset}c_S^2\,(\delta^2/(12\bar p\bar q))^{|S|}$.

**Engpass.** $B(u)=\max\{t:\text{das Netz hängt mit den Kanten }\{u_e\ge t\}\text{ zusammen}\}$; $\mathbb P(B\ge t)=R(1-t)$, also $\mathbb E B=\int_0^1R(p)\,dp=\sum_k N_k\,k!\,(m-k)!/(m+1)!$.

**Latin Hypercube.** Entfernt den additiven Anteil: $\operatorname{Var}\approx(\sigma^2-\sum_i\sigma_i^2)/N$ (Stück 5).

**Intervall.** $K$ unabhängige Randomisierungen mit je $N/K$ Punkten ⇒ Mittelwerte $\bar y_1,\dots,\bar y_K$; Intervall $\bar y \pm t_{K-1,0{,}975}\,s/\sqrt K$. Der naive Standardfehler $s_f/\sqrt N$ aus den $N$ Funktionswerten ist zu groß, weil die Werte negativ korreliert sind.

**Literatur (aus dem Gedächtnis, nicht eingesehen).** Sobol 1967, „On the distribution of points in a cube and the approximate evaluation of integrals“ (USSR Computational Mathematics and Mathematical Physics); Joe und Kuo 2008, „Constructing Sobol sequences with better two-dimensional projections“ (SIAM Journal on Scientific Computing); Owen 1995, „Randomly permuted (t,m,s)-nets and (t,s)-sequences“; Owen 1997, „Scrambled net variance for integrals of smooth functions“ (Annals of Statistics); Caflisch, Morokoff und Owen 1997, „Valuation of mortgage-backed securities using Brownian bridges to reduce effective dimension“ (Journal of Computational Finance); L'Ecuyer und Lemieux 2002, „Recent advances in randomized quasi-Monte Carlo methods“.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
