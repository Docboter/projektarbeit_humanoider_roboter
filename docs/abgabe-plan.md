# Plan: Doku und Ausarbeitung bis zur Abgabe

> **TL;DR:** Arbeitsplan für die Abgabe der Projektarbeit (Stand 2026-10-29): Markdown-Doku auf
> einen kurzen Einstieg plus Schnittstellen-Seite umbauen, LaTeX-Ausarbeitung kürzen und neu
> gliedern (Schnittstellen und Anleitung zum Weiterarbeiten im Mittelpunkt), dann gemeinsam
> prüfen. Messergebnisse stehen nicht hier, sondern in der
> [Diagnose-Chronik](ergebnisse/diagnose-chronik.md).

## Stand der Umsetzung (2026-10-29)

| Phase | Stand |
|---|---|
| 0 Entscheidungen | erledigt (Tabelle unten) |
| 1 A1, A2 | erledigt: [`schnittstellen.md`](schnittstellen.md) aus dem Code (Beispiel-Client gegen den `PolicyServer` des Forks getestet), README/Quickstart neu, Status in CLAUDE.md/AGENTS.md |
| 2 A3, A4 | erledigt: Inventur (Tabelle in A3), 6 Dateien nach [`archiv/`](archiv/README.md), „Ergebnisse auf einen Blick“, 11 veraltete Stellen mit Nachtrag, Link-Check `tools/check_links.py` in der CI |
| 3 B1, B2 | erledigt: 8 Kapitel nach B1, 20 S. Hauptteil / 41 S. gesamt; Abbildungen aus `latex/figures/make_figures.py` (Gate, Closed-Loop-Stufen, Fingerspanne, Orakel-Raster) plus Architektur, Echt-/Sim-Bild, erster Griff. Lernkurven Lauf 3/4 aus den Offline-Logs auf KISSKI (2026-10-30) |
| 4 Prüfen | offen: Zahlen-Stichprobe durch die Gruppe, Fremdtest auf dem IKR-Server, Absatz-Test |
| 5 B3 | teilweise: Zusammenfassung DE/EN neu, Literatur bereinigt (jeder Eintrag zitiert), `latex/README.md` aktualisiert. Offen: Matrikelnummern A/B, Abgabedatum, Tag, Sichtbarkeit der HF-Repos |

**Nur auf dem IKR-Server prüfbar:**

1. Quickstart §2 einmal komplett: `preflight`, `tools/sim_visgate_seeds.sh groot-rl 1.6 0 …`, `tools/sim_cl_watchdog.sh …` mit dem Standard-Container `groot-rl` (legt `eval` ihn an, wenn er fehlt?).
2. Liegen `/data/cotrain/vis_gate_v3_cube{5,4}` und `/data/eval/layout_heldout_dy3.json` dort, und wie wurde die dy3-Datei erzeugt (kein Skript im Repo)?
3. Die Beispielausgabe in `sim-eval-anleitung.md` (Format aus dem Code) gegen ein echtes Log halten.
4. Welche Läufe den zeitweise gemischten v2.1/v3.0-Datensatz gelesen haben (Lauf 56, offen).

**W&B nachgeholt (2026-10-30):** Lauf 4 (`2v02ye81`) und die Co-Training-Läufe 73–76 waren nur offline auf KISSKI; jetzt synchronisiert. Der Sync mit dem `wandb` im Image schlägt still fehl, Weg mit aktuellem wandb in [wandb-offline-sync.md](training/wandb-offline-sync.md#sync-mit-aktuellem-wandb-ab-2026-10-30).

**Neu gefunden:** `Training/kisski_open_loop_eval.sh` nutzt standardmäßig 6 Episoden × 300 Schritte,
das Protokoll, das in Lauf 3 ein Scheinoptimum lieferte. Nicht geändert; in LaTeX Kap. 4/8 und
hier vermerkt. Vorschlag: Default auf 20 × 750.

## Leitlinie

Rückmeldung des Betreuers (Prof. Dr. Lorenzen, über ein Gruppenmitglied): **wenig Text, der nichts
aussagt.** Gewünscht ist eine kurze, gute Doku, in der die **Schnittstellen** eingetragen sind,
und eine **gute Anleitung**, mit der man einfach weiterarbeiten kann.

**Prüfregel für jeden Absatz** (LaTeX und Markdown). Ein Absatz bleibt nur, wenn er mindestens
eines davon enthält:

- eine Tatsache mit Zahl oder Quelle,
- eine Entscheidung mit Grund,
- eine Anweisung (Befehl, Schritt),
- eine Schnittstellen-Angabe (Format, Name, Port, Variable, Pfad).

Gestrichen werden Meta-Sätze („Dieses Kapitel beschreibt …“), Wiederholungen,
Absicherungsfloskeln und Erzählungen des Arbeitsprozesses ohne Lehre. Lieber eine Tabelle oder
Abbildung als drei Absätze.

**Eine Quelle je Zahl.** Messwerte kommen aus der [Chronik](ergebnisse/diagnose-chronik.md)
bzw. den CSVs unter [`latex/figures/daten/`](../latex/figures/daten/README.md). Nichts wird neu
gerundet oder umgedeutet. Was die Chronik vorsichtig formuliert (z. B. „Griffrate 3 %, gegen
0/20 nicht signifikant“), bleibt vorsichtig.

## Ausgangslage (2026-10-29)

| | Umfang | Zustand |
|---|---|---|
| Markdown | 53 Dateien unter `docs/`, ~170 000 Wörter (Chronik allein 34 000) | aktuell bis Lauf 97; Einstieg verstreut, keine Schnittstellen-Seite, mehrere veraltete Plan-/Review-Dateien |
| LaTeX | 8 Kapitel + Anhang, ~11 000 Wörter (zuletzt 28 S. Hauptteil / 48 S. gesamt), 3 Abbildungen | inhaltlich Stand 2026-10-11: es fehlen N1.7, Orakel/Registrierung (Läufe 57–69), visueller Domain-Gap (66–71), Co-Training v3 (72–97) |

Kernaussage heute (für Zusammenfassung und Fazit). Zwei Blocker der Sim sind gefunden und
gemessen:

1. **Registrierung Würfel↔Hand:** In der Sim liegen die Würfel je Seite ~3 cm zu weit innen. Mit
   Korrektur hebt die menschliche Demo in 30 % statt ~0 % (Läufe 62, 69).
2. **Visueller Domain-Gap:** Auf Sim-Bildern ist der Hand-Fehler ×13 (N1.6) bzw. ×36 (N1.7)
   (Lauf 66).

Das Co-Training mit Sim-Renderbildern (Läufe 73–97) zeigt Folgendes:

- Es halbiert diesen Fehler (N1.6) bzw. senkt ihn auf 18–30 % (N1.7).
- Das vorregistrierte primäre Gate ist bestanden: Fingerspanne 73 % statt 25 % der Demo.
- Die Politik erreicht und berührt die Würfel in jeder Episode.
- Echte Griffe bleiben selten (2/60 = 3 %), gestapelt wird nie.
- N1.6 zahlt auf echten Bildern einen Preis (+50–60 % MSE Hände), N1.7 nicht.
- Ein Test auf dem Roboter fehlt.

## Teil A: Markdown-Doku

### A1. Neue Seite `docs/schnittstellen.md` (Kern, zuerst)

Sie ist die eine Quelle für alle Schnittstellen; LaTeX Kapitel 3 baut darauf auf. Geschrieben wird
sie **aus dem Code**, nicht aus der alten Doku; jede Angabe mit Datei-Verweis.

| Abschnitt | Inhalt | Quellen im Code |
|---|---|---|
| Datensatz | LeRobot-Format (v3.0 auf HF, v2.1 nach Konvertierung), `meta/modality.json`, vier Kameras mit Namen und Auflösung, Zustands- und Aktionsvektor (Dimensionen, Gelenkreihenfolge, Einheiten), Split 0:240 / 240:301 | `Training/scripts/entrypoint.sh`, `app/Groot-1.*/examples/G1_DEX3/modality_4cam.json`, `Training/scripts/lib_split.sh` |
| Checkpoint | Verzeichnisinhalt (`config.json`, Shards, `experiment_cfg/`, `trainer_state.json`, `wandb_config.json`), `GROOT_VERSION` 1.6/1.7 (nicht kreuzweise ladbar), Embodiment-Tag, HF-Namensschema `luca-mue/groot-g1dex3-*`, Upload ohne Optimizer | `Simulation/scripts/upload_checkpoint.py`, `app/Groot-1.*/examples/G1_DEX3/` |
| Policy-Server | ZMQ `tcp://localhost:5555`, Anfrage (Beobachtung: Bilder, Zustand, Prompt) und Antwort (Aktionsblock), Aktionshorizont gegen `EXECUTION_HORIZON`, Takt 30 Hz, Latenz-Budget 267 ms | `app/Groot-1.*/gr00t/eval/run_gr00t_server.py`, `Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py` |
| Container und Konfiguration | Images und Tags (Training `:latest`/`:latest-n17`, Sim `…-sim-standalone`), Einstiegspunkte (`run.sh`, `server_rl_run.sh <aktion>`, `kisski_chain.sh`, `kisski_submit.sh`), `/data`-Layout, wichtigste Umgebungsvariablen (Verweis auf [`training/env-vars.md`](training/env-vars.md)) | `Training/`, `Simulation/server_rl_run.sh`, `tools/menu/*.spec` |
| Sim-Eval | Eingaben: Layout-JSON (Format, `n_train`, Lagen in sortierter Reihenfolge), Szenen-Variablen (`SIM_TABLE_SIZE`, `SIM_WRIST_*`, `SIM_CUBE_EDGE`); Ausgaben: `results_*.json` (Felder je Episode), Metrik-Definitionen `reached` … `stacked` inkl. Grenzen von `lifted_clear` (Lauf 94), Gate-JSON `visgate_*` | `Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py`, `Simulation/g1_dex3_sim/eval_metrics.py`, `Simulation/scripts/finger_span_openloop.py` |
| Werkzeuge | `tools/sim_cl_watchdog.sh`, `tools/sim_visgate_seeds.sh`, `tools/sim_cl_summary.py`, `tools/sim_visgate_seeds_summary.py` | `tools/` |
| Echter Roboter | was **nicht** gebaut ist (keine Anbindung an den G1) und was der Policy-Server dafür von einer Roboter-Seite erwartet | — |

**Fertig, wenn** jede Zeile eine Datei- oder Code-Referenz hat und eine Person ohne
Projektwissen daraus einen eigenen Client für den Policy-Server schreiben könnte.

### A2. Einstieg auf drei Seiten

1. [`README.md`](../README.md): was das Projekt ist, Ergebnis in drei Sätzen, Karte des Repos,
   Verweis auf die zwei folgenden Seiten. Status-Absatz auf 2026-10-29 bringen.
2. [`docs/quickstart.md`](quickstart.md): in ~30 min von null zu (a) Training starten,
   (b) Gate und Closed Loop für einen HF-Checkpoint messen. Nur Befehle und deren Erwartung.
3. `docs/schnittstellen.md` (A1).

Alles andere ist Referenz und wird von dort verlinkt. Auch [`CLAUDE.md`](../CLAUDE.md) und
[`AGENTS.md`](../AGENTS.md) bekommen den aktuellen Status bzw. Verweis; der Status-Absatz in
`CLAUDE.md` steht noch auf 2026-10-13.

### A3. Inventur aller 53 Dateien

Für jede Datei unter `docs/` gibt es eine Entscheidung, als Tabelle in diesem Plan festgehalten:
**behalten und prüfen**, **zusammenführen** (mit Ziel) oder **archivieren** (nach
`docs/archiv/`, Kopfzeile „archiviert am …, ersetzt durch …“, Links umbiegen). Historie wird
nicht gelöscht (CLAUDE.md: Überholtes nach `historie.md` oder mit „überholt durch …“).

Kandidaten fürs Archiv (prüfen, nicht blind verschieben):

- `projekt-review-2026-10-11.md`, `projekt-review-2026-10-13.md` (Befunde sind umgesetzt oder in
  `next-steps.md`),
- `weiterfuehrend/livestream-plan.md`, `weiterfuehrend/wiki-migration-plan.md`,
- `ergebnisse/lauf1-zwischenstand.md`,
- `simulation/wuerfellage-rekonstruktion-lauf35-befund.md`,
- `simulation/archiv/` (bleibt Archiv, ggf. nach `docs/archiv/` ziehen).

Kandidaten zum Zusammenführen: die Lauf-Auswertungen 1–4 unter einer Ergebnis-Übersicht;
`simulation/baseline-eval.md` / `basismodell-referenz*.md` / `robocasa-referenz-eval.md`.

**Inventur (umgesetzt 2026-10-29).** 54 Dateien vorher (mit diesem Plan), 56 nachher (neu: `schnittstellen.md`,
`archiv/README.md`).

| Bereich | Behalten (Referenz) | Archiviert nach `docs/archiv/` | Bemerkung |
|---|---|---|---|
| Einstieg | `README.md` (Hub), `quickstart.md`, `schnittstellen.md` (neu) | — | A2 |
| Querschnitt | `fehlerbehebung.md`, `portabilitaet.md`, `historie.md`, `abgabe-plan.md` | `projekt-review-2026-10-11.md`, `projekt-review-2026-10-13.md` | Review-Befunde umgesetzt oder in `next-steps.md` |
| `training/` | alle 10 | — | `env-vars.md` bleibt die Variablen-Referenz |
| `augmentation/` | beide | — | Status „gebaut, nicht geprüft“ bleibt |
| `simulation/` | `README`, `sim-eval-anleitung`, `umsetzungsnotizen`, `live-ansicht`, `inferenz-optimierung`, `replay-videos-aus-realdaten`, `wuerfellage-rekonstruktion`, `wuerfellage-rekonstruktion-bewertung`, `basismodell-referenzaufgabe`, `robocasa-referenz-eval`, `archiv/` (4) | `wuerfellage-rekonstruktion-lauf35-befund.md`, `baseline-eval.md` | `simulation/archiv/` bleibt, weil Code darauf verweist; `basismodell-referenzaufgabe` wird von den RoboCasa-Skripten zitiert |
| `ergebnisse/` | `README` (mit „auf einen Blick“), `diagnose-chronik`, `lauf1-auswertung` … `lauf4-n17-vergleich`, `sim-bewertung`, `domain-gap-analyse`, `basismodell-referenz-eval` | `lauf1-zwischenstand.md` | Lauf-Auswertungen 1–4 bleiben als Detail, die Übersicht fasst sie zusammen |
| `weiterfuehrend/` | `README`, `cli-menuefuehrung`, `groot-n17-migration`, `livestream-plan`, `lokomotion-recherche`, `reinforcement-learning-plan`, `rl-anleitung` | `wiki-migration-plan.md` | `livestream-plan` bleibt: Code verweist auf ihn als Entwurf |

Zusammenführen statt archivieren wurde verworfen, wo die Dateien Konzept, Bedienung und Ergebnis
sauber trennen und sich gegenseitig verlinken (RoboCasa-Referenz).

`diagnose-chronik.md` bleibt **unverändert** als Laborbuch. Neu dazu kommt in
[`ergebnisse/README.md`](ergebnisse/README.md) eine Seite „Ergebnisse auf einen Blick“
(höchstens eine Bildschirmseite, Tabellen, Verweise auf Lauf-Nummern).

### A4. Korrektheit prüfen

- Jeden Befehl in README, Quickstart, `training/anleitung.md`, `simulation/sim-eval-anleitung.md`
  ausführen, mindestens als `--help` oder Trockenlauf. Was nicht mehr stimmt, korrigieren.
- `tools/gen_docs.sh`, `tools/check_tldr.sh`, `tools/test_menu.sh`, `ruff check .` laufen lassen.
- Alle Markdown-Links und -Anker prüfen. Bekannt kaputt: Anker
  `#lauf-62-versatz-sweep-die-würfel-liegen-3-cm-zu-weit-innen` in der Chronik.
- Veraltete „Stand“-Angaben suchen (`grep -rn "Stand" docs`) und je Datei entscheiden:
  aktualisieren oder als historisch markieren.

## Teil B: LaTeX-Ausarbeitung (`latex/`)

### B1. Neue Gliederung (Ziel ~20–25 Seiten Hauptteil, endgültig nach Entscheidung E1)

| Kap. | Inhalt | Umfang | Material |
|---|---|---|---|
| 1 Einleitung | Aufgabe, Ergebnis in fünf Sätzen, Aufbau | 1 S. | heutiges Kap. 1, stark gekürzt |
| 2 System im Überblick | **ein** Architekturbild (Datensatz → Training Docker/KISSKI → Checkpoint HF → Policy-Server ↔ Sim-Eval / Roboter); Plattform G1 + DEX3, GR00T N1.6 / N1.7 in je einem Absatz | 3 S. | Kap. 2 + 3 |
| 3 Schnittstellen | die Tabellen aus A1, auf das Nötige verdichtet | 5 S. | A1 |
| 4 Anleitung zum Weiterarbeiten | Zugänge und Tokens; Training lokal und auf KISSKI; Checkpoint hochladen; Gate und Closed Loop messen; die fünf häufigsten Fehler | 3–4 S. | Quickstart, `training/anleitung.md`, `kisski-hpc.md`, `sim-eval-anleitung.md`, `fehlerbehebung.md` |
| 5 Ergebnisse | als Kette, je Schritt eine Tabelle oder Abbildung: Training und Checkpoint-Wahl → Open Loop auf echten Bildern → Blocker 1 Registrierung → Blocker 2 Domain-Gap → Co-Training (Gate, Spanne, Griffe) → N1.7 | 6 S. | Kap. 4, Chronik Läufe 3/4, 32–34, 54–97, CSVs |
| 6 Grenzen | Stichprobe (n = 20/60, KI), Sim ≠ Roboter, Grenzen der Metriken (`lifted_clear`, `gap`), Hänger | 1 S. | Kap. 5, gekürzt |
| 7 Projektverlauf und Lessons Learned | Lehren mit Beleg (Lauf-Nr.), Arbeitsmittel, KI-gestützte Entwicklung in einem Absatz | 1–2 S. | Kap. 7, nach Prüfregel gekürzt (E2) |
| 8 Übergabe | priorisierte offene Punkte aus [`next-steps.md` §2](../next-steps.md), bekannte Defekte | 1 S. | Kap. 6 + 8 |
| Anhang | Versionen, Hyperparameter, Laufliste (Nr., Datum, Modell, Ergebnis) als Tabelle, SLURM-Auszug | — | heutiger Anhang, Chronik-Index |

Wegfallen oder auf einen halben Absatz schrumpfen: Grundlagen-Kapitel und „Weiterführende
Arbeiten“ als eigenes Kapitel. „Projektverlauf und Lessons Learned“ bleibt (E2), gekürzt.

### B2. Abbildungen statt Text (6–8 Stück)

| Abbildung | Daten |
|---|---|
| Architektur (TikZ oder Vektorgrafik) | neu zeichnen |
| Lernkurven Lauf 3 / Lauf 4 | [`docs/ergebnisse/wandb-run-charts.html`](ergebnisse/wandb-run-charts.html) bzw. W&B |
| Echtbild gegen Sim-Bild, gleiche Kamera | `Simulation/camera_reference/dataset_cam_left_high.png`, `latex/figures/sim/sim_scene70_cam_left_high.png` |
| Visuelles Gate als Balken (Echtbild, 5 cm, 4 cm; Maßstab gegen Co-Training) | `latex/figures/daten/gate_mse_haende.csv` |
| Closed-Loop-Leiter `reached` → `stacked` für 82, 88+97, 89, 93 | `latex/figures/daten/closed_loop.csv` |
| Fingerspanne 10 × 40 s (Lauf 96) | `latex/figures/daten/closed_loop.csv` (Zeilen 96) |
| Erster Griff der Politik | `latex/figures/sim/griff_lauf88_ep05_80s_ausschnitt.png` |
| optional: Orakel-Versatz-Sweep (Lauf 62) | Zahlen in der Chronik |

### B3. Formales

- `preamble/settings.tex`: Matrikelnummern und Abgabedatum (offen seit 2026-10-11).
- Zusammenfassung DE/EN neu (`frontmatter/zusammenfassung.tex`), passend zur Kernaussage oben.
- Literatur (`references.bib`) prüfen: jeder Eintrag zitiert, jede Zitierung vorhanden.
- `latex/README.md`: Stand und Seitenzahl aktualisieren.

## Prüfen vor der Abgabe

1. `make` in `latex/` baut ohne Fehler und ohne undefinierte Referenzen; Overfull-Boxen geprüft.
2. **Zahlen-Stichprobe:** je Kapitel drei Zahlen gegen Chronik oder CSV prüfen.
3. **Fremdtest:** Ein Gruppenmitglied, das Kap. 4 / Quickstart nicht geschrieben hat, misst damit
   Gate und Closed Loop für einen HF-Checkpoint (auf dem IKR-Server). Was hakt, wird korrigiert.
4. **Absatz-Test:** jeden Absatz gegen die Prüfregel oben lesen; was nichts trägt, fällt.
5. Repo: alles gepusht, Tag `abgabe-<JJJJ-MM-TT>` auf `training-luca-IKR-IS6.0`. Sichtbarkeit
   der HF-Repos klären: Die Render-Datensätze `luca-mue/g1-dex3-rendered-v3-cube{5,4}` sind privat.

## Reihenfolge

| Phase | Inhalt | Aufwand | Ort |
|---|---|---|---|
| 0 | Entscheidungen E1–E4 | — | Gruppe |
| 1 | A1 Schnittstellen, A2 Einstieg | ~1 Tag | Laptop (Code lesen genügt) |
| 2 | A3 Inventur, A4 Korrektheit | ~½ Tag | Laptop; Befehlsprüfung der Sim-Aktionen auf dem IKR-Server |
| 3 | B1/B2 LaTeX neu, Abbildungen | ~2 Tage | Laptop (latexmk) |
| 4 | Prüfen, Fremdtest, Korrektur | ~½ Tag | Gruppe, IKR-Server |
| 5 | B3 Formales, Tag, Abgabe | ~2 h | Laptop |

## Entscheidungen (getroffen 2026-10-29)

| | Frage | Entscheidung |
|---|---|---|
| E1 | Abgabedatum und Vorgaben der Hochschule? | Keine Vorgaben der Hochschule, kein festes Datum: abgegeben wird, wenn die Arbeit fertig ist. Ziel bleibt 20–25 Seiten Hauptteil. |
| E2 | Kapitel „Projektverlauf / Lessons Learned / KI-gestützte Entwicklung“? | **Behalten** als eigenes, nach der Prüfregel gekürztes Kapitel: Der Betreuer hat Lessons Learned ausdrücklich als wichtig genannt. |
| E3 | Wer übernimmt welchen Teil? | Teil A und B werden in einem Zug umgesetzt; der Fremdtest (Prüfen, Punkt 3) bleibt Aufgabe der Gruppe auf dem IKR-Server. |
| E4 | Schnittstellen und Anleitung in LaTeX? | Knapp und vollständig in LaTeX, Details per Verweis auf `docs/`. |
