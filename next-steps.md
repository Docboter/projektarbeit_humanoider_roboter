# Nächste Schritte

> **TL;DR:** Offene-Punkte-Liste des Projekts: aktueller Stand der Messkette, priorisierte
> nächste Schritte und offene Widersprüche zwischen den Ergebnisdokumenten. Bezugspunkt für
> die Prioritätensetzung, kein Ergebnisbericht selbst.

**Stand:** 2026-10-14 (Kurzstatus §0; Fassung davor 2026-10-13, 2026-10-11, 2026-08-14) · Grundlage: Ergebnis-Dokumente unter [`docs/ergebnisse/`](docs/ergebnisse/README.md),
[RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md), Trainings-Launcher unter
[`Training/scripts/`](Training/scripts/).

---

## 0. Projektstatus in Kürze (Stand 2026-10-14)

Diese Zusammenfassung ist die **eine gepflegte Statusquelle** des Projekts; CLAUDE.md und README
verweisen hierher. Den N1.7-Detailstand pflegt
[groot-n17-migration.md, „Stand der Umsetzung“](docs/weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19).

- **Bester validierter N1.6-Checkpoint:** vision-v2 (Lauf 3), **Step 44.000**. Die Nachmessung
  auf 20 vollen Test-Episoden (2026-10-12) widerlegt die frühere U-Kurve (Optimum 30.000):
  Validierungs-MSE fällt monoton bis 44.000
  ([lauf4-n17-vergleich.md §5](docs/ergebnisse/lauf4-n17-vergleich.md#5-nachgeschärfter-vergleich-20-episoden--750-steps-volle-länge)).
- **GR00T N1.7** als paralleler Pfad (`GROOT_VERSION`): voller KISSKI-Lauf (44k Steps, 7 h 53 min)
  fertig, Open Loop am Ende gleichauf mit N1.6, lernt früher, trainiert 1,6× schneller
  ([Lauf 4](docs/ergebnisse/lauf4-n17-vergleich.md)). In der Sim (Läufe 54/55, korrigierte
  Szene): `lifted` 0/10, Fingerspanne 26 % der Demo bei ckpt44000. **Offen:** Sim-Eval nach dem
  Abnahmeprotokoll mit 20 Episoden.
- **Closed Loop insgesamt:** 0 Erfolge (N1.6 Lauf 34: `lifted` 0/10, noch alte Geometrie; N1.7
  Läufe 54/55: 0/10). Die Sim-Geometrie ist seit 2026-08-24 korrigiert; die N1.6-Neumessung mit
  korrigierter Szene steht aus.
- **Neu 2026-10-14 — die Sim ist (noch) der Blocker, nicht das Modell:** Das Orakel-Replay
  (aufgezeichnete Demo-Aktionen statt Policy, Held-out-Layout, 30 Hz) hebt selbst kaum einen
  Würfel: `lifted_clear` 1/16 (KI 0,2–30 %), `stacked` 0/16. Die Hand erreicht und schiebt den
  Würfel jedes Mal, greift aber wenige Zentimeter daneben; Würfelgröße ausgeschlossen, Griff-
  stelle der echten Hand liegt 1–7 cm neben der Bild-Würfellage, seitlich spiegelbildlich je Hand
  ([Läufe 57–61](docs/ergebnisse/diagnose-chronik.md#läufe-5761-orakel-replay-würfelgröße-griffversatz-span-n16)).
  Bevor die Registrierung Würfel↔Hand stimmt, sind Closed-Loop-Modellzahlen gegen eine Schranke
  von ~0 gemessen. `span` N1.6 vision-v2 @30000 auf Test-Episoden: 90 % (N1.7 @44000: 94 %).
  Daumen-Gelenkgrenzen geprüft: kein Klemmen (Lauf 56).
- **Ursache des Closed-Loop-Versagens nicht identifiziert** (Stand 2026-10-13; teilweise
  überholt durch die Orakel-Befunde oben — mindestens die Würfellage relativ zur Hand ist ein
  Blocker). „Domain-Gap bestätigt“ ist ein
  starker Hinweis, kein Befund — Closed und Open Loop unterscheiden sich in mehreren Faktoren
  ([Review 10-13 §2.1](docs/projekt-review-2026-10-13.md#21-befunde)); billige trennende
  Experimente: [Review 10-13 §2.3](docs/projekt-review-2026-10-13.md#23-empfohlene-experimente-nach-kostennutzen).
- **Co-Training, Cosmos-Augmentierung, RL:** Werkzeuge gebaut, **kein Ergebnis**.
  `cotrain_synth_v1` (2026-09-07) war kein Co-Training-Lauf. Cosmos ist nicht end-to-end geprüft
  (kein ~65-GB-Docker-Server verfügbar). RL ist N1.6-only und erst sinnvoll ab `lifted` > 0.

## 1. Wo das Projekt steht

Die **Messkette ist geschlossen**. Der Reihe nach:

| Befund | Beleg |
|---|---|
| Die Sim-Harness ist nicht kaputt | [sim-bewertung.md](docs/ergebnisse/sim-bewertung.md) |
| Die GR00T-Inferenz-Pipeline ist extern bestätigt (47,7 % vs. 47,8 % erwartet) | [basismodell-referenz-eval.md](docs/ergebnisse/basismodell-referenz-eval.md) |
| Die Politik versucht in der Sim gar keinen Griff (0,39 rad gegen 2,09 rad = **19 %**) | Lauf 30, [diagnose-chronik.md](docs/ergebnisse/diagnose-chronik.md) |
| Auf **echten** Datensatz-Bildern kommandiert dieselbe Politik die volle Greifbewegung (Verhältnis **1,00**) | Lauf 32, `span`-Gate |
| ~~Die Checkpoint-Auswahl war bisher nicht nur unbelegt, sondern **falsch**: der letzte Checkpoint ist 25 % schlechter als der beste~~ **Widerlegt 2026-10-12:** mit 20 Episoden à 750 Steps ist der letzte Checkpoint (44.000) der beste, kein Overfitting. Die Checkpoint-Auswahl per Sweep bleibt trotzdem Pflicht | [lauf3-vision-split-auswertung.md](docs/ergebnisse/lauf3-vision-split-auswertung.md), [lauf4-n17-vergleich.md](docs/ergebnisse/lauf4-n17-vergleich.md) §5 |
| `TUNE_VISUAL` hebt die Sim-Fingerspanne von 20,5 % auf **27,6 %** (vollständige Trennung, p = 3,3 · 10⁻⁴) — und reicht trotzdem nicht: `lifted` 0/10 | Lauf 34, [diagnose-chronik.md](docs/ergebnisse/diagnose-chronik.md) |

Daraus folgt nach der vorregistrierten Regel: **Ursache ist wahrscheinlich der Domain-Gap**, nicht
„Griff nie gelernt" — als *starker Hinweis*, nicht als Beweis (siehe Vorbehalte unten). Die Kopfzahl des
Projekts bleibt **0 Erfolge im Closed-Loop** (0/20 früher, Lauf 34: 0/10).

### Was seit dem 2026-08-14 passiert ist (Stand 2026-10-11)

| Datum | Befund | Beleg |
|---|---|---|
| 08-17 bis 08-25 | **Würfellage aus den Realbildern** (Farbblob → Strahl → Würfelebene) ersetzt die Greifpunkt-Rekonstruktion; Würfelebene lag 2 cm zu hoch (korrigiert); **Gierwinkel** der Würfel gemessen (gleichverteilt 0–45°; Umstellung auf zufälligen Gierwinkel offen) | Läufe 35–37, 52, 53 in [diagnose-chronik.md](docs/ergebnisse/diagnose-chronik.md); [wuerfellage-rekonstruktion.md](docs/simulation/wuerfellage-rekonstruktion.md) |
| 08-24 | **Geometriefehler in der Sim gefunden und behoben:** der Vorzeichen-Flip der vier proximalen Fingergelenke war *falsch* (entfernt) — die Hand konnte in **keinem früheren Lauf** ganz schließen; Beckenhöhe 0,764 m; Basis x −0,057 m | Läufe 38–51 in [diagnose-chronik.md](docs/ergebnisse/diagnose-chronik.md) |
| 09-07 / 09-23 | Externer Sim-Datensatz `Cube_Stacking_synth` auf das echte Schema harmonisiert; v21 hatte vertauschte Fingerbefehle → v22 und `Cube_Stacking_synth_jointspace`; Presets in `Training/kisski_chain.sh` | [synth-datensatz.md](docs/training/synth-datensatz.md) |
| 09-07 | Lauf `cotrain_synth_v1` **abgebrochen** nach 138 s (W&B `7z81177r`) — laut W&B kein Co-Training (ein Datensatz, `tune_visual: false`, IKR-Server; Nachtrag 2026-10-13 in der Diagnose-Chronik) | [co-training.md](docs/training/co-training.md) |
| 10-11 | LaTeX-Projektarbeit überarbeitet (`latex/`, 28 Seiten Hauptteil); GR00T-N1.7-Migration lief auf separatem Branch `training-luca-IKR-IS6.0-GN1.7` (seit 10-13 gemergt) | — |
| 10-12 / 10-13 | N1.7: voller KISSKI-Lauf (Lauf 4, 44k) und Sim-Läufe 54/55 (`lifted` 0/10); Lauf-3-Nachmessung 20 × 750 widerlegt die U-Kurve, bester Checkpoint 44.000 | [lauf4-n17-vergleich.md](docs/ergebnisse/lauf4-n17-vergleich.md), [Chronik 54/55](docs/ergebnisse/diagnose-chronik.md#läufe-5455-die-ersten-n17-checkpoints-in-der-sim) |

**Konsequenzen und Vorbehalte, die jede Zahl dieses Dokuments betreffen:**

1. **Alle Sim-Greifzahlen vor dem 2026-08-24 stehen unter Vorbehalt** (Läufe 29–34, inkl. 27,6 %
   Fingerspanne und `lifted` 0/10): die Hand konnte nicht voll schließen, der Roboter stand ~12 cm
   versetzt. Mit der korrigierten Szene gemessen sind bisher nur die N1.7-Läufe 54/55
   (2026-10-12/13); für N1.6 steht die Neumessung aus.
2. **Der Renderdatensatz für das Co-Training muss neu erzeugt werden** (er entstand mit der
   fehlerhaften Geometrie).
3. **Das `span`-Gate (Median 1,00) nutzte die Trainingsepisoden 0–4**, und 2,0944 rad ist zugleich das
   De-Normalisierungs-Maximum. Auswendiglernen ist damit nicht ausgeschlossen; „Domain-Gap bestätigt /
   (a2) ausgeschlossen" ist ein starker Hinweis, kein Beweis. Für N1.7 gibt es die Messung auf
   zurückgehaltenen Episoden inzwischen (Test 240/255/270/285/300: 90 % bzw. 94 %,
   [Läufe 54/55](docs/ergebnisse/diagnose-chronik.md#läufe-5455-die-ersten-n17-checkpoints-in-der-sim));
   für N1.6 steht sie aus.
4. **Trainingsparameter der Läufe 2/3 (W&B):** Peak-LR 1e-4, Warmup 0,10, 4 GPUs × Batch 8.

> **Ergänzung 2026-08-14 (abends):** Die letzte Zeile ist das Ergebnis des `span`-Gates aus
> Schritt 3 — und sie ist zweischneidig. Der Vision-Pfad **wirkt nachweislich**: drei von zehn
> Episoden kommandieren 60–80 % der Demonstration, ein Verhalten, das der alte Checkpoint über
> sieben Episoden nie gezeigt hat, und die Anhebung läuft mit der Fingerspanne mit
> (Spearman +0,62). Er **genügt aber nicht**: 27,6 % stehen gegen 100 % auf echten Bildern, und
> kein Würfel überschreitet die 2-cm-Schwelle. Ein weiterer Lauf derselben Art würde denselben
> Faktor-3,6-Rest übriglassen.

> **Ergänzung 2026-08-14:** Der letzte Punkt der Tabelle ist neu und relativiert die früheren
> Sim-Ergebnisse *methodisch*, nicht inhaltlich. Lauf 1 und Lauf 2 wurden mit dem jeweils
> **letzten** Checkpoint evaluiert, weil es keine Alternative gab. Lauf 3 zeigt erstmals, dass
> das systematisch der falsche Griff ist. Die 0/20 sind damit nicht erklärt — der Domain-Gap-
> Befund steht unverändert — aber jede künftige Sim-Zahl gehört an den **validierten**
> Checkpoint, nicht an den letzten.

> Alles, was seit Lauf 32 gebaut wurde — Livestream (Spur A), Latenz-Tooling, Render-Knöpfe —
> ist **Instrumentierung**. Es hat diese Zahl nicht bewegt und sollte in der Projektarbeit auch
> nicht als Fortschritt an ihr dargestellt werden.

---

## 2. Nächste Schritte — in dieser Reihenfolge (Stand 2026-10-14)

> **Vorgezogen (2026-10-14), vor allen Punkten der Tabelle unten:**
> 1. **Registrierung Würfel↔Hand klären** — Versatz-Sweep im Orakel (Layout global je Hand um
>    dx/dy verschieben, Lift-Quote je Raster) und Kinematik-Ursache suchen (Taillen-/Hüftgelenke,
>    Schulterbreite, Basis-Gierwinkel). Ziel: Orakel `lifted_clear` deutlich > 0.
> 2. Orakel-Hänger (Datensatz-Ep. 256, Halte-Phase) nachstellen — eine Eval darf nicht hängen.
> 3. Erst danach die Closed-Loop-Baselines N1.6 @44000 / N1.7 @44000 (20 Ep., 120 s, Held-out-Layout).
>    N1.6 @44000 liegt nur auf KISSKI → nach HF hochladen oder herkopieren.
>
> Der `span`-Schritt (Punkt 2 unten) ist für N1.6 auf zurückgehaltenen Episoden erledigt
> (Lauf 61, @30000: 90 %); @44000 steht aus.

**Tabelle Stand 2026-10-11** (Punkt 3 überholt durch den Orakel-Befund: die Closed-Loop-Neumessung
ist erst nach der Registrierung aussagekräftig):

| # | Schritt | Warum / Ausgang |
|---|---|---|
| 1 | **Open-Loop-MSE auf allen 61 Testepisoden**, volle Länge, getrennt nach Armen und Händen, mit trivialer Baseline (z. B. konstante Pose / letzter Zustand) | Der Sweep aus Lauf 3 nutzte nur 6 Episoden × 300 Steps; ohne Baseline ist die MSE nicht einzuordnen. Billig (`checkpoint_sweep.py`, `EVAL_NUM_TRAJ`/`EVAL_STEPS`) |
| 2 | **`span`-Gate auf zurückgehaltenen Testepisoden** (`SPAN_TRAJ_IDS` mit Test-Indices) — für N1.7 erledigt (Läufe 54/55: Test 90 % / 94 %), für N1.6 offen | Schließt die Auswendiglern-Lücke des Lauf-32-Gates; trägt „Domain-Gap bestätigt“ vom Hinweis zum Befund |
| 3 | **Closed-Loop-Neumessung mit der korrigierten Szene** (identisches Protokoll wie Lauf 34: 10 Episoden × 40 s, Checkpoint vision-v2-**44000** — der seit der Nachmessung 2026-10-12 validierte; bei Lauf 34 war es noch 30000), optional **Tauschtest** (echtes Bild + Sim-Zustand vs. Sim-Bild + echter Zustand) | Erst damit sind die Sim-Zahlen wieder belastbar; entscheidet, wie groß der Domain-Gap nach der Geometriekorrektur noch ist |
| 4 | **Renderdatensatz neu erzeugen → Co-Training-Lauf** (eigener Render und/oder Synth v22 / `Cube_Stacking_synth_jointspace`); Log des Abbruchs von `cotrain_synth_v1` aus dem Trainingscontainer des IKR-Servers holen; auf KISSKI zuerst `kisski_chain.sh smoke …` | Vorregistrierte Regeln in [co-training.md](docs/training/co-training.md) / [synth-datensatz.md](docs/training/synth-datensatz.md) |
| 5 | **RL (FPO)** erst, wenn die Politik gelegentlich einen Würfel hebt | Ohne Startsignal fehlt dem Reward der Gradient ([RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md)) |
| 6 | **Echter Roboter** | setzt belastbare Sim- bzw. Open-Loop-Zahlen voraus |

Offene Entscheidung nebenbei: Umstellung der Sim auf zufälligen Würfel-Gierwinkel (gemessen: gleichverteilt
0–45°).

**Erledigt:** Der frühere Widerspruch zwischen RL-Plan und
[`lauf2-vision-auswertung.md`](docs/ergebnisse/lauf2-vision-auswertung.md) §6.1 (`TUNE_VISUAL` ja/nein) ist
aufgelöst — Lauf 2 hatte keinen Color-Jitter (`USE_AUGMENTATION` kam erst 2026-06-12, `c3a1cd6`), Lauf 3
erfüllt die Bedingung aus §6.2. Vorbehalt bleibt: Color-Jitter variiert weder Geometrie noch Textur noch
Rendering-Stil; `cam_left_wrist` liegt in der [Domain-Gap-Messung](docs/ergebnisse/domain-gap-analyse.md)
bei 0,36.

---

## 3. Bisherige Schritte (Verlauf, Stand der Umsetzung)

### Schritt 1 — Checkpoint-Auswahl falsifizierbar machen ✅ *umgesetzt 2026-08-13*

**Die ursprüngliche Fassung dieses Schritts lautete „`enable_open_loop_eval` im Launcher setzen".
Das war falsch — der Schalter ist tot.** Der Befund im Fork:

| Fundstelle | Befund |
|---|---|
| `training_config.py:100–109` | `enable_open_loop_eval`, `open_loop_eval_traj_ids`, `open_loop_eval_steps_per_traj`, `open_loop_eval_plot_indices` sind deklariert — und werden **nirgends gelesen**. |
| `data/dataset/factory.py:26` | `assert self.config.training.eval_strategy == "no"` — ein `eval_strategy="steps"` **stürzt ab**, es evaluiert nicht. |
| `data/dataset/factory.py:59, 79` | Der Loader bekommt fest `split="train"`, und `build()` gibt `eval_dataset=None` zurück. |

In-Training-Validierung ist in diesem Fork also **strukturell nicht vorhanden**, nicht bloß
abgeschaltet. Sie nachzurüsten hieße: Assertion entfernen, zweiten Datensatz mit `split="test"`
bauen, Felder durch `FinetuneConfig` + `launch_finetune.py` durchreichen — also Submodul ändern,
Fork pushen, Dockerfile-Pin nachziehen, Image neu bauen. Der `Gr00tTrainer`-Eval-Pfad ist dabei
ungetestet (`batch_eval_metrics=True` ohne `compute_metrics`).

**Stattdessen umgesetzt — Validierung nach dem Lauf, ohne Submodul-Eingriff:**

| Datei | Zweck |
|---|---|
| [`Training/scripts/checkpoint_sweep.py`](Training/scripts/checkpoint_sweep.py) | Lädt **jeden** `checkpoint-*` und misst MSE/MAE auf den zurückgehaltenen Episoden. Ergebnis: Tabelle über die Steps, `checkpoint_sweep.json`, bester Checkpoint. Die Metrik selbst kommt unverändert aus dem Fork (`evaluate_single_trajectory`). |
| [`Training/scripts/lib_split.sh`](Training/scripts/lib_split.sh) | Split-Logik, jetzt von **beiden** Trainings-Skripten geteilt; schreibt zusätzlich ein `split.json`-Protokoll. |
| [`Training/scripts/run_finetuning_vision.sh`](Training/scripts/run_finetuning_vision.sh) | Unterstützt jetzt `TRAIN_TEST_SPLIT` **und** `USE_AUGMENTATION` — beides fehlte dort, obwohl die Doku sie auswies. Für Lauf 3 ist genau das der relevante Pfad. |
| [`Training/kisski_open_loop_eval.sh`](Training/kisski_open_loop_eval.sh) | Wertet jetzt alle Checkpoints auf `test` aus statt einen einzelnen auf Trainingsdaten. |

Zwei Fallen, die dabei aufgefallen sind und im Sweep abgefangen werden:

1. **`gr00t/eval/open_loop_eval.py` reicht kein `split` durch.** Der Loader-Default ist
   `split="train"` — nach einem Split-Lauf misst das Skript also **Trainings**-MSE und kann
   Overfitting prinzipiell nicht zeigen. Genau so lief `kisski_open_loop_eval.sh` bisher.
2. **`loader[idx]` indiziert in die gefilterte Liste.** Position 0 ist nach einem 80/20-Split die
   Episode 240, nicht die Episode 0. Der Sweep protokolliert immer beides.

Nebenbei behoben: im Reset-Pfad von `run_finetuning.sh` wurde ein nirgends definiertes `warn`
aufgerufen — unter `set -e` hätte das den Lauf mit Exit 127 abgebrochen, sobald nach einem
Split-Lauf einer ohne Split folgt.

### Schritt 2 — Lauf 3 starten ✅ *umgesetzt 2026-08-13/14*

**Gelaufen:** Run `tp1nc699` / `g1_dex3_blockstacking_vision_v2`, 44.000 Steps auf 4 GPUs,
12 h 27 min, `TUNE_VISUAL=1` + `USE_AUGMENTATION=1` + `TRAIN_TEST_SPLIT=1`. Vollständige
Auswertung: [`lauf3-vision-split-auswertung.md`](docs/ergebnisse/lauf3-vision-split-auswertung.md).

Das Wesentliche:

| Befund | Zahl |
|---|---|
| Bester Checkpoint (Validierungs-MSE auf zurückgehaltenen Episoden) | **Step 30.000** — 0,00716 *(6×300-Protokoll; überholt, siehe Nachtrag: 44.000)* |
| Letzter Checkpoint (Step 44.000) | 0,00898 → **25 % schlechter** |
| Trainingsloss im selben Fenster 30k → 44k | 0,0131 → 0,0073 (**−44 %**) |

> **Nachtrag 2026-10-12 — dieser Abschnitt ist überholt.** Die Nachschärfung aus Vorbehalt 1
> unten (`EVAL_NUM_TRAJ=20 EVAL_STEPS=750`, Job 16633477) zeigt **keine** U-Kurve: Validierungs-MSE
> 30.000 = 0,00519, 44.000 = 0,00482 (bester; gepaart t = 2,15, n = 20). Die Schere unten war ein
> Artefakt der 6×300-Auswertung (nur Anfahrphase). Auch der Nebenbefund „`MAX_STEPS=30000`
> genügt“ gilt nicht mehr. Zahlen: [lauf4-n17-vergleich.md](docs/ergebnisse/lauf4-n17-vergleich.md) §5.

Die Schere zwischen fallendem Trainingsloss und steigender Validierungs-MSE ist der erste
gemessene Overfitting-Nachweis im Projekt. Schritt 1 hat also unmittelbar getragen: **ohne den
Sweep wäre wieder der letzte Checkpoint ins Gate gegangen.**

Zwei Vorbehalte, die vor Schritt 3 stehen:

1. **Signifikanz.** Der Sweep lief mit den Defaults (6 Episoden × 300 Steps). Gegen Step
   20.000/25.000/35.000 ist Step 30.000 statistisch **nicht** unterscheidbar (t ≤ 1,0), und
   selbst 30.000 vs. 44.000 ist mit t = 1,45 nur ein Trend. Belastbar ist allein: *ab ~20.000
   auskonvergiert, der letzte Checkpoint ist nicht der beste.* Nachschärfen mit
   `EVAL_NUM_TRAJ=20 EVAL_STEPS=750` kostet einen Bruchteil des Trainings.
2. **Split-Verifikation.** Dass der Split beim **Training** aktiv war (und nicht erst beim
   Sweep), ist noch nicht belegt — `split.json` im Lauf-Verzeichnis und das SLURM-Log prüfen
   ([§6 der Auswertung](docs/ergebnisse/lauf3-vision-split-auswertung.md#6-noch-zu-verifizieren-war-der-split-beim-training-aktiv)).
   Trifft es nicht zu, wären die Testepisoden mittrainiert und der Sweep wertlos.

~~Nebenbefund für den nächsten Lauf: **`MAX_STEPS=30000` genügt** — die letzten 14.000 Steps
(≈ 4 h auf 4 GPUs) haben die Validierung verschlechtert.~~ *Überholt (2026-10-12), siehe Nachtrag
oben: bis 44.000 fällt die Validierungs-MSE weiter.*

<details>
<summary>Ursprüngliche Startanweisung (zur Reproduktion)</summary>

Der eine Lauf, den die Beweiskette begründet:

```bash
export HF_TOKEN=hf_... WANDB_API_KEY=...
export TUNE_VISUAL=1          # → routet auf run_finetuning_vision.sh, eigener
                              #   blockstacking_vision-Namespace, LLM bleibt eingefroren
export USE_AUGMENTATION=1     # Default, hier bewusst explizit — das ist der Unterschied zu Lauf 2
export TRAIN_TEST_SPLIT=1     # 80/20, Testepisoden zurückgehalten
export GLOBAL_BATCH_SIZE=32
sbatch Training/kisski_submit.sh
```

Überwachung wie gewohnt: `squeue -u $USER`, `tail -f logs/slurm-<jobid>.out`.

Danach — **nicht** blind den letzten Checkpoint nehmen:

```bash
export RUN_DIR=/data/g1_dex3_finetune/blockstacking_vision
sbatch Training/kisski_open_loop_eval.sh
```

Das liefert die Validierungs-MSE je Checkpoint und benennt den besten. Erst dieser Checkpoint geht
in Schritt 3.

</details>

### Schritt 3 — ✅ erledigt (2026-08-14): das `span`-Gate ist gefahren

Die vorregistrierte Frage an Lauf 3 lautete **nicht** „stapelt er jetzt", sondern:

> Kommandiert die Politik in der **Sim** mehr als die heutigen 19 % Fingerspanne?

**Antwort: ja — 27,6 % statt 20,5 %, aber `lifted` bleibt 0/10.** Gemessen am 2026-08-14 mit
`checkpoint-30000` (dem validierten, nicht dem letzten), zehn Episoden à 40 s, also im identischen
Zeitfenster wie die Vergleichsmessung Lauf 31. Jede der zehn neuen Episoden liegt über jeder der
fünf alten — vollständige Trennung, exakter einseitiger Rangtest **p = 3,3 · 10⁻⁴**. Drei Episoden
erreichen 60–80 % der Demonstration; der alte Checkpoint blieb über sieben Episoden in einem engen
Band von 0,35–0,49 rad. Die Anhebung läuft mit der Fingerspanne mit (Spearman **+0,62**), die
Würfel-Verschiebung nicht (+0,05) — der Greifbefehl wirkt also mechanisch und ist kein Nebenprodukt
von mehr Kontakt.

Vollständige Auswertung inklusive Episodentabelle, Zuordnung und Vorbehalten:
[diagnose-chronik.md, Läufe 33/34](docs/ergebnisse/diagnose-chronik.md#läufe-3334-runs2026081403-runs2026081404-der-tune_visual-checkpoint-im-closed-loop).

**Damit ist die Regel angewandt.** Sie sagte: bewegt sich die Spanne nicht deutlich über 19 %, ist
der Vision-Pfad ausgereizt. Sie hat sich bewegt — der Pfad ist also **nicht** ausgereizt, aber
27,6 % gegen 100 % auf echten Bildern heißt auch: er schließt den Gap nicht. Rund 48 GPU-Stunden
haben ein Drittel des Weges gebracht. *(Nachtrag 2026-10-13: die beiden Zahlen haben verschiedene
Nenner — 27,6 % ist die Sim-Spanne relativ zur Konstante 2,094 rad, 100 % das Verhältnis
Vorhersage/Ground Truth auf echten Bildern, das an genau dieser De-Normalisierungsgrenze sättigt.
„Ein Drittel des Weges“ ist deshalb kein belastbares Maß, [Review 10-13 §2.2](docs/projekt-review-2026-10-13.md#22-zahlen--und-aussage-inkonsistenzen).)* Der Rest kommt nicht durch mehr vom Gleichen, sondern muss
auf der **Sim-Seite** angegangen werden → Schritt 4. *(Nachtrag 2026-10-11: die 27,6 % und `lifted` 0/10 stammen aus der Zeit vor der Geometriekorrektur vom 2026-08-24 und sind neu zu messen — siehe §1 und §2.)*

<details>
<summary>Wiederholen (Belege sind gitignored)</summary>

```bash
CKPT=/data/checkpoints/groot-g1dex3-vision-v2-30000
# Realbild-Referenz (Lauf 33: Median 1,00)
CHECKPOINT_PATH=$CKPT HF_TOKEN=hf_... ./Simulation/server_rl_run.sh span
# Sim-Spanne (Lauf 34: Median 0,577 rad = 27,6 %)
NUM_EPISODES=10 EPISODE_LENGTH_S=40 CHECKPOINT_PATH=$CKPT \
  HF_TOKEN=hf_... ./Simulation/server_rl_run.sh eval
```

`EPISODE_LENGTH_S` **immer** setzen und beim Vergleich konstant halten: ohne die Variable laufen
Episoden bis 9000 Steps (≈ 6 h für 20 Episoden), und die Fingerspanne ist ein Maximum über die
Episode — ein längeres Fenster hebt sie allein dadurch. Genau daran krankt Lauf 33 (60 s) als
Vergleich, weshalb Lauf 34 mit 40 s wiederholt wurde.

</details>

### Schritt 4 — Co-Training auf gerenderten Bildern (Werkzeuge gebaut, Lauf steht aus)

> **Stand 2026-10-11:** der Renderdatensatz ist wegen der Geometriekorrektur vom 2026-08-24 neu zu erzeugen,
> die Würfellage kommt inzwischen aus den Realbildern (`layout`), und mit `Cube_Stacking_synth` gibt es eine
> zweite Co-Training-Quelle ([synth-datensatz.md](docs/training/synth-datensatz.md)). Der erste Lauf
> (`cotrain_synth_v1`) scheiterte am 2026-09-07. Der Text unten ist der Stand vom 2026-08-14.

**Werkzeuge gebaut (2026-08-14), Lauf steht aus.** Vollständige Anleitung inklusive
Rauchtest, Transportweg und Erfolgsregel: [docs/training/co-training.md](docs/training/co-training.md).

Die Eskalation ist genau das, was [lauf2 §6.2](docs/ergebnisse/lauf2-vision-auswertung.md)
verlangt. Das Gate aus Schritt 3 hat die Voraussetzung geliefert, unter der sich der Aufwand
rechtfertigen lässt: der Encoder ist der wirksame Hebel (er hat die Spanne nachweislich bewegt),
er ist nur mit Realbildern allein nicht weit genug zu bringen. Und die +0,62-Kopplung zwischen
Fingerspanne und Anhebung sagt, woran sich ein Erfolg zeigen wird, bevor die Erfolgsrate
reagiert: die 2-cm-Schwelle liegt nur noch 0,32 cm über der heute besten Anhebung.

| Datei | Rolle |
|---|---|
| [`render_cotrain_dataset.py`](Simulation/g1_dex3_sim/render_cotrain_dataset.py) | Renderer: echte Aktionen abspielen, die vier Policy-Kameras aufzeichnen, LeRobot-v2.1-Datensatz schreiben |
| [`server_rl_run.sh render`](Simulation/server_rl_run.sh) | Wrapper auf dem Sim-Server (beide Stufen, fortsetzbar) |
| [`launch_cotrain.py`](Training/scripts/launch_cotrain.py) | Trainings-Einstieg für zwei Datensätze mit `mix_ratio` |
| [`run_finetuning_cotrain.sh`](Training/scripts/run_finetuning_cotrain.sh) | Trainings-Launcher, Namespace `blockstacking_cotrain`, `USE_COTRAIN=1` |

**Kein Submodul-Eingriff nötig.** Der Mischbetrieb steckt schon im Fork
(`SingleDatasetConfig.mix_ratio` → `ShardedMixtureDataset`); nur der CLI-Einstieg
`launch_finetune.py` verdrahtet einen einzigen Datensatz fest. `launch_cotrain.py` ist dessen
Kopie mit genau einer Abweichung und liegt in `Training/scripts/` — das wird auf KISSKI als
`/scripts` eingehängt, wirkt also ohne Fork-Push und ohne Image-Rebuild.

**Der Renderer hat zwei Stufen, und das ist kein Umweg.** Der Datensatz speichert keine
Objektposen. Würden die Aktionen abgespielt, während die Env ihre Würfel zufällig auslegt,
entstünden Bild-Aktions-Paare, in denen der Arm dorthin greift, wo kein Würfel liegt — der
Encoder lernte, den Würfel zu *ignorieren*. Stufe `scan` sucht deshalb erst (mit
Kameras auf 1/10 Auflösung, also billig) je Hand den Moment des Zugreifens und den
Fingerkuppen-Schwerpunkt dort; Stufe `render` legt die Würfel an genau diese x/y und zeichnet
in kalibrierter Auflösung auf.

**Die beiden Vorfragen sind beantwortet** (Herleitung in
[§4 der Anleitung](docs/training/co-training.md#4-die-beiden-entscheidungen--und-wie-sie-begründet-sind)):

1. **Wie viele Episoden? → 60 für den ersten Lauf** (~3 h Wanduhr für beide Stufen, ~56 000
   Frames). Kein Optimum, sondern das, was an einem Abend entsteht — und weil der Renderer
   fortsetzbar ist, keine Einbahnstraße: Aufstocken auf 120 rendert nur die neuen.
2. **Welches Verhältnis? → 0,25 gerendert.** Zwei unabhängige Argumente treffen sich dort.
   (a) `mix_ratio` ist eine Sampling-Wahrscheinlichkeit, kein Längenverhältnis: bei 56 k
   gerenderten gegen 224 k echten Frames sieht das Modell jedes gerenderte Bild sonst 4× so
   oft. Gleich häufig ist es bei 56/(56+224) ≈ 0,20. (b) Die gerenderten Episoden tragen
   **keine neuen Aktionen** — sie sind die Zwillinge von 60 der 240 Trainingsepisoden. Bei 0,5
   verbrächte das Modell die Hälfte aller Updates auf einem Viertel des Repertoires; das ist
   die Verengung aus Lauf 2 mit umgekehrtem Vorzeichen. Mehr Sim-Anteil kommt über **mehr
   Episoden**, nicht über ein höheres Verhältnis.

**Zurückgehaltene Episoden werden nie gerendert.** Der Renderer zieht die Grenze mit derselben
Formel wie [`lib_split.sh`](Training/scripts/lib_split.sh) und bricht bei explizit angegebenen
Test-Indices ab. Sonst sähe das Modell die Testepisoden in gerenderter Form und die
Validierungs-MSE — die einzige Zahl, die Overfitting sichtbar macht — wäre wertlos.

**Vorregistriert, bevor der Lauf startet:** primäres Gate ist die Sim-Fingerspanne im
identischen Protokoll wie Lauf 34 (10 Episoden × **40 s**, nicht verhandelbar) gegen die
0,577 rad von dort. Leitplanke ist die Validierungs-MSE auf echten Testepisoden gegen die
0,00716 aus Lauf 3: wird sie merklich schlechter, verdrängt das gerenderte Material echtes
Lernen → Verhältnis senken statt weiterzurendern. (Die 0,00716 gelten nur im 6×300-Protokoll;
im belastbareren 20×750-Protokoll ist die Lauf-3-Referenz 0,00482 bei Step 44.000 — immer mit
demselben Protokoll vergleichen.)

### Schritt 5 — RL bleibt hinten an

Steht so bereits im [RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md): der FPO-Pfad ist
gebaut und auf Hardware validiert, aber er wartet auf eine BC-Policy, die **überhaupt gelegentlich
Erfolg hat**. Einem Nullpunkt-Reward fehlt das Startsignal. Nicht vorziehen.

---

## 4. Nebenbei offen (Komfort, nicht Fortschritt)

| Punkt | Aufwand | Status |
|---|---|---|
| Hardware-Test der Live-Ansicht (Stand 2026-10-11 weiterhin offen; nicht belegt, ob inzwischen gelaufen): `./Simulation/server_rl_run.sh livecheck`, dann `… view` (Szene ohne Gewichte, braucht kein `HF_TOKEN`). Als Client wahlweise die native App oder `… webview` im Browser — letzteres ohne Installation | ~15 min | Gebaut 2026-08-13, `view`/`webview` 2026-08-17. Der Web-Viewer ist lokal end-to-end geprüft; der **WebRTC-Handshake lief nie auf Hardware**. Bleibt das Bild trotz offenem UDP schwarz: `RL_NETWORK_MODE=host` |
| `SCENE_CAM=0` / `RL_AA_MODE=Off` Wirkung auf Steps/s | ~15 min | Ungemessen (~15 % erwartet) |
| Doku-Abschnitt zur Latenz-Zerlegung | — | Erledigt: steht in [inferenz-optimierung.md](docs/simulation/inferenz-optimierung.md) |

Nicht dringend, weil keiner davon die 0/20 bewegt.

---

## 5. Was ausdrücklich *kein* nächster Schritt ist

- **Inferenz-Optimierung.** Die Zerlegung `t(n) = 35,6 ms + n × 9,6 ms` steht in
  [inferenz-optimierung.md](docs/simulation/inferenz-optimierung.md), aber bei 74 ms gegen 267 ms Budget
  (28 % Auslastung) ist heute nichts zu tun. Das ist ein Planungswerkzeug für die Portierung auf
  Zielhardware, keine offene Baustelle.
- **`TUNE_VISUAL=1` ohne Augmentierung.** Das wäre die Wiederholung von Lauf 2 mit dem bereits
  gemessenen Ergebnis (Politik-Kollaps).
- **Qualitätsmessung der Denoising-Schritte über `open_loop_eval.py --denoising-steps`.** Die
  Option ist im Fork **tot** (deklariert, nirgends angewandt) — sie vergleicht zwei identische
  Läufe. Details und der korrekte Weg stehen in
  [inferenz-optimierung.md](docs/simulation/inferenz-optimierung.md).
