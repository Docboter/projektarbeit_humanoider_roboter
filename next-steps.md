# Nächste Schritte

**Stand:** 2026-08-13 · Grundlage: Ergebnis-Dokumente unter [`docs/ergebnisse/`](docs/ergebnisse/README.md),
[RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md), Trainings-Launcher unter
[`Training/scripts/`](Training/scripts/).

---

## 1. Wo das Projekt steht

Die **Messkette ist geschlossen**. Der Reihe nach:

| Befund | Beleg |
|---|---|
| Die Sim-Harness ist nicht kaputt | [sim-bewertung.md](docs/ergebnisse/sim-bewertung.md) |
| Die GR00T-Inferenz-Pipeline ist extern bestätigt (47,7 % vs. 47,8 % erwartet) | [basismodell-referenz-eval.md](docs/ergebnisse/basismodell-referenz-eval.md) |
| Die Politik versucht in der Sim gar keinen Griff (0,39 rad gegen 2,09 rad = **19 %**) | Lauf 30, [rl-anleitung.md](docs/weiterfuehrend/rl-anleitung.md) |
| Auf **echten** Datensatz-Bildern kommandiert dieselbe Politik die volle Greifbewegung (Verhältnis **1,00**) | Lauf 32, `span`-Gate |

Daraus folgt nach der vorregistrierten Regel: **Ursache ist der Domain-Gap**, nicht „Griff nie
gelernt". Die Kopfzahl des Projekts bleibt damit **0/20 im Closed-Loop**.

> Alles, was seit Lauf 32 gebaut wurde — Livestream (Spur A), Latenz-Tooling, Render-Knöpfe —
> ist **Instrumentierung**. Es hat diese Zahl nicht bewegt und sollte in der Projektarbeit auch
> nicht als Fortschritt an ihr dargestellt werden.

---

## 2. Ein Widerspruch in den Docs, vor dem nächsten Lauf zu klären

Der [RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md) nennt `TUNE_VISUAL=1` als
begründeten nächsten Lauf.
[`lauf2-vision-auswertung.md`](docs/ergebnisse/lauf2-vision-auswertung.md) §6.1 sagt wörtlich das
Gegenteil: *„`tune_visual = true` auf reinen Realdaten nicht weiterverfolgen"* — Lauf 2 kollabierte
damit auf reines Arm-Zurückziehen. Sinnvoll sei es **nur**, wenn der Encoder sim-ähnliche oder
DR-variierte Bilder sieht (§6.2).

**Auflösung:** Diese Bedingung ist inzwischen erfüllt, und zwar erst nachträglich.

| | Datum |
|---|---|
| Lauf 2 (`ajgoskon`, `tune_visual=true`) | 2026-06-04/05 |
| Einführung von `USE_AUGMENTATION` (Color-Jitter) | 2026-06-12 (`c3a1cd6`) |

Lauf 2 hatte also **gar keinen Color-Jitter**. Ein neuer `TUNE_VISUAL`-Lauf ist deshalb keine
Wiederholung — er ist der erste, der die von §6.2 geforderte Voraussetzung überhaupt erfüllt.

**Vorbehalt, der im Ergebnisprotokoll stehen muss:** Color-Jitter variiert Helligkeit, Kontrast,
Sättigung und Farbton — **nicht** Geometrie, Textur oder Rendering-Stil. Der Ausreißer der
[Domain-Gap-Messung](docs/ergebnisse/domain-gap-analyse.md) ist `cam_left_wrist` bei **0,36**, eine
Nahbereichskamera. Dass Farb-Jitter allein diesen Gap schließt, ist nicht gesagt. Der Lauf ist ein
begründeter, billiger Test — keine Erfolgsgarantie.

---

## 3. Priorisierte Schritte

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

### Schritt 2 — Lauf 3 starten

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

### Schritt 3 — Das Gate ist der `span`-Test, nicht 20 Closed-Loop-Episoden

Die Frage an Lauf 3 lautet **nicht** „stapelt er jetzt", sondern:

> Kommandiert die Politik in der **Sim** mehr als die heutigen 19 % Fingerspanne?

Das ist billig, eindeutig und vorregistrierbar — dieselbe Metrik, die Lauf 30 und 32 entschieden
hat. Bewegt sie sich nicht deutlich über 19 %, ist der Vision-Pfad ausgereizt und die ehrliche
Schlussfolgerung lautet: der Gap muss auf der **Sim-Seite** geschlossen werden.

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh span     # Realbild-Referenz (heute 1,00)
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh eval     # Sim-Spanne (heute 0,19)
```

### Schritt 4 — Falls das Gate fällt: Co-Training auf gerenderten Bildern

Die Eskalation ist genau das, was [lauf2 §6.2](docs/ergebnisse/lauf2-vision-auswertung.md)
verlangt. Die Infrastruktur existiert **halb**:
[`run_g1_dex3_replay.py`](Simulation/g1_dex3_sim/run_g1_dex3_replay.py) fährt die Sim mit
Ground-Truth-Aktionen des Datensatzes und rendert dabei die Policy-Kameras. Über mehr als die eine
gebündelte Episode (`replay_episode0.npz`) ausgerollt liefert das
**(Sim-Bild, Ground-Truth-Aktion)-Paare** — die Trainingsdaten, die der Encoder braucht, um
domäneninvariant zu werden.

Deutlich mehr Arbeit als die Schritte 1–3. Deshalb **erst nach dem Gate**, nicht parallel.

### Schritt 5 — RL bleibt hinten an

Steht so bereits im [RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md): der FPO-Pfad ist
gebaut und auf Hardware validiert, aber er wartet auf eine BC-Policy, die **überhaupt gelegentlich
Erfolg hat**. Einem Nullpunkt-Reward fehlt das Startsignal. Nicht vorziehen.

---

## 4. Nebenbei offen (Komfort, nicht Fortschritt)

| Punkt | Aufwand | Status |
|---|---|---|
| Hardware-Test der Live-Ansicht: `./Simulation/server_rl_run.sh livecheck` | ~10 min | Gebaut 2026-08-13, **nie auf Hardware gelaufen** |
| `SCENE_CAM=0` / `RL_AA_MODE=Off` Wirkung auf Steps/s | ~15 min | Ungemessen (~15 % erwartet) |
| Doku-Abschnitt zur Latenz-Zerlegung in [live-ansicht.md](docs/simulation/live-ansicht.md) | — | **uncommitted** |

Nicht dringend, weil keiner davon die 0/20 bewegt.

---

## 5. Was ausdrücklich *kein* nächster Schritt ist

- **Inferenz-Optimierung.** Die Zerlegung `t(n) = 35,6 ms + n × 9,6 ms` steht in
  [live-ansicht.md](docs/simulation/live-ansicht.md), aber bei 74 ms gegen 267 ms Budget
  (28 % Auslastung) ist heute nichts zu tun. Das ist ein Planungswerkzeug für die Portierung auf
  Zielhardware, keine offene Baustelle.
- **`TUNE_VISUAL=1` ohne Augmentierung.** Das wäre die Wiederholung von Lauf 2 mit dem bereits
  gemessenen Ergebnis (Politik-Kollaps).
- **Qualitätsmessung der Denoising-Schritte über `open_loop_eval.py --denoising-steps`.** Die
  Option ist im Fork **tot** (deklariert, nirgends angewandt) — sie vergleicht zwei identische
  Läufe. Details und der korrekte Weg stehen in
  [live-ansicht.md](docs/simulation/live-ansicht.md).
