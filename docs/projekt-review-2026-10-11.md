# Projekt-Review 2026-10-11

> **TL;DR:** Gesamtdurchsicht des Repos (Branch `training-luca-IKR-IS6.0`) und des Branchs
> `training-luca-IKR-IS6.0-GN1.7` vom 2026-10-11: Fehler, Inkonsistenzen und Empfehlungen zu
> Simulation, Training/KISSKI, GN1.7-Migration, Doku und LaTeX, mit Priorisierung. Ursprünglich
> ein reiner Befundbericht; die Befunde aus §3 (Simulation), §4 (Training/KISSKI) und §5
> (GN1.7, seit dem Merge vom 2026-10-13) sind im Code korrigiert, Stand siehe die Tabelle am
> Anfang des jeweiligen Abschnitts. Den laufenden Projektstand und die
> Prioritäten pflegt [`next-steps.md`](../next-steps.md).

## Inhalt

1. [Vorgehen und Belegstatus](#1-vorgehen-und-belegstatus)
2. [Was geprüft und in Ordnung ist](#2-was-geprüft-und-in-ordnung-ist)
3. [Simulation: Abweichungen zwischen Eval und Training](#3-simulation-abweichungen-zwischen-eval-und-training)
4. [Training und KISSKI](#4-training-und-kisski)
5. [GN1.7-Branch](#5-gn17-branch)
6. [Doku und LaTeX](#6-doku-und-latex)
7. [Empfohlene Reihenfolge](#7-empfohlene-reihenfolge)
8. [Weitere Verbesserungsideen](#8-weitere-verbesserungsideen)

---

## 1. Vorgehen und Belegstatus

Der Review wurde in vier parallele Teilprüfungen aufgeteilt: GN1.7-Branch, Training/KISSKI,
Simulation, Doku/LaTeX. Jede Teilprüfung hat den Code gelesen und, wo möglich, Werkzeuge
ausgeführt: `bash -n`, `shellcheck -S warning`, `ruff check` mit der Fork-Konfiguration,
`tools/check_tldr.sh`, `tools/gen_docs.sh`, `tools/test_menu.sh`, `git merge-tree`,
`git range-diff` und ein eigener Link-Checker für alle Markdown-Dateien.

Jeder Befund trägt eine Kennzeichnung:

| Kennzeichnung | Bedeutung |
|---|---|
| **[nachgeprüft]** | Nach der Teilprüfung noch einmal direkt im Repo bzw. auf HF bestätigt |
| **[Teilprüfung]** | Von einer Teilprüfung mit Datei und Zeile belegt, aber nicht noch einmal gegengeprüft |
| **[Vermutung]** | Plausibel, aber ohne Log oder Laufzeitdaten nicht belegbar |

Schweregrade: **kritisch** (verfälscht Messergebnisse), **Fehler** (Skript tut nicht, was es soll),
**Inkonsistenz** (Widerspruch zwischen Code, Konfiguration oder Doku), **Kleinigkeit**.

Zeilennummern beziehen sich auf den Stand vom 2026-10-11 (HEAD `ebe28f5`, GN1.7-Branch `253d9ad`).

---

## 2. Was geprüft und in Ordnung ist

Damit klar ist, was **nicht** das Problem ist:

### Werkzeuge
- `tools/check_tldr.sh`: 176/176 Dateien mit TL;DR (auf dem GN1.7-Branch 182/182).
- `tools/gen_docs.sh`: 402 Parameter-Verwendungen, keine Drift; die 4 Abweichungen sind
  bewusst markiert (`BEWUSST`, `view`-Overrides).
- `tools/test_menu.sh`: 81/81 bestanden, läuft ohne TTY und ohne Docker; auf dem GN1.7-Branch
  ebenfalls 81/81.
- Sim-Menü: 23 Specs in 7 Gruppen, wie in `CLAUDE.md` angegeben.
- `bash -n` sauber für alle `.sh`-Dateien.
- Markdown-Links: 54 Dateien, 0 kaputte Links, 0 kaputte Anker.
- LaTeX-Log: 0 undefinierte Referenzen und Zitate, 0 Overfull-Boxes.

### Code
- **Gelenkreihenfolge stimmt überall.** `ALL_JOINTS_ORDERED` in `g1_dex3_cfg.py` entspricht
  den Namen in `info.json` des HF-Datensatzes, einschließlich der vertauschten Reihenfolge
  Index vor Middle an der rechten Hand.
- **`client.build_obs`** schneidet den State in 0:7 / 7:14 / 14:21 / 21:28, genau wie
  `modality_4cam.json`.
- **Der Vorzeichen-Fix vom 2026-08-24** (`_SIGN_FLIP_IDX=[]`) greift in Beobachtung und
  Aktion. Render und Tipcheck lesen ihn aus dem Env, und durch den Sim-Code-Mount
  (`server_rl_run.sh:442`) ist er ohne Rebuild aktiv.
- **Aktionen:** Relative Arm- und absolute Handaktionen werden auf dem Server dekodiert und
  als absolute Ziele angewendet. Das ist korrekt.
- **Train/Test-Split:** Alle drei Launcher schließen die zurückgehaltenen Episoden aus
  (`split="train"` fest in `factory.py:59`). Der Cotrain-Split gilt nur für den echten
  Datensatz, was richtig ist. Training, Sweep und `render_cotrain_dataset.select_episodes`
  rechnen alle mit `int(total*ratio)` und 0,8, also 240 Train- und 61 Test-Episoden. Es gibt
  keinen Seed, der auseinanderlaufen könnte.
- **`launch_cotrain.py`** stimmt Zeile für Zeile mit `launch_finetune.py` überein, bis auf den
  `datasets`-Block. Das gilt für den gepinnten Commit und für den Submodul-HEAD.
- **`lib_resume_guard.sh`** prüft das richtige Verzeichnis (`experiment.py:131`).
- **`kisski_chain.sh`:** `afterany` zwischen den Trainings, `afterok` vor jedem Sweep, der
  Sweep zeigt auf den echten Datensatz.
- **Routing im Entrypoint:** RL wird abgelehnt, dann kommt COTRAIN vor VISUAL vor Standard.
- **Erfolgskriterien:** 3 Paare innerhalb 3 cm in xy, Höhe über 8 cm,
  Geschwindigkeit unter 0,05, `lifted` über 2 cm. Code und Doku stimmen überein.
- **Zahlen, die überall gleich sind:**
  - bester Checkpoint vision-v2 step 30000, MSE 0,00716 gegenüber 0,00898 bei step 44000 (+25 %)
  - Fingerspanne 27,6 % gegenüber 20,5 % (0,577 gegenüber 0,43 rad), p = 3,3·10⁻⁴
  - `lifted` 0/10, beste Hebung 1,68 cm
  - 240/61 Episoden
  - Laufzeiten 21,4 h / 8,4 h / 12,5 h
  - Konfidenzgrenzen 0/20 → 17 % und 0/10 → 31 %
- **Der Geometrie-Vorbehalt vom 2026-08-24** steht in den Hub-Seiten, im Kopf der
  Chronik, in `lauf3-vision-split-auswertung.md`, `co-training.md`, `fehlerbehebung.md` und
  im LaTeX (04:365-367, 04:501-505, §5.5.1 `sec:vorbehalt-greifzahlen`). Einzige Lücke: siehe
  [6.1](#61-latex).

---

## 3. Simulation: Abweichungen zwischen Eval und Training

Diese Punkte betreffen direkt die Aussagekraft von „0/10 `lifted`“ und von „Domain-Gap
bestätigt“. Sie wiegen deshalb am schwersten.

**Stand der Korrektur (2026-10-13).** Alles ist nur auf der CPU geprüft (82 bestehende plus
10 neue Tests in `test_eval_setup.py`, `py_compile`, `bash -n`, `gen_docs.sh`,
`test_menu.sh`). Ein Sim-Lauf mit den Änderungen steht aus.

| Punkt | Stand | Was geändert wurde | Offen |
|---|---|---|---|
| 3.1 | behoben | `g1_dex3_sim/task_text.py` liest den Text aus `meta/tasks.*` (Fallback `stack three block`); Eval, Env, Client, Render, Replay, Baseline, Latenz, TensorRT, `harmonize_synth_dataset.py`, Entrypoints, Dockerfile und `server_rl_run.sh` umgestellt; Text und Herkunft stehen im Ergebnis-JSON; Doku korrigiert | der isolierte 10-Episoden-Lauf nur mit dieser Änderung |
| 3.2 | behoben | Env-Option `block_layout` (echte Startlagen samt Gierwinkel, je Episode fest zugeordnet); Eval `--layout` / `EVAL_LAYOUT`; `extract_block_layout.py extract --held-out` bzw. `LAYOUT_HELD_OUT=1 server_rl_run.sh layout` → `/data/eval/layout_heldout.json`; benutzte Lage je Episode im JSON | Held-out-Layout auf dem Server erzeugen; Zufallsbänder bleiben als zweite Bedingung |
| 3.3 | behoben | `dt = 1/210`, decimation 7 → exakt 30 Hz (Dex3- und Gripper-Env) | — |
| 3.4 | Schalter eingebaut | `HEAD_CAM_FOLLOWS_PELVIS=1` verschiebt Auge und Blickziel der Kopfkameras starr mit der Basis; `server_rl_run.sh` reicht ihn (mit `ROBOT_BASE_X/Z`) an jeden Sim-Aufruf durch. Default aus | Entscheidung per `tipcheck` + `layoutcheck` in beiden Varianten, **vor** dem Neu-Rendern |
| 3.5 | behoben | `g1_dex3_sim/scene_geometry.py` als gemeinsame Quelle für Tisch, Würfelhöhe und Basis; Dex3-Env, Gripper-Env/-Cfg, Layout-Extraktion und Replay-Kalibrierung lesen daraus | — |
| 3.6 | Prämisse korrigiert | Skriptkopf, Ausgabe und JSON (`note`) sowie Doku und LaTeX (04_ergebnisse) nennen die Messung jetzt eine Näherung | Einbetten mit dem Vision-Tower des Checkpoints |
| 3.7 | behoben | `DR_SEED` Default 0 (fixiert auch den Torch-Zufall der Würfelbänder), negativ = zufällig; Seed, dt, decimation und Layout im Ergebnis-JSON | — |
| 3.8 | behoben | Docstring 300 s; Kommentare zu Wurzel/Taille und Hand-Steifigkeit; neue Messgröße `waist_drift_max_rad` je Episode | Wert im nächsten Lauf ansehen |
| 3.9 | behoben | `ruff --fix` (I001, F401, F541) plus F841 von Hand, übrig nur die erzwungenen E402; vier `replay-*`-Specs; Prüfung 23 in `test_menu.sh` erkannte keine Bindestriche (Muster korrigiert) | — |

**Folge für alte Zahlen:** Läufe vor 2026-10-13 liefen mit falschem Task-Text, 28,57 Hz,
Zufallsbändern und ohne festen Seed. Sie sind mit neuen Läufen nicht direkt vergleichbar.

### 3.1 Falscher Sprachbefehl in allen Closed-Loop-Läufen — kritisch [nachgeprüft]

Der echte Datensatz `unitreerobotics/G1_Dex3_BlockStacking_Dataset` hat genau eine Aufgabe.
Aus `meta/tasks.parquet` direkt von HF gelesen:

```
[{'task_index': 0, '__index_level_0__': 'stack three block'}]
```

Das Training sieht also ausschließlich `"stack three block"`. Die Closed-Loop-Pfade schicken
aber standardmäßig `"stack the blocks"`:

| Datei | Zeile | Verwendung |
|---|---|---|
| `Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py` | 56 | `--task-description`-Default (Eval) |
| `Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py` | 447 | `task_description` im Env-Config (auch RL) |
| `Simulation/g1_dex3_sim/client.py` | 169 | Default im Policy-Client |
| `Simulation/Dockerfile.standalone` | 246 | `ENV TASK_DESCRIPTION` |
| `Simulation/server_rl_run.sh` | 963 | `-e TASK_DESCRIPTION=…`-Default |
| `Simulation/scripts/entrypoint_sim.sh` | 22 (Kommentar) | dokumentierter Default |
| `Simulation/scripts/entrypoint_baseline.sh` | 65 | Baseline-Default |
| `Simulation/kisski_sim_submit.sh` | 90 | KISSKI-Default |
| `Simulation/g1_gripper_sim/*.py` | 45, 182, 194 | Baseline-Gripper-Sim |
| `Simulation/scripts/policy_latency.py` | 101 | Latenzmessung (unkritisch) |
| `Simulation/scripts/optimize_groot_inference.py` | 406 | TensorRT-Export (unkritisch) |
| `Simulation/g1_dex3_sim/render_cotrain_dataset.py` | 782, 790, 1162 | Fallback, falls `tasks` fehlt |
| `Simulation/g1_dex3_sim/run_dataset_replay_videos.py` | 382, 409, 423 | fest eingetragen |

Nur `render_cotrain_dataset.py:38` (Docstring) und `finger_span_openloop.py`, das die
Sprache aus der Datensatz-Trajektorie übernimmt, haben den richtigen Text.

**Folgen:**
- Läufe 29–34 (und alle früheren Closed-Loop-Läufe) haben das Modell mit einem Befehl
  gefahren, der im Training nie vorkam. Wie stark GR00T N1.6 darauf reagiert, ist offen: Das
  LLM ist eingefroren, das Sprach-Embedding fließt aber über das Backbone in den
  Action-Head.
- Der Spannweiten-Test aus Lauf 32 lief **mit** dem echten Befehl, die Closed-Loop-Läufe
  **ohne**. Zwischen Open-Loop und Closed-Loop unterscheiden sich also Bilder **und** Sprache.
  Das schwächt die Aussage „Domain-Gap bestätigt“ zusätzlich, über den schon dokumentierten
  Memorisierungs-Vorbehalt hinaus.
- `run_dataset_replay_videos.py` und der `render_cotrain_dataset.py`-Fallback würden
  Datensätze mit falschem Task-Text erzeugen. Für den Co-Training-Datensatz wäre das ein
  zweiter Task, den das Modell nur mit Sim-Bildern sieht, also genau die Verknüpfung, die
  man vermeiden will.
- Auch die Doku nennt den falschen Text:
  - `docs/simulation/sim-eval-anleitung.md:288`
  - `docs/ergebnisse/lauf1-auswertung.md:10`
  - `docs/ergebnisse/lauf2-vision-auswertung.md:11`
  - `docs/ergebnisse/lauf3-vision-split-auswertung.md:11`
  - `docs/simulation/archiv/isaac-lab-plan.md:27`

**Empfehlung:**
1. Den Text nicht fest eintragen, sondern aus `meta/tasks.jsonl` bzw. `tasks.parquet` des
   Datensatzes lesen. Ersatzweise aus den Metadaten des Checkpoints. `"stack three block"`
   nur als letzter Fallback.
2. Den verwendeten Task-Text in jedes Eval-Ergebnis-JSON schreiben.
3. Danach **nur mit dieser Änderung** 10 Episoden Closed Loop fahren, damit ihr Effekt
   isoliert sichtbar ist.

### 3.2 Würfelpositionen in der Eval liegen außerhalb der Trainingsdaten — kritisch [Teilprüfung]

Die Eval verteilt die Würfel in `_reset_idx` (`g1_dex3_blockstack_env.py:996-1020`) zufällig
in festen Bändern, jede Farbe in ihrem eigenen y-Band. Vergleich mit den echten Startlagen
aus `runs/20260822/04/cotrain/layout.json` (60 Episoden, z = 0,895):

| Würfel | y-Band in der Eval | echte Daten |
|---|---|---|
| rot | immer −0,17 … −0,10 m | Median +0,108 m, Minimum −0,130 m |
| grün | −0,037 … +0,037 m | Median −0,031 m |
| gelb | immer +0,10 … +0,17 m | Median −0,05 m |

Weitere Unterschiede:
- **x-Spanne:** in den echten Daten 0,23–0,50 m, in der Eval 0,30–0,40 m.
- **Drehwinkel:** in der Eval immer 0°, in den echten Daten im Median 10°.
- **Veraltete Herleitung:** `block_x_range` stammt noch aus dem Replay mit der alten
  Basisposition (Kommentar bei 424-426).

Der rote Würfel liegt damit fast immer dort, wo nie ein echter Würfel lag. Gelb liegt auf der
falschen Seite des Medians. „0/10 `lifted`“ misst also zum Teil auch Startlagen, die das
Modell nie gesehen hat.

**Empfehlung:** einen Reset-Modus `--layout layout.json` einbauen, der echte Startlagen samt
Drehwinkel aus den **zurückgehaltenen** Episoden 240–300 übernimmt. Die Zufallsbänder
bleiben als zweite Bedingung, um Generalisierung zu messen.

### 3.3 Regelfrequenz 28,57 Hz statt 30 Hz — Inkonsistenz [nachgeprüft]

`g1_dex3_blockstack_env.py:405` hat `dt = 1/200`, Zeile 413 `decimation = 7`. Ein
Policy-Schritt dauert also 35 ms, im Datensatz sind es 33,3 ms (30 fps). Jeder Aktionsblock
läuft etwa 5 % zu langsam, während die Eval `policy_hz=30` annimmt. Für Greifbewegungen
ist das kein großer Effekt, aber eine unnötige Abweichung.

**Empfehlung:** `dt = 1/210` mit decimation 7 oder `dt = 1/240` mit decimation 8. Dabei
prüfen, ob `render_interval = decimation` weiterhin passt.

### 3.4 Kopfkamera nicht mit dem Roboter verschoben — Inkonsistenz [Teilprüfung]

Mit dem Geometrie-Fix wurde der Roboter um (−0,057 / 0 / −0,086) m verschoben (Becken
0,85 → 0,764 m, Basis x −0,057 m). Die Kameras stehen aber noch an der alten Pose
(`camera_geometry.py:146-150, 180`, z = 1,3239). Gegenüber dem URDF-`d435_link` ist die
Transformation Kamera→Torso damit etwa 5,7 cm in x und 8,6 cm in z versetzt. Der Code
markiert das selbst als „OFFEN“.

Die Herleitung ist zirkulär: Sowohl die Würfellagen-Extraktion als auch die Basiskorrektur
wurden aus genau dieser Kamera abgeleitet. Das Ergebnis ist in sich stimmig, aber nicht
unabhängig belegt. Es ist möglich, dass die Policy ihre eigenen Arme an verschobenen
Bildpositionen sieht.

**Empfehlung:** die Alternative testen, bei der die Kamera dem Becken folgt und die Lagen
neu extrahiert werden, und beide Varianten mit `tipcheck` und `layoutcheck` vergleichen.
Das muss **vor** dem Neu-Rendern des Co-Training-Datensatzes geklärt sein, sonst erbt der
Datensatz den Fehler.

### 3.5 Baseline-Gripper-Sim mit alter Geometrie — Inkonsistenz [Teilprüfung]

`g1_gripper_sim/g1_gripper_blockstack_env.py:62, 77, 188` und `g1_gripper_cfg.py:138`
verwenden noch Tisch 0,89, Würfel 0,915 und Becken 0,85. Die Dex3-Szene verwendet 0,87 /
0,895 / 0,764. Baseline- und Dex3-Zahlen sind damit nicht vergleichbar.

**Empfehlung:** die Szenengeometrie in ein gemeinsames Modul auslagern, das beide Envs
importieren.

### 3.6 Falsche Annahme im Domain-Gap-Skript — Inkonsistenz [Teilprüfung]

`Simulation/scripts/measure_domain_gap.py:4-6, 154` geht davon aus, dass bei
`tune_visual=false` der ViT dem vortrainierten `siglip-so400m-224` entspricht, und misst damit.
Der beste Checkpoint (vision-v2) wurde aber mit `tune_visual=true` trainiert, und N1.6 nutzt
das Eagle-Backbone. Die Kosinus-Distanz ist also ein Näherungswert und nicht die
Wahrnehmungslücke der Policy selbst.

**Empfehlung:** entweder die Prämisse im Skript und in der Doku korrigieren, oder mit dem
Vision-Tower des Checkpoints einbetten. Das Zweite ist aussagekräftiger.

### 3.7 Zufall in der Eval nicht fixiert — Kleinigkeit [Teilprüfung]

In der Eval ist Domain Randomization standardmäßig an und hat keinen festen Seed
(`run_g1_dex3_sim_eval.py:426-431`). A/B-Läufe sind dadurch nicht reproduzierbar.

**Empfehlung:** `DR_SEED` mit festem Default setzen und zusammen mit Checkpoint-Fingerprint,
Task-Text und `EPISODE_LENGTH_S` ins Ergebnis-JSON schreiben.

### 3.8 Veraltete oder irreführende Kommentare — Kleinigkeit [Teilprüfung]

- `g1_dex3_blockstack_env.py:10`: „5400 Schritte / 180 s“, konfiguriert sind 300 s.
- `g1_dex3_cfg.py:166`: „Beine/Waist werden gesperrt“. Tatsächlich ist nur die Wurzel fixiert.
  Taille und Beine haben keine Aktuatorgruppe, und `_pre_physics_step` (Zeile 1124) setzt ihr
  Ziel auf die aktuelle Position. Es lohnt sich, `waist_*` über eine Episode zu loggen, um ein
  Absacken des Torsos auszuschließen.
- Hand-Aktuator: Laut Kommentar ist die Steifigkeit 60, der Wert ist 400.

### 3.9 Lint-Befunde — Kleinigkeit [Teilprüfung]

- **ruff:** 79 Meldungen, alle kosmetisch.
  - 44 × I001 (Import-Reihenfolge)
  - 19 × E402, erzwungen durch das AppLauncher-Setup
  - 14 × F401
  - 1 × F541
  - 1 × F841 in `replay_grasp_metrics.py:229`
- **shellcheck:** In `server_rl_run.sh` nur Fehlalarme: `local extra` ist in einer Funktion ein
  String und in einer anderen ein Array, und `MENU_PROFILE` wird als unbenutzt gemeldet.
- **Menü:** `server_rl_run.sh` kennt 27 Aktionen, die vier `replay-*`-Aktionen haben keine
  Spec und erscheinen nicht im Menü.

---

## 4. Training und KISSKI

**Stand der Korrektur (2026-10-13).** Geprüft mit `bash -n`, `py_compile`, Trockenläufen
(`kisski_chain.sh --dry-run`, `setup_and_train_container_build.sh --dry-run`), `gen_docs.sh`,
`test_menu.sh` und shellcheck/ruff. Kein Lauf auf KISSKI oder mit dem Container.

| Punkt | Stand | Was geändert wurde | Offen |
|---|---|---|---|
| 4.1 | behoben | Launcher baut nur noch das Image (`LOCAL_IMAGE`) und übergibt an `setup_and_train_dockerhub_pull.sh --skip-pull`: ein dauerhafter Container, Entrypoint-Weiche, kein erzwungenes W&B; Bootstrap-Klon und alte Flags bleiben | — |
| 4.2 | behoben | `set +e` / `set -e` um die Trainings-Pipe in allen drei `run_finetuning*.sh` | — |
| 4.3 | behoben | Fehlender bzw. falscher Split oder `info.json` ≠ `split.json` bricht ab (`--allow-train-split` / `ALLOW_TRAIN_SPLIT=1` als Ausnahme); der Loader wird einmal gebaut, ein paralleles `split_apply` kann den Split nicht mehr mitten im Sweep tauschen | — |
| 4.4 | behoben | Vorschlag nur bei `_vN`-Suffix hochgezählt, sonst `<name>_v2` | — |
| 4.5 | teilweise | `kisski_submit.sh`, `kisski_open_loop_eval.sh` und schon die Vorprüfung von `kisski_chain.sh` vergleichen `repo-groot` mit dem Pin im Dockerfile und brechen mit Anleitung ab (`GROOT_FORK_ALLOW_DRIFT=1`); veralteter Kommentar ersetzt; `kisski-hpc.md` nennt den Checkout-Schritt. Befund dabei: der Fork-HEAD ist `dd87740` „(in testing!) add support for older gpus“ | Submodul-Zeiger: `26ad1f7` ist ein nur lokal vorhandener Commit (nur `FINETUNING_GUIDE.md`) — pushen oder auf `9508b49` setzen, Entscheidung offen; `repo-groot` auf KISSKI auf den Pin stellen |
| 4.6 | behoben | `MAX_STEPS`-Fallback in `_vision`/`_cotrain` auf 20000 (wie ENV), Entrypoint-Kommentar korrigiert; `COTRAIN_MIX_RATIO` überall 0.25 (Skript, `launch_cotrain.py`, beide Menü-Specs, Doku) | — |
| 4.7 | behoben | `--export=ALL`-Erklärungen korrigiert; W&B nur unter Apptainer/SLURM offline, `WANDB_DIR` aus `$DATA_DIR`; `USE_RL=1` vor dem Download abgelehnt; `run_finetuning.ps1` als veraltet markiert (Kopf + Laufzeitwarnung); `stats.json`-Hinweis in `train-test-split.md`; `kisski_chain.sh` exportiert `KISSKI_PROJECT_DIR`/`DATA_DIR`, und beide Jobs setzen dafür `DATA_DIR=/data` im Container fest (sonst wäre der Host-Pfad im Entrypoint gelandet) | — |
| 4.8 | behoben | `ruff --fix` (5 × I001); SC2206 entfällt mit 4.1 | SC2034/SC2163 in `kisski_menu.sh` und im Pull-Launcher sind Fehlalarme, bewusst nicht annotiert (GN1.7-Merge) |
| 4.9 | Befund korrigiert | W&B-Konfiguration ausgelesen: `7z81177r` war **kein** Co-Training (ein Datensatz, `mix_ratio` 1, `tune_visual: false`), lief auf dem IKR-Server mit 2 GPUs, RAM 16 % — die Kandidaten unten (KISSKI-Log, RAM-OOM, `merge_statistics`) treffen nicht zu. Eingetragen in Chronik, `co-training.md`, `synth-datensatz.md`, `next-steps.md`. Kurztest-Preset `smoke` in `kisski_chain.sh` (50 Schritte, volle Konfiguration, Folgeläufe per `afterok`) | Absturzursache nur aus `/data/logs/finetune-*.log` im Container `groot-train` des IKR-Servers |

### 4.1 `setup_and_train_container_build.sh` verliert die Daten zwischen den Schritten — Fehler [nachgeprüft]

Download (Zeile 245), Konvertierung (265) und Training (323) laufen jeweils mit
`docker compose run --rm`. `Training/docker-compose.yml` bindet kein Volume ein. Jeder Schritt
läuft also in einem neuen Container, der danach gelöscht wird: Die heruntergeladenen Daten
fehlen beim Konvertieren, die Konvertierung beim Training, und Checkpoints gehen mit dem
Container verloren. Die Host-Prüfungen auf `./data/...` (z. B. Zeile 257) schauen in ein
Verzeichnis, das nie gefüllt wird.

Dazu kommt:
- Der Launcher ruft `run_finetuning.sh` direkt auf. Die Weiche im Entrypoint für
  `TUNE_VISUAL` und `USE_COTRAIN` wird dadurch umgangen.
- Er erzwingt `-e USE_WANDB=1` (Zeile 328). Ohne Key bricht das Training in
  `run_finetuning.sh:122` ab.
- Er widerspricht dem Speichermodell aus `CLAUDE.md` („never pass `--rm`“).

**Empfehlung:** den Launcher auf das Muster von `setup_and_train_dockerhub_pull.sh` umbauen
(ein dauerhafter Container, nur die Image-Quelle ist anders) oder ihn entfernen.

### 4.2 `PIPESTATUS` unter `set -euo pipefail` — Fehler [nachgeprüft]

`run_finetuning.sh:15` setzt `set -euo pipefail`, Zeile 208 liest danach
`EXIT_CODE=${PIPESTATUS[0]}`. Gleiches gilt für `run_finetuning_vision.sh` und
`run_finetuning_cotrain.sh:257-258`. Schlägt das Training in der Pipe
`"${TRAIN_CMD[@]}" 2>&1 | tee …` fehl, bricht das Skript sofort ab. Die eigene Fehlermeldung
„Training mit Exit-Code … siehe $LOG_FILE“ erscheint nie. Der Exit-Code selbst stimmt.

**Empfehlung:**
```bash
set +e
"${TRAIN_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e
```

### 4.3 Kaputter Split ergibt im Sweep stillschweigend Trainings-MSE — Fehler [Teilprüfung]

Fehlt der Test-Split oder passt `info.json` nicht mehr zu `split.json`, gibt
`checkpoint_sweep.py:237-283` nur Warnungen aus und misst trotzdem. Die angebliche
Validierungszahl ist dann Trainings-MSE. Das ist realistisch: In `kisski_chain.sh` läuft der
Sweep parallel zum **nächsten** Training, und dieses schreibt über `split_apply` die
`info.json` um.

**Empfehlung:** mit Exit-Code ≠ 0 abbrechen, wenn `--split test` fehlt oder nicht zum
gespeicherten Split passt, außer bei explizitem `--allow-train-split`. Den Sweep zusätzlich
mit einer eigenen Kopie von `info.json` arbeiten lassen, statt mit der des Datensatzes.

### 4.4 Hinweis-Zeile im Resume-Guard stürzt ab — Fehler [Teilprüfung]

`lib_resume_guard.sh:74` rechnet `$(( ${experiment_name##*_v} + 1 ))`. Bei Namen ohne
numerisches `_vN` ergibt das unter `set -u` „unbound variable“ oder einen Syntaxfehler. Das
Skript endet trotzdem mit Exit 1, aber der Namensvorschlag geht verloren.

**Empfehlung:** nur rechnen, wenn der Name auf `_v[0-9]+$` passt, sonst `<name>_v2`
vorschlagen.

### 4.5 KISSKI nutzt eine andere Fork-Version als das Image — Inkonsistenz [nachgeprüft: Pin-Abweichung]

- `kisski_submit.sh:343-354` bindet `repo-groot/gr00t` ein, einen Depth-1-Klon des
  **Branch-HEAD** des Forks. Das überdeckt den Commit, den das Image pinnt (`9508b49`,
  `Training/Dockerfile:69`). Auf KISSKI läuft also nicht der Code, der im Image steht.
- Der Kommentar in Zeile 352 („Container hat nur gr00t_n1d7-Code“) ist veraltet.
- Das Submodul `app/Groot-1.6` zeigt auf `26ad1f7` (`git ls-tree HEAD app/Groot-1.6`), das
  Dockerfile baut `9508b49`. Laut `groot-n17-migration.md` lässt sich `26ad1f7` nicht mehr vom
  Fork abrufen. `git submodule update` schlägt damit auf frischen Klonen fehl.

**Empfehlung:** In `repo-groot` den gepinnten SHA auschecken oder den Bind entfernen. Das
Submodul auf `9508b49` setzen, sodass es eine einzige Quelle für den Fork-Stand gibt.

### 4.6 Uneinheitliche Standardwerte — Inkonsistenz [Teilprüfung]

**`MAX_STEPS`:**

| Ort | Wert |
|---|---|
| `Training/Dockerfile:125` (`ENV`) | 20000 |
| `entrypoint.sh:79` | 20000 (exportiert immer) |
| `entrypoint.sh:15` (Kommentar) | „default 30000“ |
| `run_finetuning_vision.sh`, `run_finetuning_cotrain.sh:50` | 30000 — wirkt nie, weil das ENV immer gesetzt ist |
| Launcher, `run_finetuning.ps1` | 30000 |
| `kisski_submit.sh` | 44000 |

**`COTRAIN_MIX_RATIO`:**

| Ort | Wert |
|---|---|
| `run_finetuning_cotrain.sh:48`, `launch_cotrain.py:62` | 0,5 |
| `kisski_submit.sh:133` | 0,25 |
| Presets in `kisski_chain.sh:45` | 0,10 |
| Empfehlung in `co-training.md` / `CLAUDE.md` | 0,25 |

`env-vars.md` dokumentiert diese Lücke, aber der Code-Default widerspricht der eigenen
Empfehlung.

**Empfehlung:** einen Wert je Variable an einer Stelle festlegen (Dockerfile-ENV oder
Skript-Default, nicht beides) und den Code-Default für `COTRAIN_MIX_RATIO` auf 0,25 setzen.

### 4.7 Sonstiges — Inkonsistenz [Teilprüfung]

- **`--export=ALL` wird an zwei Stellen falsch erklärt.** `kisski_open_loop_eval.sh:17, 49-51`
  und `kisski_rl_submit.sh:45-46` sagen, `#SBATCH --export=ALL` reiche. Richtig ist, was
  `kisski_submit.sh:22-27` schreibt: `SBATCH_EXPORT=none` auf dem Login-Knoten überschreibt
  die Direktive. Overrides wie `RUN_DIR` oder `KISSKI_PROJECT_DIR` kommen bei einfachem
  `sbatch` nicht im Job an.
- **W&B wird überall auf offline gestellt.** `entrypoint.sh:181-182` setzt
  `WANDB_MODE=offline` auch lokal und auf vast.ai, obwohl die Begründung nur für
  Compute-Knoten gilt. Außerdem ist `WANDB_DIR=/data/...` fest eingetragen statt
  `$DATA_DIR`. Besser nur unter Apptainer/SLURM offline.
- **`USE_RL=1` wird zu spät abgelehnt.** `entrypoint.sh:215` lehnt erst nach dem
  ~25-GB-Download und der Konvertierung ab. Die Prüfung gehört vor Schritt 1.
- **`run_finetuning.ps1` ist deutlich hinter dem Bash-Skript zurück.**
  - Es fehlen `--no-sync`, Split, Resume-Guard, `USE_AUGMENTATION` und
    `GRADIENT_ACCUMULATION_STEPS`, und das Color-Jitter ist fest eingetragen.
  - Die Defaults weichen ab: MaxSteps 30000, SaveSteps 1000/5, 2 Worker gegenüber
    20000, 2000/10 und 8.
  - Entweder nachziehen oder als veraltet markieren.
- **Normalisierungsstatistik enthält die Test-Episoden.** `generate_stats` im Fork
  (`gr00t/data/stats.py:98`) rechnet `stats.json` über alle echten Episoden. Das ist kein
  Trainingsleck, gehört aber als ein Satz in `train-test-split.md`.
- **`kisski_chain.sh:28-29`:** `KISSKI_PROJECT_DIR` und `DATA_DIR` werden für die Vorprüfungen
  benutzt, aber nicht exportiert, und kommen deshalb nicht im Job an.
- **`kisski_rl_submit.sh:27`** hat noch `PLACEHOLDER_RTCORE_PARTITION`. Als Vorlage ist das
  gekennzeichnet, nur der Vollständigkeit halber erwähnt.

### 4.8 Lint — Kleinigkeit [Teilprüfung]

- **shellcheck:**
  - SC2034 für unbenutzte `MENU`, `MENU_PROFILE` und `ONLY_TRAIN`
  - SC2163 in `kisski_menu.sh:84`, ein Fehlalarm
  - SC2206 in `setup_and_train_container_build.sh:331` (`$WANDB_ENV_FLAG` ohne Anführungszeichen)
- **ruff:** 5 × I001 in `checkpoint_sweep.py:39,324`, `dex3_fingerorder_probe.py:45`,
  `harmonize_synth_dataset.py:61` und `launch_cotrain.py:29`. `ruff check --fix` behebt alle.

### 4.9 Abbruch von `cotrain_synth_v1` (2026-09-07, 138 s, W&B `7z81177r`) — ungeklärt

Die Ursache ist nirgends dokumentiert. `synth-datensatz.md` §10.1, `co-training.md` und
`next-steps.md:81` halten nur den Abbruch fest. Die vertauschten Handaktionen in v21 sind ein
Datenqualitätsproblem und keine Absturzursache. Alles, was seitdem korrigiert wurde, betrifft
die Datenqualität:
- v22-Korrektur und `_jointspace`-Re-Export
- `verify --max-action-rmse`
- Pinning der Datensatz-Revision
- `modality.json`- und Checkpoint-Vorprüfungen in der Chain

Nichts davon behebt nachweislich den Absturz.

Die W&B-Laufzeit beginnt bei `wandb.init` im Trainer, also **nach** dem Aufbau des
Datensatzes (Stats-Schritt, `factory.py:42-47`). Der Lauf ist demnach in den ersten
Trainingsschritten gestorben, am wahrscheinlichsten bei den ersten Dataloader-Batches.
Kandidaten **[Vermutung]**, nicht ohne Log prüfbar:
- **Video-Dekodierung des Synth-Datensatzes:** Er wurde mit libx264 und in anderer
  Auflösung neu kodiert.
- **RAM-OOM auf dem Host:** 2 Datensätze × 4 Worker × 4 Ranks. Genau dieser OOM ist der Grund
  für den Kommentar in `kisski_submit.sh:113-115`.
- **`KeyError` in `merge_statistics`** (`sharded_mixture_dataset.py:219-222`), falls dem
  Synth-Satz `relative_action`- oder Quantil-Statistiken fehlen.

**Empfehlung:** `logs/slurm-<jobid>.err` des Laufs aus dem Repo-Verzeichnis auf dem
KISSKI-Projektspeicher holen und die Ursache in `diagnose-chronik.md` eintragen, **bevor**
die Chain eingereicht wird. Außerdem ein Kurztest-Preset in `kisski_chain.sh` vorschalten
(`MAX_STEPS=50`, 1 GPU, beide Datensätze). Die zwei 44k-Läufe schreiben zusammen etwa
400 GB Checkpoints, ein Fehlstart wäre teuer.

---

## 5. GN1.7-Branch

Branch `training-luca-IKR-IS6.0-GN1.7` (`253d9ad`), Merge-Basis `853fd71`. Der Branch hat 10
Commits, die HEAD fehlen, HEAD hat 6, die dem Branch fehlen. Der Diff umfasst 70 Dateien,
+3215/−274 Zeilen.

**Stand der Korrektur (2026-10-13).** Der Branch ist gemergt (`77e4d9b`, Zeiger-Fix `60e336e`);
die Befunde unten sind danach auf `training-luca-IKR-IS6.0` behoben. Geprüft mit `bash -n`,
`py_compile`, ruff, `gen_docs.sh`, `check_tldr.sh`, `test_menu.sh` und Trockenläufen; kein Lauf
auf KISSKI oder im Container, `.ps1` ohne PowerShell nicht geparst.

| Punkt | Stand | Was geändert wurde | Offen |
|---|---|---|---|
| 5.2a | behoben | `kisski_submit.sh` und `kisski_open_loop_eval.sh` normalisieren `GROOT_VERSION` über `groot_normalize_version` **vor** der SIF-Wahl; ungültige Werte brechen vor dem Einreichen ab. Getestet mit `n1.7`, `N17`, `n1d7`, `1.6`, leer, `auto`, Unsinn | — |
| 5.2b | behoben | `read_pin` mit `\|\| true`; ein fehlender Pin führt jetzt zur Warnung in `check_pin` | — |
| 5.2c | behoben | Mit 4.1 erledigt (Launcher übergibt an den Entrypoint, der in `$GROOT_ROOT` konvertiert; Build mit `GROOT_VERSIONS`/Label je Generation). Dazu `docker-compose.yml`: `working_dir: /app/Groot-${GROOT_VERSION:-1.6}` | — |
| 5.2d | entschieden + behoben | Semantik: Augmentierung **an** → leer = Modell-Default (N1.6 0, N1.7 0,2, so lief Lauf 4); **aus** → leer = 0, ein gesetztes `STATE_DROPOUT_PROB` gilt in beiden Fällen. Alle drei `run_finetuning*.sh`, `env-vars.md` | — |
| 5.2e | behoben | `.ps1` (weiter veraltet): setzt `GROOT_VERSION`, bei 1.7 `GROOT_IMAGE_TAG=latest-n17`; prüft `/app/Groot-$GrootVersion/.venv`; Training mit `CUDA_VISIBLE_DEVICES=0` | — |
| 5.2f | behoben | Kommentar in beiden KISSKI-Skripten korrigiert: der Grund, `groot_resolve` auf dem Host zu meiden, ist das HF_HOME aus dem **Host**-`DATA_DIR`, das Apptainer durchreichen würde | — |
| 5.2g | behoben | Entrypoint exportiert `MODEL_DIR` und `MODEL_PATH="${MODEL_PATH:-$MODEL_DIR}"`; `download_data.sh` lädt nach `$MODEL_DIR` | — |
| 5.2h | behoben | doppelte Kopfzeile und doppeltes `mkdir` entfernt; Kopf beschreibt `GROOT_VERSIONS` (Default 1.6, ein Image je Generation) | — |
| 5.2i | behoben | TL;DR, Kurzfazit und „Stand der Umsetzung“ in `groot-n17-migration.md` auf den Stand nach Lauf 4 und Merge; Default `GROOT_VERSIONS`; Pin `cf2032d` | — |
| 5.2j | behoben | `launch_cotrain_n17.py` bricht ab, wenn `ds_weights_alpha` gesetzt ist (würde `mix_ratio` per `len(ds)**alpha` ersetzen) | — |
| 5.5.1/2 | erledigt | siehe a–d | — |
| 5.5.3 | erledigt | Merge-Konflikte gelöst, „untracked Groot-1.7“ aus `CLAUDE.md`/`docs/README.md` entfernt, Statuszeilen in `CLAUDE.md`, `README.md` | — |
| 5.5.4 | behoben | `tools/gen_docs.sh` meldet eine Abweichung der beiden `lib_groot_version.sh`-Kopien als Fehler (beidseitig getestet) | — |
| Merge | Befund | Beim Merge fiel auf: N1.7-Submodul auf `cf2032d`, Dockerfile-Pin noch `efa0169` → beide Dockerfiles nachgezogen; Pin-Prüfung aus 4.5 liest jetzt `ARG GROOT16/17_COMMIT` je Version | Images neu bauen, damit sie `cf2032d` enthalten |

### 5.1 Was stimmt

- **N1.6 bleibt überall Standard:** in `lib_groot_version.sh`, Dockerfile-ARG/ENV, Launchern
  und Menü-Specs.
- **N1.6-Verhalten bleibt gleich:** Ein leeres `STATE_DROPOUT_PROB` ändert nichts, der
  N1.6-Default ist 0,0 wie der früher fest eingetragene Wert.
- **`launch_cotrain_n17.py` passt zum Fork:** Es ist identisch mit `launch_finetune.py` aus
  `app/Groot-1.7@efa0169`, bis auf den argparse-Vorlauf und den `datasets`-Block. `mix_ratio`
  wirkt als relatives Gewicht je Spec (`factory.py:75`).
- **Alle CLI-Flags aus `run_finetuning*.sh`** existieren in der `FinetuneConfig` von N1.7.
- **`checkpoint_sweep.py` nutzt die richtige N1.7-API:** `execution_horizon`,
  `decoder_kwargs`, `split`, `EmbodimentTag.resolve` (wirft `ValueError`) und
  `Gr00tPolicy(device=…)`.
- **`app/Groot-1.7` ist auf dem Branch ein echtes Submodul:** in `.gitmodules` mit Branch
  `luca/g1-dex3-n17`, Gitlink `efa0169`, gleich `GROOT17_COMMIT` im Dockerfile und dem
  lokalen Checkout. N1.7 wird nur mit `--build-arg GROOT_VERSIONS=1.7` bzw.
  `update_image.sh --groot=1.7` gebaut.
- **Werkzeuge auf dem Branch-Stand:** `gen_docs.sh` ohne Drift, `check_tldr.sh` 182/182,
  `test_menu.sh` 81/81.
- **Docker Hub** hat `latest-n17` und `training-luca-IKR-IS6.0-GN1.7-n17` vom 2026-09-29.

### 5.2 Befunde

**(a) Falsche SIF bei abweichender Schreibweise — Fehler [nachgeprüft]**

`kisski_submit.sh:89` (Branch) und `kisski_open_loop_eval.sh:67` wählen die SIF mit eigenem
`case 1.7|n17|N1.7|17` aus, **bevor** `groot_normalize_version` läuft (Zeile 356 bzw. 85).
`GROOT_VERSION=n1.7`, `N17` oder `n1d7` normalisiert der Entrypoint zwar zu 1.7, gemountet
wird aber die N1.6-SIF. Der Entrypoint bricht dann mit „Image enthält N1.7 nicht“ ab, und
zwar erst **nach** der Wartezeit in der SLURM-Queue.
→ Erst normalisieren (Lib sourcen), dann `SIF_SUFFIX` ableiten.

**(b) `read_pin` beendet `update_image.sh` ohne Meldung — Fehler [Teilprüfung]**

`Training/update_image.sh:213-216, 246-247`: Findet `read_pin` keinen Pin, schlägt sein
`grep` unter `set -euo pipefail` fehl. `current="$(read_pin …)"` beendet dann das Skript,
und die Warnung „kein gepinnter Commit — überspringe“ wird nie erreicht. Im Moment
existieren beide ARGs, der Fehler ist also latent.
→ `read_pin() { grep … || true; }`.

**(c) Container-Build-Launcher mit N1.7 — Fehler [Teilprüfung]**

`setup_and_train_container_build.sh:272-274` und `docker-compose.yml:37` arbeiten fest in
`/app/Groot-1.6`. Mit `GROOT_VERSION=1.7` nutzt Compose `:latest-n17`, und dort ist
`/app/Groot-1.6` ein leeres Verzeichnis (Dockerfile:91). Schritt 6 schlägt fehl. Der
Kopfkommentar räumt das ein, aber nichts verhindert es. Zusätzlich gilt 4.1.
→ In `/app/Groot-$GROOT_VERSION` konvertieren (der N1.7-Fork hat
`scripts/lerobot_conversion/`) oder bei 1.7 früh abbrechen.

**(d) State-Dropout bei ausgeschalteter Augmentierung — Inkonsistenz [Teilprüfung]**

`run_finetuning.sh:200` (ebenso `_vision.sh:205` und `_cotrain.sh:260`): Mit
`USE_AUGMENTATION=0` wird `--state_dropout_prob` nicht übergeben. N1.7 trainiert dann mit
seinem Default 0,2 (`app/Groot-1.7` `finetune_config.py:61`), obwohl der Zweig „Aug aus“
behauptet, die Modell-Defaults zu nullen. Ein vom Nutzer gesetztes `STATE_DROPOUT_PROB`
wird bei ausgeschalteter Augmentierung ebenfalls ignoriert.
→ Im else-Zweig `--state_dropout_prob "${STATE_DROPOUT_PROB:-0}"` übergeben. Die Semantik
**vor** dem Vergleichslauf N1.6 gegen N1.7 festlegen und in `03_methoden.tex` festhalten,
sonst vergleicht man zwei verschiedene Regularisierungen.

**(e) `run_finetuning.ps1` mit `-GrootVersion 1.7` — Inkonsistenz [Teilprüfung]**

Der Schalter ändert nur Pfade. Er setzt weder `GROOT_IMAGE_TAG` noch übergibt er
`GROOT_VERSION`, und er geht per `exec` in den laufenden Compose-Container. Standard ist dort
`:latest` = N1.6, ohne `/app/Groot-1.7`. Außerdem fehlt der Single-GPU-Pin
`CUDA_VISIBLE_DEVICES` aus Commit `22c3183`, der genau den dokumentierten
DataParallel-Absturz unter N1.7 verhindert.
→ Vor dem Training prüfen, ob `/app/Groot-$GrootVersion/.venv` existiert, und den GPU-Pin
übernehmen.

**(f) Veralteter Kommentar in `kisski_submit.sh:351-355` — Kleinigkeit [Teilprüfung]**

Der Kommentar sagt, `groot_resolve` werde vermieden, damit N1.6 „kein HF_HOME gesetzt“
behält. Das Dockerfile setzt inzwischen für jedes Image `ENV HF_HOME=/data/hf_cache`
(Dockerfile:184), und der Entrypoint ruft `groot_resolve` ohnehin auf.

**(g) `MODEL_DIR` wird nur halb überschreibbar — Kleinigkeit [Teilprüfung]**

`entrypoint.sh:160` macht `MODEL_DIR` überschreibbar, aber `download_data.sh` und
`run_finetuning*.sh` (`MODEL_PATH`) ignorieren ihn.
→ Den Override entfernen oder `MODEL_PATH="$MODEL_DIR"` exportieren.

**(h) Reste vom Rebase im Dockerfile — Kleinigkeit [Teilprüfung]**

- Zeilen 3 und 4: doppelter Header.
- Zeile 6 sagt „Das Image enthält BEIDE“, aber `ARG GROOT_VERSIONS="1.6"` (Zeile 77) baut
  nur N1.6.
- Zeilen 162 und 166: doppeltes `mkdir` mit veraltetem Kommentar „per Volume gemountet“.

**(i) Migrations-Doku widerspricht Code und sich selbst — Inkonsistenz [Teilprüfung]**

In `docs/weiterfuehrend/groot-n17-migration.md` auf dem Branch:
- Zeilen 44-45 und 53: Standard sei `GROOT_VERSIONS="1.6 1.7"`. Das Dockerfile hat `1.6`,
  und die Doku bestätigt das selbst in Zeilen 166-168.
- Zeile 28 sagt, alle Image-Builds und N1.7-Läufe stünden noch aus, Zeile 52 „Image nicht
  gebaut“. TL;DR (Zeilen 5-6) und die Smoke-Test-Tabelle (Zeile 217) sagen, sie liefen
  erfolgreich.
- Zeilen 74-76 nennen `checkpoint_sweep.py` unter N1.7 ungetestet, Smoke-Test 3 (Zeile 217)
  hat ihn erfolgreich ausgeführt.

**(j) Kleine Absicherung im Co-Training-Launcher fehlt — Kleinigkeit [Teilprüfung]**

Ein gesetztes `ds_weights_alpha` würde in `launch_cotrain_n17.py` den Mix stillschweigend
überschreiben. Eine Prüfung, die das abfängt, wäre billig.

### 5.3 Merge-Risiko [Teilprüfung]

`git merge-tree --write-tree HEAD training-luca-IKR-IS6.0-GN1.7` meldet drei Konflikte, alle
in Doku bzw. LaTeX:

| Datei | Konflikt |
|---|---|
| `CLAUDE.md` | HEAD: `app/Groot-1.7` „untracked … not in .gitmodules“; Branch: echtes Submodul |
| `docs/weiterfuehrend/README.md` | parallele Überarbeitung |
| `latex/chapters/03_methoden.tex` | HEAD hat das Kapitel umgeschrieben; der `TODO(N1.7)`-Kommentar des Branchs muss von Hand neu platziert werden |

Von den 21 auf beiden Seiten geänderten Dateien lassen sich die Code-Dateien sauber
zusammenführen (`kisski_submit.sh`, `run_finetuning_vision.sh`, `rl_finetune.py`). Die
Kommentar-Fixes aus HEAD `253ff7d` bleiben erhalten.

### 5.4 Backup-Branch [Teilprüfung]

`backup/GN1.7-pre-rebase-20260820` enthält nichts, was verloren ginge:
- 4c17008, 224772a und 3b1c4f0 wurden als 7fcf83a, 4e76beb und 564a429 neu angelegt; der
  `range-diff` zeigt nur Kontext-Drift.
- 5ac173d entspricht 25b01e9, das schon in der Basis steckt.

Der Branch kann gelöscht werden.

### 5.5 Empfehlungen zum Branch

1. (a), (b) und (c) vor dem nächsten N1.7-Lauf beheben. Sie sind billig, und (a) schlägt erst
   nach der Queue-Wartezeit fehl.
2. (d) entscheiden, bevor N1.6 und N1.7 verglichen werden.
3. Die Status-Abschnitte von `groot-n17-migration.md` im selben Merge-Commit neu schreiben,
   der den `CLAUDE.md`-Konflikt löst. Dabei den Hinweis „untracked Groot-1.7“ aus
   `CLAUDE.md` entfernen.
4. Einen Check in `test_menu.sh` oder CI ergänzen, der die beiden Kopien von
   `lib_groot_version.sh` per `diff` vergleicht. Heute sind sie identisch, aber nur weil beim
   Review darauf geachtet wurde.

---

## 6. Doku und LaTeX

### 6.1 LaTeX

| Ort | Befund | Schwere | Korrektur |
|---|---|---|---|
| `latex/chapters/04_ergebnisse.tex:492-494` | „in vier Episoden einen erkennbaren Greifversuch mit 60 bis 80 %“. Laut Chronik (`diagnose-chronik.md:1593-1603, 1614-1615`): 4/10 `grasp_attempted` (Schwelle ≥ 30 %), aber nur **3/10** bei 60–80 %; Episode 5 lag bei 35 %. `next-steps.md:66` sagt korrekt „drei von zehn“ | mittel [Teilprüfung] | „in vier Episoden einen Greifversuch, in dreien davon mit 60–80 % Schließung“ |
| `latex/chapters/04_ergebnisse.tex:456` | Tabellenzeile „Greifversuch“, Zelle „Öffnung 24–80 %“: Das ist die Spanne über alle 10 Episoden, nicht über die 4 Versuche (35–80 %) | niedrig [Teilprüfung] | Zelle umbenennen oder 35–80 % angeben |
| `latex/chapters/08_fazit.tex:24-31` (F2) | Wiederholt die Stufen aus Lauf 34 (4/10 Versuche, kein Heben) **ohne** Geometrie-Vorbehalt. Zusammenfassung, Kap. 4 und Kap. 5 haben ihn | niedrig [Teilprüfung] | Einen Nebensatz plus `\cref{sec:vorbehalt-greifzahlen}` ergänzen |
| `latex/preamble/settings.tex:19` | Matrikelnummern A und B stehen noch auf `[Matrikelnummer]`, das Abgabedatum fehlt | vor Abgabe nötig | ausfüllen |
| `latex/references.bib` | `act2023` und `diffusion_policy2023` werden nie zitiert, es gibt kein `\nocite` | niedrig | zitieren (z. B. in den Grundlagen zu Action-Chunking bzw. Diffusion-Heads) oder entfernen |
| `latex/chapters/07_fazit.aux` | Rest eines alten Kapitelnamens (gitignored) | harmlos | löschen |
| Kap. 3 / Kap. 4 (neu) | Nach Korrektur von 3.1 muss in der Methodik stehen, welcher Task-Text in den Closed-Loop-Läufen verwendet wurde | mittel | als Einschränkung der bisherigen Sim-Zahlen aufnehmen |

Die LaTeX-Arbeit wurde nicht neu gebaut: Die Build-Ausgabe landet direkt in `latex/` und
hätte `main.pdf` überschrieben. Das vorhandene `main.log` (22:58) ist jünger als alle Kapitel
(zuletzt 22:53).

### 6.2 Markdown-Doku

| Ort | Befund | Schwere | Korrektur |
|---|---|---|---|
| `next-steps.md:43` | „der Roboter stand ~12 cm versetzt“. Laut Chronik ergaben die **drei** Geometriefehler **zusammen** ~12 cm Abstand Fingerkuppe zu Würfel; die Basis allein war 5,7 cm versetzt (so auch LaTeX 04:529-531) | mittel [Teilprüfung] | „Fingerkuppe und Würfel lagen durch drei Geometriefehler zusammen ~12 cm auseinander“ |
| `CLAUDE.md:29`, `docs/README.md:83` | „Lauf 32 bestätigte den Domain-Gap“ bzw. „Domain-Gap bestätigt“ **ohne** den Zusatz zur Memorisierung. `CLAUDE.md:13`, `next-steps.md:45-46`, der Kopf der Chronik und LaTeX 04:358-363 sagen alle „starkes Indiz, kein Beweis“ | mittel [Teilprüfung] | „(starkes Indiz; der Test nutzte Trainings-Episoden 0–4)“ ergänzen und nach 3.1 um den Prompt-Unterschied erweitern |
| `docs/weiterfuehrend/reinforcement-learning-plan.md:215` | RT-Core-GPUs der jupyter-Partition seien „teils zu alt“; überall sonst ist die RTX 5000 eindeutig inkompatibel | niedrig | „zu alt (Turing)“ |
| `CLAUDE.md` (Lauf-Zählung) | `CLAUDE.md` sagt, projektweite „Lauf N“-Verweise lösen sich in `diagnose-chronik.md` auf; die Chronik umfasst aber Läufe 08–53. Läufe 1/2/3 (README.md:27, next-steps, LaTeX 04:342-343) sind **Trainingsläufe** in `lauf{1,2,3}-*-auswertung.md`. Die Bereiche überlappen nicht, die Aussage ist aber irreführend | niedrig | ein Satz in `CLAUDE.md` und `docs/ergebnisse/README.md`, der die zwei Zählungen trennt |
| `CLAUDE.md`, Kommentar zu `server_rl_run.sh` (~Zeile 370) | nennt `view` doppelt; es fehlen layout, layoutcheck, yawcheck, topface, tipcheck, latency, optimize und replay-* | niedrig | alle Aktionen aufzählen oder „siehe `usage`“ |
| `CLAUDE.md:349` | `kisski_sim_submit.sh` als „jupyter partition, RTX 5000“ ohne Hinweis, dass der Weg blockiert ist (vgl. `CLAUDE.md:40`, `umsetzungsnotizen.md:34`) | niedrig | „(blockiert: Turing)“ ergänzen |
| `CLAUDE.md`, Architektur-Baum | Es fehlen: `Simulation/compare_domain_gap.sh`, `Simulation/kisski_replay_submit.sh`, `Simulation/kisski_robocasa_ref_submit.sh`, `docs/training/synth-datensatz.md` (nur in einem Kommentar erwähnt), `docs/weiterfuehrend/cli-menuefuehrung.md`, `docs/weiterfuehrend/wiki-migration-plan.md`, `docs/ergebnisse/basismodell-referenz-eval.md` | niedrig | ergänzen |
| `docs/simulation/sim-eval-anleitung.md:288`, `docs/ergebnisse/lauf{1,2,3}-*.md` | Task „stack the blocks“, siehe 3.1 | mittel [nachgeprüft] | auf „stack three block“ korrigieren |
| `docs/training/train-test-split.md` | Hinweis fehlt, dass `stats.json` über alle Episoden einschließlich Test gerechnet wird (siehe 4.7) | niedrig | einen Satz ergänzen |
| `docs/ergebnisse/diagnose-chronik.md` | Ursache des Abbruchs von `cotrain_synth_v1` fehlt (siehe 4.9) | mittel | nach Log-Auswertung nachtragen |

---

## 7. Empfohlene Reihenfolge

1. **Sprachbefehl korrigieren** (3.1). Den Text aus dem Datensatz lesen und dann **nur mit
   dieser Änderung** 10 Episoden Closed Loop fahren, dazu den Spannweiten-Test auf den
   zurückgehaltenen Episoden 240–300. Das ist die günstigste Änderung, die das Ergebnis allein
   verändern könnte, und ihr Effekt ist so isoliert messbar.
2. **Messaufbau festziehen**, bevor weitere Sim-Zahlen entstehen:
   - echte Startlagen aus den zurückgehaltenen Episoden (3.2)
   - 30 Hz Regelfrequenz (3.3)
   - fester `DR_SEED`, feste `EPISODE_LENGTH_S` (40–60 s), Checkpoint-Fingerprint und
     Task-Text im Ergebnis-JSON (3.7)
   - Kamera und Becken klären (3.4), **bevor** der Render-Datensatz neu entsteht
   - Baseline-Geometrie angleichen (3.5)
3. **Vor dem nächsten KISSKI-Lauf:**
   - Abbruchursache von `cotrain_synth_v1` klären und dokumentieren (4.9)
   - Kurztest-Preset in `kisski_chain.sh`
   - kaputten Split im Sweep zum harten Fehler machen (4.3)
   - Fork-Bind auf den Image-Commit pinnen, Submodul auf `9508b49` (4.5)
   - Defaults für `MAX_STEPS` und `COTRAIN_MIX_RATIO` vereinheitlichen (4.6)
4. **Startskripte reparieren:**
   - `container_build`-Launcher (4.1), `PIPESTATUS` (4.2), Resume-Guard (4.4)
   - auf dem GN1.7-Branch Versions-Normalisierung (5.2a), `read_pin` (5.2b), Konvertierung
     unter N1.7 (5.2c)
5. **LaTeX vor der Abgabe:**
   - 3 statt 4 Episoden bei 60–80 % und die Tabellenzelle (6.1)
   - Geometrie-Vorbehalt in F2
   - Matrikelnummern und Abgabedatum
   - ungenutzte Literatur
   - nach Schritt 1 die Task-Text-Einschränkung
6. **GN1.7-Branch mergen.** State-Dropout-Semantik (5.2d) vorher festlegen, Migrations-Doku
   im Merge-Commit aktualisieren (5.2i), Backup-Branch löschen (5.4).
7. **Doku-Kleinigkeiten** aus 6.2 und der Architektur-Baum in `CLAUDE.md`.

---

## 8. Weitere Verbesserungsideen

- **Eine Quelle für Szenen- und Kamerageometrie.** Tischhöhe, Würfelhöhe, Beckenhöhe,
  Basis-Offset und Kamerapose stehen heute in mehreren Dateien (Dex3-Env, Gripper-Env,
  `camera_geometry.py`, Render, Layout-Extraktion). Ein gemeinsames Modul mit einem Test, der
  die Werte gegen die URDF prüft, hätte 3.4 und 3.5 verhindert.
- **Eval-Manifest.** Jede Sim-Messung schreibt ein JSON mit:
  - Git-SHA und Checkpoint-Pfad mit Hash
  - Task-Text, `DR_SEED`, Physik-dt und decimation
  - Würfel-Startlagen-Quelle und Kamerapose

  Das macht Läufe vergleichbar und den Geometrie-Vorbehalt künftig überflüssig.
- **Kleiner Python-Unit-Test-Satz für die Sim-Schnittstelle**, ohne Isaac und auf der CPU
  lauffähig:
  - `build_obs`-Slices gegen `modality_4cam.json`
  - Gelenkreihenfolge gegen `info.json`
  - Task-Text gegen `tasks.parquet`
  - Split-Rechnung von Sweep, Render und `lib_split.sh` identisch

  `camera_geometry.py` ist schon Isaac-frei und passt in dieses Muster.
- **Link-Checker** (`links.py` aus diesem Review) als `tools/check_links.sh` neben
  `check_tldr.sh` aufnehmen. Heute sind alle Links sauber, so bleibt es auch.
- **`ruff check --fix`** einmal über `Training/scripts/` und `Simulation/` laufen lassen und
  danach als pre-commit-Hook für diese Verzeichnisse aktivieren. Der Fork hat die Konfiguration
  schon.
- **`run_finetuning.ps1`:** Wenn Windows nicht mehr gebraucht wird, als veraltet markieren
  statt nachziehen. Die Paritätslücke wächst mit jeder Änderung am Bash-Launcher.
- **`setup_and_train_container_build.sh`:** entfernen, statt ihn zu reparieren, falls niemand
  lokal baut. `update_image.sh` und der Dockerhub-Pull-Launcher decken beide Wege ab.
