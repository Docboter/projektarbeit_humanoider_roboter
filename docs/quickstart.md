# Quickstart

> **TL;DR:** Vom frischen Checkout zu (a) einem laufenden Training und (b) Gate und Closed
> Loop für einen Checkpoint von Hugging Face: nur Befehle und was danach zu sehen sein muss.
> Formate, Ports und Variablen stehen in [schnittstellen.md](schnittstellen.md), Hintergründe in
> den verlinkten Anleitungen.

## 0. Voraussetzungen

| | Training lokal | Simulation | KISSKI |
|---|---|---|---|
| GPU | NVIDIA ≥ 24 GB (bs 1–2); 80 GB für bs 64+ ([VRAM-Tabelle](training/env-vars.md)) | Ampere oder neuer **mit RT-Cores** (L40, RTX 4090, A6000, RTX PRO 6000), ≥ 24 GB. Nicht: A100, H100, Turing | A100 80 GB (`kisski`), H100 94 GB (`kisski-h100`), max. 48 h |
| Host | Linux oder WSL 2, Docker, NVIDIA Container Toolkit, Treiber ≥ 570 | wie links, Linux | `module load apptainer` |
| Platz | ≥ 80 GB | ≥ 60 GB | VAST-Projektspeicher |

### Accounts und Tokens

| Konto | Wofür | Pflicht |
|---|---|---|
| Hugging Face, Token `hf_…` | Lizenz von `nvidia/GR00T-N1.6-3B` akzeptieren; für N1.7 Zugang zu `nvidia/Cosmos-Reason2-2B` (gated) | ja |
| Weights & Biases | Lernkurven | lokal optional, KISSKI Pflicht (sonst `USE_WANDB=0`) |
| Docker Hub (`docker login`) | nur wer eigene Images baut | nein |

Namensräume: `lucam03` = Docker Hub (Images), `lucam06` = GitHub (GR00T-Fork), `luca-mue` =
Hugging Face (Checkpoints). Alle öffentlich lesbar; zum Pushen eigener Images fragen die
Build-Skripte nach `DOCKER_NAMESPACE` ([portabilitaet.md](portabilitaet.md#34-eigene-images-bauen-eigenes-docker-hub-konto)).

```bash
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
git checkout training-luca-IKR-IS6.0
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi   # GPU im Container sichtbar?
```

`--rm` ist nur bei diesem Test erlaubt. Der Trainings-Container hält alle Daten; `--rm` löscht
sie beim Stopp.

Alle Wege lassen sich auch über das Menü starten: `./run.sh`.

## 1. Training starten

### Lokal

```bash
export HF_TOKEN=hf_... WANDB_API_KEY=...       # WANDB optional
./Training/setup_and_train_dockerhub_pull.sh    # N1.7: GROOT_VERSION=1.7 davor
```

Erwartung:

- Image `lucam03/projekt-humanoider-roboter:latest` wird gezogen, Container `groot-train` läuft.
- Im Log: Download von Modell und Datensatz (~18 GB), Konvertierung v3.0 → v2.1, dann
  Trainingsschritte.
- Alle 2000 Steps entsteht ein Checkpoint unter
  `/data/g1_dex3_finetune/blockstacking/g1_dex3_blockstacking_v1/checkpoint-<N>`.
  Der Launcher-Default `MAX_STEPS` ist 30000.

```bash
docker start -ai groot-train                    # nach Stopp fortsetzen
docker cp groot-train:/data/g1_dex3_finetune ./checkpoints
```

### KISSKI

```bash
# einmalig auf glogin-gpu.hpc.gwdg.de
module load apptainer
apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest

export HF_TOKEN=hf_... WANDB_API_KEY=... GLOBAL_BATCH_SIZE=32
sbatch --export=ALL Training/kisski_submit.sh   # ohne --export=ALL fehlen die Tokens
squeue -u $USER; tail -f logs/slurm-<jobid>.out
```

Erwartung: Job im Zustand `R`. Checkpoints alle 5000 Steps unter
`/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/…`, 44000 Steps Default.
Details und Varianten (`TUNE_VISUAL`, Co-Training, Ketten): [kisski-hpc.md](training/kisski-hpc.md).

### Checkpoint hochladen

```bash
python Simulation/scripts/upload_checkpoint.py \
    --checkpoint <…>/checkpoint-44000 --repo <dein-konto>/groot-g1dex3-<lauf>-ckpt44000
```

Erwartung: privates HF-Repo ohne Optimizer-Zustand (`--include-optimizer` nimmt ihn mit). Das
Skript legt Repos immer privat an; öffentlich machen auf der HF-Webseite unter *Settings* →
*Change repository visibility*. Die `luca-mue/*`-Repos sind seit 2026-10-30 öffentlich.

## 2. Checkpoint messen (Sim-Server, z. B. IKR)

Alle Befehle auf dem Sim-Server im Repo-Wurzelverzeichnis. Daten liegen in
`~/groot-rl-data` (= `/data` im Container `groot-rl`).

```bash
cp .env.local.example .env.local                # Host-Pfade eintragen, falls abweichend
./Simulation/server_rl_run.sh preflight         # Image aktuell, GPU mit RT-Cores?
```

Erwartung: `preflight` meldet Python 3.12, die GPU und `ok` für flash-attn und gr00t und endet
mit Exit-Code 0. Sonst Image bauen:
`./Simulation/update_sim_image.sh --standalone`.

### 2a. Visuelles Gate (Open Loop, ohne Isaac Sim)

Einmalig den Container `groot-rl` anlegen. `setup` erzeugt ihn mit dem `HF_TOKEN` des Hosts und
lädt den Default-Checkpoint (~10 GB):

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh setup
```

Die Werkzeuge unten laden fehlende Checkpoints per `docker exec` mit dem `HF_TOKEN` **des
Containers**; ein späteres `export` auf dem Host wirkt nicht. Neuer Token → `server_rl_run.sh
clean`, dann `setup` erneut.

```bash
tools/sim_visgate_seeds.sh groot-rl 1.6 0 groot-g1dex3-cotrain-v3-cube4-ckpt44000
python3 tools/sim_visgate_seeds_summary.py ~/groot-rl-data/sim_results cotrain-v3-cube4-ckpt44000
```

Argumente: Container, `GROOT_VERSION`, GPU, HF-Repo ohne Konto. Erwartung: eine Zeile je
Checkpoint mit MSE Hände auf Echtbild, Prüfset 5 cm und 4 cm. Für diesen Checkpoint stehen die
Vergleichswerte in [co-training.md §3.5](training/co-training.md#35-auswerten--zwei-messungen-nicht-eine)
(Echtbild 0,0021, 5 cm 0,0115, 4 cm 0,0104).

Voraussetzung: Die Prüfsets `/data/cotrain/vis_gate_v3_cube5` und `…_cube4` liegen auf dem
Server (IKR: `~/groot-rl-data/cotrain/`). Sie stehen nicht auf Hugging Face. Neu erzeugen
(Parameter aus [Chronik Lauf 72](ergebnisse/diagnose-chronik.md#lauf-72-co-training-datensätze-v3)):

```bash
SIM_TABLE_SIZE="0.8 2.0" SIM_WRIST_EYE="0.02 0.0 0.08" SIM_WRIST_TARGET="0.30 0.0 -0.08" \
SIM_WRIST_HFOV=85 RENDER_LAYOUT=/data/cotrain/layout_all240_dy3.json \
RENDER_EPISODE_IDS="0 4 8 12" RENDER_STOP_AT_GRASP=0 RENDER_MAX_FRAMES=600 RENDER_NO_DR=1 \
RENDER_OUT=/data/cotrain/vis_gate_v3_cube5 ./Simulation/server_rl_run.sh render
# 4-cm-Satz: SIM_CUBE_EDGE=0.04 und RENDER_OUT=/data/cotrain/vis_gate_v3_cube4
```

Das Layout `layout_all240_dy3.json` (alle 240 Trainings-Episoden, Würfel 3 cm nach außen)
vorher erzeugen ([Chronik Lauf 72](ergebnisse/diagnose-chronik.md#lauf-72-co-training-datensätze-v3)):

```bash
LAYOUT_OUT=/data/cotrain/layout_all240.json RENDER_EPISODES=240 ./Simulation/server_rl_run.sh layout
python3 tools/layout_shift.py ~/groot-rl-data/cotrain/layout_all240.json \
    ~/groot-rl-data/cotrain/layout_all240_dy3.json --dy 0.03
```

### 2b. Closed Loop (Protokoll Lauf 82)

```bash
tools/sim_cl_watchdog.sh cl_meinlauf groot-rl 0 1.6 \
    luca-mue/groot-g1dex3-cotrain-v3-cube4-ckpt44000 /data/checkpoints/cotrain-v3-cube4-ckpt44000
# 4-cm-Modelle: SIM_CUBE_EDGE=0.04 davor
python3 tools/sim_cl_summary.py ~/groot-rl-data/sim_results/results_<zeitstempel>_cotrain-v3-cube4-ckpt44000.json
```

Erwartung:

- Log in `~/groot-rl-data/logs/cl_meinlauf_<datum>.log`, endet mit `[eval] fertig.`
- 20 Episoden à 120 s auf den Testlagen (`layout_heldout_dy3.json`).
- Die Zusammenfassung zeigt `reached` … `stacked` als k/20 mit 95-%-Intervall.
- Ein `lifted_clear`-Treffer zählt erst nach Sichtung des Videos unter
  `~/groot-rl-data/sim_videos/cl_meinlauf/`.

Voraussetzung: `/data/eval/layout_heldout_dy3.json` (Default von `EVAL_LAYOUT` im Wächter).
Inhalt: `layout_heldout.json` (Würfellagen der Test-Episoden 240–300, erzeugt mit
`LAYOUT_HELD_OUT=1 ./Simulation/server_rl_run.sh layout`), darin jeder Würfel 3 cm **seitlich
nach außen** verschoben: y ← y + 0,03·sign(y), x unverändert
([Lauf 62](ergebnisse/diagnose-chronik.md#lauf-62-versatz-sweep--die-würfel-liegen-3-cm-zu-weit-innen),
[Lauf 64](ergebnisse/diagnose-chronik.md#lauf-64--closed-loop-auf-dem-korrigierten-layout-abgebrochen)).
Erzeugen (die Datei der Läufe 64–97 wurde am 2026-10-14 von Hand mit derselben Verschiebung
angelegt; `LAYOUT_BIAS` kann das nicht, es verschiebt gleichsinnig statt gespiegelt):

```bash
LAYOUT_HELD_OUT=1 ./Simulation/server_rl_run.sh layout
python3 tools/layout_shift.py ~/groot-rl-data/eval/layout_heldout.json \
    ~/groot-rl-data/eval/layout_heldout_dy3.json --dy 0.03
```

Bedeutung der Stufen und JSON-Felder: [schnittstellen.md §5](schnittstellen.md#5-sim-eval).

## 3. Wenn etwas hakt

| Symptom | Ursache, Abhilfe |
|---|---|
| `sbatch` startet, Job ohne Token | `--export=ALL` fehlt ([kisski-hpc.md](training/kisski-hpc.md)) |
| Checkpoint lädt nicht, Warnung zu `model_type` | falsche `GROOT_VERSION`; N1.6/N1.7 sind nicht kreuzweise ladbar ([fehlerbehebung.md](fehlerbehebung.md)) |
| CUDA OOM im Training | `GLOBAL_BATCH_SIZE` halbieren |
| Closed Loop hängt mitten in der Episode | `sim_cl_watchdog.sh` startet neu (Chronik Läufe 89/90) |
| Sim startet nicht auf A100/H100 | keine RT-Cores, anderen Server nehmen |

Weitere Fehlerbilder: [fehlerbehebung.md](fehlerbehebung.md), Sim zuerst
[umsetzungsnotizen.md](simulation/umsetzungsnotizen.md). Vollständige Anleitungen:
[Training](training/anleitung.md), [Sim-Eval](simulation/sim-eval-anleitung.md).
