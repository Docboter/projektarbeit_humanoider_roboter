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
    --checkpoint <…>/checkpoint-44000 --repo luca-mue/groot-g1dex3-<lauf>-ckpt44000
```

Erwartung: privates HF-Repo ohne Optimizer-Zustand (`--include-optimizer` nimmt ihn mit).

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

```bash
export HF_TOKEN=hf_...
tools/sim_visgate_seeds.sh groot-rl 1.6 0 groot-g1dex3-cotrain-v3-cube4-ckpt44000
python3 tools/sim_visgate_seeds_summary.py ~/groot-rl-data/sim_results cotrain-v3-cube4-ckpt44000
```

Argumente: Container, `GROOT_VERSION`, GPU, HF-Repo ohne Konto. Erwartung: eine Zeile je
Checkpoint mit MSE Hände auf Echtbild, Prüfset 5 cm und 4 cm. Für diesen Checkpoint stehen die
Vergleichswerte in [co-training.md §3.5](training/co-training.md#35-auswerten--zwei-messungen-nicht-eine)
(Echtbild 0,0021, 5 cm 0,0115, 4 cm 0,0104).

Voraussetzung: Die Prüfsets `/data/cotrain/vis_gate_v3_cube5` und `…_cube4` liegen auf dem
Server. Sie stehen nicht auf Hugging Face; auf einem neuen Server müssen sie mit
`server_rl_run.sh render` neu erzeugt werden (Parameter:
[Chronik Lauf 72](ergebnisse/diagnose-chronik.md#lauf-72-co-training-datensätze-v3)).

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
