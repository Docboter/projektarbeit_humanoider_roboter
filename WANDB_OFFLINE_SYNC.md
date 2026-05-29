# W&B Offline-Sync auf KISSKI

KISSKI-Compute-Nodes haben keinen Internetzugang. W&B läuft daher automatisch
im Offline-Modus und schreibt Runs lokal. Nach dem Training müssen sie einmalig
vom Login-Node aus hochgeladen werden.

## Job starten (mit W&B)

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=<dein-key>   # von wandb.ai → Settings → API Keys
(sbatch ~/.project/dir.project/kisski_submit.sh)
sbatch --export=ALL,HF_TOKEN=$HF_TOKEN,WANDB_API_KEY=$WANDB_API_KEY   ~/.project/dir.project/kisski_submit.sh
```

`WANDB_MODE=offline` und `WANDB_DIR` werden automatisch vom Submit-Script gesetzt.
Kein manuelles Konfigurieren nötig.

## Wo werden die Runs gespeichert?

Die Offline-Runs landen auf dem Cluster unter:

```
/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/wandb/offline-run-*/
```

## Nach dem Training: Sync zum W&B-Dashboard

Auf dem **Login-Node** (`glogin10`, hat Internetzugang) ausführen:

```bash
module load apptainer

apptainer exec \
    --bind /mnt/vast-kisski/projects/kisski-humrob/data:/data \
    --env "WANDB_API_KEY=<dein-key>" \
    ~/.project/dir.project/images/projekt-humanoider-roboter.sif \
    /app/Groot-1.6/.venv/bin/wandb sync /data/g1_dex3_finetune/wandb/offline-run-*/
```

Danach sind alle Metriken unter `wandb.ai/<dein-username>/gr00t-g1-dex3` sichtbar.

## Mehrere Runs auf einmal syncen

```bash
apptainer exec \
    --bind /mnt/vast-kisski/projects/kisski-humrob/data:/data \
    --env "WANDB_API_KEY=<dein-key>" \
    ~/.project/dir.project/images/projekt-humanoider-roboter.sif \
    bash -c "for run in /data/g1_dex3_finetune/wandb/offline-run-*/; do
        /app/Groot-1.6/.venv/bin/wandb sync \"\$run\"
    done"
```

## Bereits gesyncte Runs überspringen

`wandb sync` überspringe bereits hochgeladene Runs automatisch — der Befehl
kann also bedenkenlos mehrfach ausgeführt werden.
