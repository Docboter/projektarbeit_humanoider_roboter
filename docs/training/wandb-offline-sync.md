# W&B Offline-Sync auf KISSKI

KISSKI-Compute-Nodes haben keinen Internetzugang. W&B läuft daher automatisch
im Offline-Modus und schreibt Runs lokal. Nach dem Training müssen sie einmalig
vom Login-Node aus hochgeladen werden.

## Job starten (mit W&B)

Token **einmalig** in eine Datei hinterlegen (mode 600) — das Submit-Script liest
sie automatisch ein:

```bash
printf '<dein-key>\n' > ~/.wandb_key   # von wandb.ai → Settings → API Keys
chmod 600 ~/.wandb_key
sbatch ~/.project/dir.project/kisski_submit.sh
```

> **Warum eine Datei statt `export ... && sbatch`?** KISSKI setzt auf dem
> Login-Node `SBATCH_EXPORT=none`. Das überstimmt das `#SBATCH --export=ALL` im
> Script, sodass inline/exportierte Variablen **nicht** im Job ankommen — W&B
> liefe dann ohne Key (kein Logging). Der Datei-Weg umgeht das zuverlässig.
> Alternativ muss `--export=ALL` auf der Kommandozeile stehen (schlägt die
> Env-Variable): `WANDB_API_KEY=<key> sbatch --export=ALL ~/.project/dir.project/kisski_submit.sh`

`WANDB_MODE=offline` und `WANDB_DIR` werden automatisch vom Submit-Script gesetzt.
Kein manuelles Konfigurieren nötig.

## Wo werden die Runs gespeichert?

Die Offline-Runs landen auf dem Cluster unter:

```
/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/wandb/offline-run-*/
```

## Live-Tracking während des Runs (inkrementeller Sync)

Man muss **nicht** bis zum Ende warten: Der Compute-Node schreibt den Offline-Run
fortlaufend auf das geteilte Filesystem, das auch der Login-Node sieht. Mit
`wandb sync --append` lässt sich der laufende Run vom Login-Node aus wiederholt
hochladen — jeder Aufruf hängt nur die **neuen** Schritte an denselben W&B-Run an
(kein Duplikat, kein Neustart). Ergebnis: nahezu-Live-Tracking mit einem Verzug,
der dem Sync-Intervall entspricht.

**Einmal-Sync (Test):**

```bash
module load apptainer
SIF=~/.project/dir.project/images/projekt-humanoider-roboter.sif
DATA=/mnt/vast-kisski/projects/kisski-humrob/data
RUN=$(ls -dt $DATA/g1_dex3_finetune/wandb/offline-run-* | head -1)   # neuester Run

apptainer exec \
    --bind "$DATA":/data \
    --env "WANDB_API_KEY=$(tr -d '[:space:]' < ~/.wandb_key)" \
    "$SIF" \
    /app/Groot-1.6/.venv/bin/wandb sync --append "/data/g1_dex3_finetune/wandb/$(basename "$RUN")"
```

**Periodischer Auto-Sync** in einer `screen`-Session (überlebt SSH-Trennung,
stoppt automatisch, wenn der Job fertig ist — `<jobid>` anpassen):

```bash
screen -S wandbsync
module load apptainer
SIF=~/.project/dir.project/images/projekt-humanoider-roboter.sif
DATA=/mnt/vast-kisski/projects/kisski-humrob/data
RUN=$(basename "$(ls -dt $DATA/g1_dex3_finetune/wandb/offline-run-* | head -1)")
JOBID=<jobid>

while squeue -j "$JOBID" -h 2>/dev/null | grep -q .; do
    apptainer exec --bind "$DATA":/data \
        --env "WANDB_API_KEY=$(tr -d '[:space:]' < ~/.wandb_key)" \
        "$SIF" /app/Groot-1.6/.venv/bin/wandb sync --append "/data/g1_dex3_finetune/wandb/$RUN"
    sleep 120
done
# finaler Sync nach Job-Ende
apptainer exec --bind "$DATA":/data \
    --env "WANDB_API_KEY=$(tr -d '[:space:]' < ~/.wandb_key)" \
    "$SIF" /app/Groot-1.6/.venv/bin/wandb sync --append "/data/g1_dex3_finetune/wandb/$RUN"
```

Mit `Ctrl+A, D` abdocken. Hinweise:

- **`--append` ist Pflicht** für wiederholtes Syncen desselben Runs — ohne das
  Flag würde `wandb sync` den Run als bereits hochgeladen überspringen.
- Gleichzeitiges Lesen, während der Container die `.wandb`-Datei schreibt, ist
  unkritisch: `wandb sync` liest nur bis zum aktuellen Dateiende.
- Erfordert `wandb >= 0.15` (im Image: 0.23.0 ✓).

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
