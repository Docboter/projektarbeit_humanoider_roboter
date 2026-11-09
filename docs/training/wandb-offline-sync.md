# W&B Offline-Sync auf KISSKI

> **TL;DR:** Anleitung zum manuellen Hochladen der W&B-Offline-Runs vom KISSKI-Login-Node, da
> die Compute-Nodes keinen Internetzugang haben — inkl. Live-Tracking per periodischem
> Auto-Sync.

KISSKI-Compute-Nodes haben keinen Internetzugang. W&B läuft daher automatisch
im Offline-Modus und schreibt Runs lokal. Nach dem Training müssen sie einmalig
vom Login-Node aus hochgeladen werden.

> **Stand 2026-10-30 — der Sync mit dem `wandb` im Image (0.23.0) lädt nichts mehr hoch.** Er
> meldet `Syncing: … done.` und legt `run-*.wandb.synced` an, der Run erscheint aber nicht in
> W&B; mit `WANDB_DEBUG=true` zeigt sich ein `AttributeError` in
> `server_settings_introspection` (alter Client gegen den heutigen Server). Die Befehle unten mit
> `apptainer exec … /app/Groot-1.6/.venv/bin/wandb sync` sind dadurch **überholt durch den
> Abschnitt [Sync mit aktuellem wandb](#sync-mit-aktuellem-wandb-ab-2026-10-30)**. Ob frühere
> Syncs betroffen waren, zeigt nur der Blick in W&B (Run-ID suchen), nicht die `.synced`-Datei.

## Sync mit aktuellem wandb (ab 2026-10-30)

Auf dem Login-Node, ohne Container, mit dem `uv`-Modul (das System-Python 3.6 ist zu alt):

```bash
module load uv
mkdir -p ~/wandb-sync && cd ~/wandb-sync          # nicht in /tmp: dort gehört ./wandb evtl. fremden Nutzern
export WANDB_API_KEY=$(tr -d '[:space:]' < ~/.wandb_key)
W=/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/wandb
uv run --no-project --python 3.12 --with wandb wandb sync --yes --include-synced \
    -e projektarbeit_humanoider_roboter -p gr00t-g1-dex3 $W/offline-run-<datum>-<id>
```

- `--include-synced` ist nötig, wenn der alte Client die Datei schon als `.synced` markiert hat.
  `--append` gibt es in wandb 0.31 nicht mehr.
- Erfolg prüfen: Die Ausgabe zeigt `uploading history steps …` und `Finished syncing`, und der
  Run ist unter `wandb.ai/projektarbeit_humanoider_roboter/gr00t-g1-dex3/runs/<id>` sichtbar.
- So am 2026-10-30 nachgeholt: Lauf 4 (`2v02ye81`) und Co-Training 73–76 (`ir0qjm0y`,
  `6i2r07hm`, `a2g4md95`, `388unxow`).

## Job starten (mit W&B)

Token **einmalig** in eine Datei hinterlegen (mode 600) — das Submit-Script liest
sie automatisch ein:

```bash
printf '<dein-key>\n' > ~/.wandb_key   # von wandb.ai → Settings → API Keys
chmod 600 ~/.wandb_key
sbatch Training/kisski_submit.sh
```

> **Warum eine Datei statt `export ... && sbatch`?** KISSKI setzt auf dem
> Login-Node `SBATCH_EXPORT=none`. Das überstimmt das `#SBATCH --export=ALL` im
> Script, sodass inline/exportierte Variablen **nicht** im Job ankommen — W&B
> liefe dann ohne Key (kein Logging). Der Datei-Weg umgeht das zuverlässig.
> Alternativ muss `--export=ALL` auf der Kommandozeile stehen (schlägt die
> Env-Variable): `WANDB_API_KEY=<key> sbatch --export=ALL Training/kisski_submit.sh`

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
SIF="${KISSKI_SIF_DIR:-$HOME/images}/projekt-humanoider-roboter.sif"
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
SIF="${KISSKI_SIF_DIR:-$HOME/images}/projekt-humanoider-roboter.sif"
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
    "${KISSKI_SIF_DIR:-$HOME/images}/projekt-humanoider-roboter.sif" \
    /app/Groot-1.6/.venv/bin/wandb sync /data/g1_dex3_finetune/wandb/offline-run-*/
```

Danach sind alle Metriken unter `wandb.ai/<entity>/gr00t-g1-dex3` sichtbar (im Projekt: Entity `projektarbeit_humanoider_roboter`).

## Mehrere Runs auf einmal syncen

```bash
apptainer exec \
    --bind /mnt/vast-kisski/projects/kisski-humrob/data:/data \
    --env "WANDB_API_KEY=<dein-key>" \
    "${KISSKI_SIF_DIR:-$HOME/images}/projekt-humanoider-roboter.sif" \
    bash -c "for run in /data/g1_dex3_finetune/wandb/offline-run-*/; do
        /app/Groot-1.6/.venv/bin/wandb sync \"\$run\"
    done"
```

## Bereits gesyncte Runs überspringen

`wandb sync` überspringe bereits hochgeladene Runs automatisch — der Befehl
kann also bedenkenlos mehrfach ausgeführt werden.
