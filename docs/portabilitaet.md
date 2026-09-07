# Portabilität — das Repo auf einem fremden Rechner betreiben

> **TL;DR:** Anleitung, um das Repo auf einem fremden Rechner oder KISSKI-Projekt zum Laufen
> zu bringen: `.env.local` für den Docker-Server, `KISSKI_PROJECT_DIR`/`KISSKI_SIF_DIR` für
> den Cluster, plus die Migrationsschritte für die bestehenden Maschinen.

**Stand: 2026-08-18.** Bis zu diesem Datum enthielten mehrere Skripte fest verdrahtete
Pfade auf genau *einen* Rechner bzw. *ein* HPC-Konto. Wer das Repo klonte, lief in
`Permission denied` oder in Jobs, die gar nicht erst starteten. Dieses Dokument beschreibt

1. **was sich geändert hat**,
2. **was du einmalig tun musst, um auf deinen eigenen Maschinen exakt das alte Verhalten
   zurückzubekommen** (§2 — das ist der wichtige Teil, wenn du von einem älteren Stand kommst),
3. **wie eine fremde Person das Repo in Betrieb nimmt** (§3),
4. eine **Referenz aller Konfigurationsknöpfe** (§4).

---

> **Nachtrag 2026-08-20 — `.env.local` gilt jetzt wirklich überall.** Bis zu diesem Datum
> lasen nur [`server_rl_run.sh`](../Simulation/server_rl_run.sh) und
> `server_robocasa_ref_run.sh` die Datei; in den Trainings-Launchern fehlte der Block
> ganz, die unten beschriebene Vorrangregel galt dort also nicht. Ein in `.env.local`
> hinterlegter `HF_TOKEN` wurde von
> [`setup_and_train_dockerhub_pull.sh`](../Training/setup_and_train_dockerhub_pull.sh)
> trotzdem abgefragt. Die Mechanik steckt jetzt einmal in
> [`tools/lib_env_local.sh`](../tools/lib_env_local.sh) und wird von beiden Seiten
> gesourct. Aufgefallen beim Umsetzen der
> [CLI-Menüführung](weiterfuehrend/cli-menuefuehrung.md).

## 1. Was sich geändert hat

| Wo | Vorher (fest verdrahtet) | Jetzt |
|---|---|---|
| [`Simulation/server_rl_run.sh`](../Simulation/server_rl_run.sh) | `HOST_DATA_DIR=/home/lmuecke/project/data/RL` | `$HOME/groot-rl-data`, überschreibbar per `RL_HOST_DATA_DIR` **oder** `.env.local` |
| dieselbe Datei | — | liest beim Start eine gitignorierte `.env.local` aus dem Repo-Wurzelverzeichnis (Vorlage: [`.env.local.example`](../.env.local.example)) |
| dieselbe Datei | `mkdir -p` scheiterte still | `ensure_host_data_dir()` bricht mit konkreter Anleitung ab |
| [`Simulation/server_robocasa_ref_run.sh`](../Simulation/server_robocasa_ref_run.sh) | `HOST_DATA_DIR=/home/lmuecke/project/data/RoboCasa` | `$HOME/groot-robocasa-data`, `RC_HOST_DATA_DIR` bzw. `.env.local`, gleicher Guard |
| 6 KISSKI-Skripte | `SIF=/user/luca.muecke/u28320/.project/dir.project/images/…` | Kandidatenliste: `$KISSKI_SIF_DIR` (Default `$HOME/images`) → `$KISSKI_PROJECT_DIR/images` |
| 6 KISSKI-Skripte | `DATA_DIR`, `REPO_DIR`, `GROOT_FORK_DIR`, `ASSETS_DIR`, `ISAAC_CACHE`, `SIM_CODE` je einzeln auf `/mnt/vast-kisski/projects/kisski-humrob/…` | alle abgeleitet aus **einem** `KISSKI_PROJECT_DIR` |
| 4 KISSKI-Skripte | `export APPTAINER_CACHEDIR="…"` überschrieb eine gesetzte Variable | `${APPTAINER_CACHEDIR:-…}` respektiert sie |
| 4 KISSKI-Skripte | `#SBATCH --output=/user/luca.muecke/…/logs/…` | `logs/<name>-%j.out`, relativ zum Submit-Verzeichnis |
| [`latex/.latexmkrc`](../latex/.latexmkrc) | Windows-Python-Pfad eines bestimmten Rechners im `PATH` | greift nur noch unter Windows und nur, wenn der Pfad existiert |

Die Rangfolge ist überall dieselbe:

```
explizit gesetzte Umgebungsvariable   >   .env.local (nur server_rl_run.sh)   >   Default im Skript
```

---

## 2. Migration — das alte Verhalten auf den eigenen Maschinen wiederherstellen

> **Zeitgebundener Abschnitt:** gilt nur für die Umstellung nach dem Umbau vom 2026-08-18.
> Sobald die Checkliste in § 2.3 auf beiden Maschinen abgearbeitet ist, ist dieser Abschnitt
> nur noch historisch relevant (→ dann nach [historie.md](historie.md) verschieben);
> für neue Nutzer gilt allein § 3.

### 2.1 IKR-Server (Docker, `server_rl_run.sh`)

Bisher lag `/data` eines einzelnen Benutzerkontos fest unter
`/home/lmuecke/project/data/RL`. Ohne Konfiguration landet es jetzt portabel unter
`$HOME/groot-rl-data`. Auf dem IKR-Server empfiehlt sich der ebenfalls portable
Projektpfad `$HOME/project/data/RL` — im Repo-Wurzelverzeichnis:

```bash
cp .env.local.example .env.local
echo ': "${RL_HOST_DATA_DIR:=$HOME/project/data/RL}"'       >> .env.local
echo ': "${RC_HOST_DATA_DIR:=$HOME/project/data/RoboCasa}"' >> .env.local
```

Die zweite Zeile gilt für [`server_robocasa_ref_run.sh`](../Simulation/server_robocasa_ref_run.sh)
(Referenz-Eval) — eigenes Verzeichnis, eigener Container, dieselbe Mechanik.

Prüfen, dass es gegriffen hat (die Ausgabe muss den alten Pfad zeigen):

```bash
./Simulation/server_rl_run.sh help | head -3      # keine Nebenwirkungen
grep RL_HOST_DATA_DIR .env.local
```

> Der Bind-Mount wird nur beim **Anlegen** des Containers gesetzt; ein bereits existierender
> `groot-rl` behält seinen alten Mount. `server_rl_run.sh` vergleicht deshalb den realen
> `/data`-Mount mit `RL_HOST_DATA_DIR` und bricht bei einer Abweichung ab. Dann entweder
> `.env.local` auf den bestehenden Datenpfad korrigieren oder den Container mit `clean`
> neu anlegen. `clean` löscht das Host-Datenverzeichnis nicht.

Praktisch: In dieselbe Datei gehören auch `HF_TOKEN` und `WANDB_API_KEY` — dann entfällt das
Voranstellen bei jedem Aufruf. Siehe [`.env.local.example`](../.env.local.example).

### 2.2 KISSKI (Apptainer + SLURM)

Drei Dinge haben sich verschoben. Alle drei sind mit je einem Handgriff erledigt.

**a) SIF-Ablage.** Die Skripte suchen zuerst in `$HOME/images` (das ist der Ort, den
`apptainer pull` in [CLAUDE.md](../CLAUDE.md) und [kisski-hpc.md](training/kisski-hpc.md)
seit jeher nennt), dann in `$KISSKI_PROJECT_DIR/images`. Deine SIFs liegen unter
`/user/luca.muecke/u28320/.project/dir.project/images`. Der billigste Weg — ein Symlink auf
dem Login-Knoten, nichts wird kopiert:

```bash
ln -s /user/luca.muecke/u28320/.project/dir.project/images "$HOME/images"
```

Alternative ohne Symlink, einmalig in `~/.bashrc` auf dem Cluster:

```bash
export KISSKI_SIF_DIR=/user/luca.muecke/u28320/.project/dir.project/images
```

Beides ist gleichwertig. Ohne eines von beiden meldet der Job sauber
`FEHLER: SIF nicht gefunden: …` samt `apptainer pull`-Kommando, statt kryptisch zu scheitern.

**b) Projektspeicher.** Der Default ist unverändert `/mnt/vast-kisski/projects/kisski-humrob`
— **hier musst du nichts tun.** Neu ist nur, dass ein einziges `KISSKI_PROJECT_DIR` alle
abgeleiteten Pfade umschaltet (siehe §3.2).

**c) SLURM-Logs.** Vorher schrieben vier Skripte nach
`/user/luca.muecke/u28320/.project/dir.project/logs/`. Jetzt schreiben **alle sechs**
einheitlich nach `logs/` **relativ zum Verzeichnis, aus dem du `sbatch` aufrufst**. Grund:
In `#SBATCH`-Zeilen kann SLURM keine Umgebungsvariablen auflösen — ein Knopf wie
`KISSKI_PROJECT_DIR` wirkt dort prinzipiell nicht, ein absoluter Pfad ist also entweder
fremd-unbrauchbar oder gar nicht parametrierbar.

Damit ein Job startet, muss das Verzeichnis existieren (SLURM legt die *Datei* an, nicht den
*Ordner*). Einmalig im Repo-Checkout auf dem Cluster:

```bash
cd /mnt/vast-kisski/projects/kisski-humrob/repo && mkdir -p logs
```

Willst du die Logs weiterhin an deinem alten Ort haben, geht das ohne Skript-Änderung:

```bash
sbatch --output=/user/luca.muecke/u28320/.project/dir.project/logs/slurm-sim-%j.out \
       --error=/user/luca.muecke/u28320/.project/dir.project/logs/slurm-sim-%j.err \
       Simulation/kisski_sim_submit.sh
```

### 2.3 Checkliste

| | Rechner | Handgriff |
|---|---|---|
| ☐ | IKR-Server | `cp .env.local.example .env.local` + `RL_HOST_DATA_DIR`- und `RC_HOST_DATA_DIR`-Zeile eintragen |
| ☐ | IKR-Server | optional `HF_TOKEN` / `WANDB_API_KEY` in dieselbe Datei |
| ☐ | KISSKI Login-Knoten | `ln -s …/images "$HOME/images"` **oder** `export KISSKI_SIF_DIR=…` in `~/.bashrc` |
| ☐ | KISSKI Repo-Checkout | `mkdir -p logs` |
| ☐ | KISSKI Repo-Checkout | `git pull`, damit die neuen Skripte dort ankommen |
| ☐ | IKR-Server | Sim-Standalone-Image auf den neuen Namen bringen (§2.4) |

### 2.4 Sim-Standalone-Image umbenannt (ehemals `sim-vastai`)

Das kombinierte Isaac-Lab+GR00T-Image heißt jetzt nach seinem Zuschnitt statt nach einer
Plattform — **Standalone**, weil derselbe Container unverändert auf dem IKR-Server **und** auf
vast.ai läuft: `lucam03/projekt-humanoider-roboter-sim-standalone`.
`server_rl_run.sh` erwartet diesen Namen inzwischen als `RL_IMAGE`-Default (siehe §4). Auf dem
IKR-Server einmalig eine der folgenden Optionen:

```bash
# a) neu bauen und pushen (dauert am längsten, aber sauber)
./Simulation/update_sim_image.sh --standalone

# b) oder lokal nur umtaggen, wenn das alte Image bereits vorhanden ist
docker tag lucam03/projekt-humanoider-roboter-sim-vastai:latest \
    lucam03/projekt-humanoider-roboter-sim-standalone:latest

# c) oder übergangsweise beim alten Tag bleiben — in .env.local:
: "${RL_IMAGE:=lucam03/projekt-humanoider-roboter-sim-vastai:latest}"
```

Option c) ist nur ein Übergang, kein Dauerzustand — `--vastai` existiert bei
[`update_sim_image.sh`](../Simulation/update_sim_image.sh) nur noch als deprecated Alias für
`--standalone`.

---

## 3. Fremdnutzung — Repo klonen und in Betrieb nehmen

Alles Nötige ist öffentlich: das Repo, beide Submodule
(`lucam06/Isaac-GR00T`, `unitreerobotics/unitree_ros`), beide Docker-Images
(`lucam03/projekt-humanoider-roboter`, `…-sim-standalone`) sowie Modell und Datensatz auf
HuggingFace (`nvidia/GR00T-N1.6-3B` ist **nicht** gated). Mitbringen muss man nur einen
**eigenen HF-Token**, optional einen W&B-Key und passende GPU-Hardware.

```bash
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
```

Der Ordnername und der Ort sind frei wählbar — alle Skripte lösen ihre eigene Position über
`BASH_SOURCE` auf. Ein `--recurse-submodules` ist für das Training **nicht** nötig: Das
Dockerfile klont den GR00T-Fork selbst auf einen gepinnten Commit.

### 3.1 Eigener GPU-Server (Docker) — Sim, Greif-Diagnose, RL

```bash
cp .env.local.example .env.local
# in .env.local eintragen (die := -Form beibehalten!):
#   : "${RL_HOST_DATA_DIR:=/pfad/mit/mindestens/60GB}"
#   : "${HF_TOKEN:=hf_…}"
./Simulation/server_rl_run.sh preflight     # prüft Image + GPU, braucht keinen Token
./Simulation/server_rl_run.sh view          # erstes Bild, ohne Modell/Checkpoint
```

GPU-Anforderung: **Ampere+ mit RT-Cores** (L40, RTX 4090/5090, A6000, RTX PRO 6000). A100/H100
haben keine RT-Cores und können die Kameras nicht rendern — Details in
[simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md).

### 3.2 Anderes KISSKI-Projekt

Ein Knopf schaltet alle abgeleiteten Pfade um; er wirkt beim Absenden, weil in allen sechs
Skripten `#SBATCH --export=ALL` steht und SLURM damit die Submit-Umgebung in den Job reicht:

```bash
# einmalig auf dem Login-Knoten
module load apptainer && mkdir -p "$HOME/images"
apptainer pull "$HOME/images/projekt-humanoider-roboter.sif" \
    docker://lucam03/projekt-humanoider-roboter:latest
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git \
    /mnt/vast-kisski/projects/<dein-projekt>/repo
cd /mnt/vast-kisski/projects/<dein-projekt>/repo && mkdir -p logs

# bei jedem Absenden (oder einmal in ~/.bashrc exportieren)
export KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<dein-projekt>
export HF_TOKEN=hf_…
sbatch Training/kisski_submit.sh
```

Erwartete Ableitung aus `KISSKI_PROJECT_DIR`:

| Variable | Wert |
|---|---|
| `DATA_DIR` | `$KISSKI_PROJECT_DIR/data` |
| `REPO_DIR` | `$KISSKI_PROJECT_DIR/repo` |
| `GROOT_FORK_DIR` | `$KISSKI_PROJECT_DIR/repo-groot` |
| `ASSETS_DIR` | `$KISSKI_PROJECT_DIR/assets` |
| `ISAAC_CACHE` | `$KISSKI_PROJECT_DIR/isaac-cache` |
| `APPTAINER_CACHEDIR` / `_TMPDIR` | `$KISSKI_PROJECT_DIR/apptainer-cache` / `-tmp` |
| `SIM_CODE` | `$REPO_DIR/Simulation` |

Jede davon lässt sich weiterhin einzeln überschreiben und gewinnt dann gegen die Ableitung.

### 3.3 Anderer HPC-Cluster (kein KISSKI)

`KISSKI_PROJECT_DIR` auf einen beliebigen beschreibbaren Projektpfad setzen und die
`#SBATCH`-Zeilen (Partition, GPU-Typ, Walltime) an den fremden Scheduler anpassen — das ist
der einzige Teil, den Umgebungsvariablen prinzipiell nicht erreichen.

---

## 4. Referenz — alle Portabilitäts-Knöpfe

### `Simulation/server_rl_run.sh` und `server_robocasa_ref_run.sh` (Docker)

Das RoboCasa-Skript nutzt dieselbe Mechanik mit `RC_`-Präfix (`RC_HOST_DATA_DIR` → `$HOME/groot-robocasa-data`, `RC_REPO_DIR`, `RC_IMAGE`, `RC_CONTAINER`, `RC_GPUS`).

| Variable | Default | Zweck |
|---|---|---|
| `RL_HOST_DATA_DIR` | `$HOME/groot-rl-data` | Host-Verzeichnis, das im Container `/data` wird (Checkpoints, Shader-Cache, Logs). Rechne mit 40–60 GB |
| `RL_REPO_DIR` | Elternverzeichnis des Skripts | Repo-Wurzel; bestimmt auch, wo `.env.local` gesucht wird |
| `RL_IMAGE` | `lucam03/…-sim-standalone:latest` | Abweichendes/lokal gebautes Image |
| `RL_CONTAINER` | `groot-rl` | Container-Name (mehrere Läufe pro Server) |
| `RL_GPUS` | `"device=1,0"` | GPU-Auswahl; erste Karte trägt Rendering + Training |
| `HF_TOKEN`, `WANDB_API_KEY` | — | Zugangsdaten; gehören in `.env.local` |

### `Training/setup_and_train_dockerhub_pull.sh` (Docker, Training)

| Variable | Default | Zweck |
|---|---|---|
| `TRAIN_HOST_DATA_DIR` | — (kein Mount) | Host-Verzeichnis, das im Container `/data` wird. Leer = altes vast.ai-Verhalten, alles lebt im Container und stirbt mit `--destroy`. Auf einem Rechner, der **auch die Sim fährt**, gehört hier `$HOME/groot-rl-data` hin — derselbe Pfad, den `server_rl_run.sh` einhängt; sonst sieht die Sim-Eval die Checkpoints nur nach einem `docker cp` über hunderte GB |
| `MOUNT_SCRIPTS` | `auto` | `auto` bindet `Training/scripts` über `/scripts`, sobald das Repo daneben liegt — genau wie [`kisski_submit.sh:336`](../Training/kisski_submit.sh#L336) es seit jeher tut. `0` = der Stand im Image gilt |
| `CONTAINER_NAME` | `groot-train` | mehrere Läufe pro Server |
| `DOCKER_GPUS` | `all` | Wert für `docker run --gpus`, z. B. `'"device=0"'` |

> **Warum `MOUNT_SCRIPTS` per Default an ist.** Am 2026-09-08 stellte sich heraus, dass
> `lucam03/projekt-humanoider-roboter:latest` ein `/scripts` von **vor dem 2026-06-03** trug:
> ohne `run_finetuning_cotrain.sh`, ohne `lib_split.sh`, ohne `torchrun`. `USE_COTRAIN=1` und
> `TRAIN_TEST_SPLIT=1` liefen darüber still ins Leere, `NUM_GPUS=2` landete in `DataParallel`
> statt unter `torchrun` und riss das Training ab. Auf KISSKI war das nie sichtbar, weil dort
> der Bind schon immer gesetzt war. Der Mount macht den Docker-Weg fehlerfrei-gleich —
> er ersetzt aber **keinen Rebuild**: liegt auch `/app/Groot-1.6` im Image falsch, hilft nur
> [`Training/update_image.sh`](../Training/update_image.sh), und auf vast.ai gibt es gar kein
> Repo zum Einhängen.

### KISSKI-SLURM-Skripte

| Variable | Default | Zweck |
|---|---|---|
| `KISSKI_PROJECT_DIR` | `/mnt/vast-kisski/projects/kisski-humrob` | **Hauptknopf** — alle Projektpfade leiten sich hieraus ab |
| `KISSKI_SIF_DIR` | `$HOME/images` | Wo die `.sif`-Dateien liegen (erster Kandidat) |
| `SIF_IMAGE` / `SERVER_SIF` / `SIM_SIF` | Kandidatenliste | einzelnes SIF hart setzen |
| `DATA_DIR`, `REPO_DIR`, `GROOT_FORK_DIR`, `ASSETS_DIR`, `ISAAC_CACHE`, `SIM_CODE` | aus `KISSKI_PROJECT_DIR` | einzeln überschreibbar |
| `APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR` | aus `KISSKI_PROJECT_DIR` | werden jetzt respektiert, wenn gesetzt |

Fachliche Parameter (`MAX_STEPS`, `GLOBAL_BATCH_SIZE`, `NUM_GPUS`, `TUNE_VISUAL`, …) stehen
unverändert in [training/env-vars.md](training/env-vars.md) — dieses Dokument behandelt nur
Pfade und Rechner-Bindung.

---

## 5. Was bewusst hart bleibt

Nicht jeder absolute Pfad ist ein Portabilitätsproblem:

- **`/data/…`, `/workspace/…`, `/app/Groot-1.6` in Python-Skripten** sind *container-intern*.
  Sie hängen nicht am Host-Layout und bleiben unverändert.
- **Image-Namen** (`lucam03/…`) und **HF-Repos** (`luca-mue/groot-g1dex3-checkpoint`) sind
  öffentlich und funktionieren für alle. Wer eigene Artefakte nutzt, setzt `RL_IMAGE` bzw.
  `HF_CHECKPOINT_REPO`.
- **`#SBATCH`-Ressourcenzeilen** (Partition, GPU, Walltime) sind cluster-spezifisch und
  müssen für einen fremden Scheduler von Hand angepasst werden.
- **Der Portable-Block ist in allen sechs SLURM-Skripten dupliziert** statt in eine
  gemeinsame Datei ausgelagert. Das ist Absicht: SLURM kopiert das Batch-Skript vor der
  Ausführung in sein Spool-Verzeichnis, weshalb `$0` und `BASH_SOURCE` im Job nicht mehr ins
  Repo zeigen — ein `source "$(dirname "$0")/lib_…"` schlüge fehl.

---

## 6. Selbsttest

Ohne Cluster und ohne GPU prüfbar:

```bash
# 1) Syntax aller angefassten Skripte
for f in Simulation/server_*.sh Training/kisski_*.sh Simulation/kisski_*.sh; do
    bash -n "$f" && echo "OK  $f"
done

# 2) Der Guard greift und bricht VOR jedem docker-Aufruf ab
RL_HOST_DATA_DIR=/proc/unmoeglich ./Simulation/server_rl_run.sh preflight   # -> exit 1 + Anleitung

# 3) Keine rechnergebundenen Pfade mehr im Code (Treffer nur in Kommentaren/Logs erwartet)
grep -rn "luca\.muecke\|/home/lmuecke" Training/ Simulation/ --include='*.sh'
```
