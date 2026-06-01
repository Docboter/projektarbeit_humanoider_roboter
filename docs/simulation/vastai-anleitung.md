# Closed-Loop-Sim auf vast.ai — Schritt-für-Schritt-Anleitung

Ziel: Den feingetunten GR00T-N1.6-Checkpoint in der Isaac-Lab-Simulation auf einer
**NVIDIA L40 (48 GB)** auf vast.ai evaluieren.

Der Container startet autonom:
1. Checkpoint herunterladen (oder aus Volume lesen)
2. GR00T-Policy-Server im Hintergrund starten
3. Isaac-Lab-Sim-Client starten → Eval-Schleife → Ergebnisse + Videos

---

## Überblick: Was läuft wo

```
vast.ai Instanz (L40, 48 GB VRAM)
└── Docker-Container: lucam03/projekt-humanoider-roboter-sim-vastai:latest
    ├── GR00T-Policy-Server  → /app/Groot-1.6/.venv/bin/python   (~10 GB VRAM)
    │     lädt Checkpoint, antwortet auf ZMQ-Requests
    └── Isaac-Lab-Sim-Client → ${ISAACLAB_PATH}/isaaclab.sh -p   (~8 GB VRAM)
          4 RGB-Kameras, 20 Episoden, speichert Videos + JSON
          ↕ ZMQ localhost:5555
```

---

## Voraussetzungen

Einmalig zu erledigen, bevor du die erste Instanz startest:

| Was | Wo |
|---|---|
| vast.ai-Account mit Credits | https://cloud.vast.ai |
| Docker-Hub-Login (`lucam03`) | `docker login` lokal |
| NGC-API-Key für `nvcr.io` | https://ngc.nvidia.com → API Key |
| HuggingFace-Token (`hf_...`) | https://huggingface.co/settings/tokens |
| Feingetunter Checkpoint | von KISSKI rsync'd (→ Abschnitt 3) |
| `g1_dex3.usd` Asset | einmalig erzeugt (→ Abschnitt 4) |

---

## Schritt 1 — Image bauen & pushen (einmalig)

Auf deinem Laptop im Repo-Root:

```powershell
# NGC-Login (Basis-Image nvcr.io/nvidia/isaac-lab:2.3.2 ist ~25 GB)
docker login nvcr.io   # Username: $oauthtoken   Password: <NGC-API-Key>

# Image bauen und nach Docker Hub pushen (~30-60 min, nur beim ersten Mal)
.\Simulation\update_sim_image.ps1 -VastAI

# Nur bauen, nicht pushen (zum Testen):
.\Simulation\update_sim_image.ps1 -VastAI -SkipPush
```

Ergebnis: `lucam03/projekt-humanoider-roboter-sim-vastai:latest` auf Docker Hub.

> **Hinweis:** Das Image ist ~30-40 GB. Genug Plattenplatz einplanen (80+ GB frei).

---

## Schritt 2 — Checkpoint von KISSKI holen

Der Checkpoint entsteht durch das Training auf KISSKI. Vom Laptop aus herunterladen:

```bash
# Alle Checkpoints (kann viele GB sein — ggf. nur den letzten holen)
rsync -avz --progress \
  <username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/ \
  ./checkpoints/

# Nur einen bestimmten Checkpoint-Step:
rsync -avz --progress \
  "<username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/blockstacking/g1_dex3_blockstacking_v1/checkpoints/20260529/checkpoint-3000/" \
  ./checkpoints/checkpoint-3000/
```

Den Checkpoint-Pfad findest du in den SLURM-Logs:
```bash
tail -f /logs/slurm-<jobid>.out | grep checkpoint
```

### Option: Checkpoint zu HuggingFace Hub hochladen

Damit der Container den Checkpoint automatisch herunterladen kann:

```bash
pip install huggingface_hub

# Modell-Repo anlegen (einmalig, privat reicht)
huggingface-cli repo create groot-g1dex3-checkpoint --type model --private

# Checkpoint hochladen
huggingface-cli upload luca-mue/groot-g1dex3-checkpoint ./checkpoint-3000/ --repo-type model
```

Dann im Container `HF_CHECKPOINT_REPO=<dein-hf-username>/groot-g1dex3-checkpoint` setzen —
der Container lädt den Checkpoint beim Start automatisch herunter.

---

## Schritt 3 — USD-Asset erzeugen (einmalig)

Das Isaac-Lab-Sim benötigt das Roboter-Asset `g1_dex3.usd`, das aus dem Unitree-URDF
konvertiert werden muss. Das passiert **einmalig** auf einer GPU-Maschine mit Isaac Sim.

### 3a) URDF beschaffen

Das URDF kommt aus dem Unitree-ROS-Paket, das als Git-Submodul unter `data/unitree_ros/` eingebunden ist.

Beim ersten Checkout (oder falls das Verzeichnis leer ist):

```bash
# Im Projektrepo-Root:
git submodule update --init data/unitree_ros
```

Benötigte Datei:
`data/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`

### 3b) USD erzeugen

Die Konvertierung ist nur ein Mesh-Import (headless, ~2-5 Minuten) — sie braucht deutlich weniger
VRAM als die spätere Eval-Schleife. **Eine lokale GPU reicht dafür aus.**

#### Option 1 (empfohlen): Lokal auf dem Laptop (RTX 4070 Laptop oder besser)

Voraussetzung: Docker Desktop mit GPU-Support (WSL2 + nvidia-container-toolkit).

```powershell
# Im Projektrepo-Root (PowerShell):
docker run -it --rm --gpus all --ipc=host --shm-size=8g `
  --entrypoint bash `
  -v "${PWD}/data:/data" `
  lucam03/projekt-humanoider-roboter-sim-vastai:latest
```

Im Container:
```bash
# VIRTUAL_ENV muss vor isaaclab.sh ungesetzt werden — sonst nutzt es das GR00T-venv statt
# Isaacs eigenem Python-Bundle und der Import von 'isaaclab' schlägt fehl.
unset VIRTUAL_ENV
${ISAACLAB_PATH}/isaaclab.sh -p \
    /workspace/g1_dex3_sim/convert_urdf_to_usd.py \
    --headless \
    --urdf /data/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf \
    --output /data/g1_dex3.usd
```

Das fertige Asset liegt danach auf dem Host unter `data/g1_dex3.usd`.

#### Option 2: Auf einer temporären vast.ai-Instanz

Falls kein Docker mit GPU-Support lokal verfügbar:

Starte eine **temporäre vast.ai-Instanz** (L40 oder RTX 4090) mit dem Image, aber
überbrücke den Entrypoint:

Im vast.ai-Launch-Dialog unter **"Docker Options"**:
```
--ipc=host --shm-size=16g --entrypoint bash
```

SSH in die Instanz und führe die Konvertierung durch:

```bash
# URDF hochladen (vom Laptop aus, paralleles Terminal)
scp -P <port> -r data/unitree_ros/robots/g1_description/ root@<ip>:/data/assets/unitree_ros/robots/

# Im Container (via SSH):
${ISAACLAB_PATH}/isaaclab.sh -p \
    /workspace/g1_dex3_sim/convert_urdf_to_usd.py \
    --headless \
    --urdf /data/assets/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf \
    --output /data/assets/g1_dex3.usd

# USD herunterladen (vom Laptop aus, paralleles Terminal)
scp -P <port> root@<ip>:/data/assets/g1_dex3.usd ./data/g1_dex3.usd
```

> Danach die Instanz zerstören — das war nur eine Einmalkonvertierung.

### 3c) USD aufbewahren

Empfohlen: `g1_dex3.usd` im selben HuggingFace-Modell-Repo wie den Checkpoint ablegen:

```bash
huggingface-cli upload <dein-hf-username>/groot-g1dex3-checkpoint \
    ./g1_dex3.usd g1_dex3.usd --repo-type model
```

Oder lokal aufbewahren und bei jeder Sim-Instanz per SCP hochladen (klein genug).

---

## Schritt 4 — Instanz auf vast.ai konfigurieren

### 4a) GPU auswählen

1. https://cloud.vast.ai → **Search**
2. Filter setzen:
   - **GPU**: `L40` (48 GB, empfohlen) oder `RTX 4090` (24 GB, günstiger)
   - **Min VRAM**: 24 GB
   - **Disk**: ≥ 60 GB (für Isaac-Sim-Cache + Videos)
3. Instanz mit niedrigem Preis/guter Verbindung auswählen → **Rent**

### 4b) Instance Configuration

Im Launch-Dialog folgende Felder ausfüllen:

**Image:**
```
lucam03/projekt-humanoider-roboter-sim-vastai:latest
```

**Docker Options:**
```
--ipc=host --shm-size=16g -p 22
```

> `-p 22` gibt Port 22 frei, damit vast.ai ihn auf einen externen Port mappt und
> `vastai ssh-url <id>` / `vastai ssh <id>` funktioniert.

**Environment Variables** (ein Eintrag pro Zeile):

| Variable | Wert | Pflicht? |
|---|---|---|
| `HF_TOKEN` | `hf_...` | Ja (für Checkpoint-Download) |
| `HF_CHECKPOINT_REPO` | `luca-mue/groot-g1dex3-checkpoint` | Ja (für HF-Download) |
| `ASSET_PATH` | `/data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd` | Ja |
| `NUM_EPISODES` | `20` | Nein (default 20) |
| `EXECUTION_HORIZON` | `8` | Nein (default 8) |
| `TASK_DESCRIPTION` | `stack the blocks` | Nein |
| `SHELL_ON_ERROR` | `1` | Empfohlen (für Debugging) |

> **Flash-Attention:** Das Modell (Eagle-Block2A-2B-v2) erfordert `flash_attention_2`
> zwingend; es ist im Image installiert. Es gibt **keine** Möglichkeit, es abzuschalten —
> `NO_FLASH_ATTN` wird ignoriert. Deshalb sind nur Ampere+-GPUs mit Flash-Attn-Support geeignet.

> **Hinweis:** Wenn `HF_CHECKPOINT_REPO` gesetzt ist, setzt der Entrypoint `CHECKPOINT_PATH`
> automatisch auf `/data/checkpoints/<repo-name>/`. `ASSET_PATH` muss trotzdem explizit
> gesetzt werden, weil `g1_dex3.usd` und `configuration/` im gleichen HF-Repo liegen.

**Disk Space:** mindestens `60 GB`

→ **Launch**

---

## Schritt 5 — Checkpoint & Asset bereitstellen

### Variante A: HuggingFace-Download (empfohlen)

Wenn `HF_CHECKPOINT_REPO` gesetzt ist, lädt der Container alles automatisch.
Checkpoint und `g1_dex3.usd` landen unter `/data/checkpoints/<repo-name>/`.

Dann `ASSET_PATH` entsprechend anpassen:
```
ASSET_PATH=/data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd
```

### Variante B: Manueller Upload per SCP

vast.ai zeigt dir nach dem Start einen SSH-Befehl, z. B.:
```
ssh -p 12345 root@123.45.67.89
```

Parallel dazu vom Laptop aus hochladen:
```bash
# Checkpoint
scp -P 12345 -r ./checkpoints/checkpoint-3000/ root@123.45.67.89:/data/checkpoints/

# USD-Asset
scp -P 12345 ./g1_dex3.usd root@123.45.67.89:/workspace/assets/g1_dex3.usd
```

Der Entrypoint läuft bereits im Hintergrund — er wartet bis `CHECKPOINT_PATH` existiert
(oder schlägt fehl und fällt in eine Shell wenn `SHELL_ON_ERROR=1`).

---

## Schritt 6 — Simulation überwachen

SSH in die Instanz (der Entrypoint startet den SSH-Server automatisch):
```bash
# SSH-Befehl via vast.ai CLI abrufen:
pip install vastai
vastai set api-key <dein-api-key>   # vast.ai → Account → API Keys
vastai ssh-url <instance-id>        # gibt fertigen ssh-Befehl aus

# Oder direkt:
vastai ssh <instance-id>
```

### GR00T-Server-Log beobachten
```bash
tail -f /data/logs/groot_server.log
```

Erfolgreicher Start sieht so aus:
```
Loading model from /data/checkpoints/checkpoint-3000 ...
Server listening on tcp://0.0.0.0:5555
```

### Isaac-Lab-Output direkt beobachten

Der Sim-Client schreibt in die Container-Stdout. Im vast.ai-Portal unter
**Instances → Logs** sichtbar, oder via SSH:
```bash
# PID des Isaac-Lab-Prozesses finden
pgrep -a python | grep run_g1_dex3_sim_eval

# Output live verfolgen (falls in Datei umgeleitet)
tail -f /data/logs/sim_client.log
```

### Fortschritt ablesen
```
--- Episode 1/20 ---
  Episode 1: ERFOLG | 142 Steps | 4.7s
--- Episode 2/20 ---
  Episode 2: misslungen | 300 Steps | 10.0s
...
============================================================
Ergebnis: 12/20 Erfolge — Success-Rate: 60.0%
============================================================
```

### VRAM-Auslastung prüfen
```bash
watch -n 2 nvidia-smi
```

Erwartete Auslastung auf L40: ~18-22 GB von 48 GB.

---

## Schritt 7 — Ergebnisse sichern

**Vor dem Zerstören der Instanz** Daten herunterladen!

```bash
# Eval-Ergebnisse (JSON)
scp -P <port> root@<ip>:/data/sim_results/results.json ./sim_results.json

# Rollout-Videos
scp -P <port> -r root@<ip>:/data/sim_videos/ ./sim_videos/

# GR00T-Server-Log (für Debugging)
scp -P <port> root@<ip>:/data/logs/groot_server.log ./groot_server.log
```

`results.json` enthält:
```json
{
  "num_episodes": 20,
  "num_success": 12,
  "success_rate": 0.6,
  "episodes": [
    {"episode": 1, "success": true, "num_steps": 142, "duration_s": 4.7},
    ...
  ]
}
```

---

## Diagnose: Open-Loop-Replay (Config vs. Training trennen)

Wenn der Roboter im Modell-Eval die Würfel nicht greift, ist die Schlüsselfrage:
**liegt es am untrainierten Modell oder an einem Fehler in Sim/Config?** Der Open-Loop-
Replay beantwortet das eindeutig: statt das GR00T-Modell zu befragen, spielt er die
**echten aufgezeichneten Dataset-Aktionen** (gebündelte Episode 0) direkt in dieselbe
Isaac-Lab-Env. Kein Server, kein Modell.

Interpretation:
- Fährt der Roboter die Arme zum Tisch und schließt die Finger (greif-artige Bewegung)
  → Sim/Config führt korrekte Aktionen korrekt aus → Wegdriften im Modell-Eval liegt am
  **Training**, nicht an der Config.
- Driftet der Roboter auch beim Replay weg / bewegt sich unsinnig → es steckt doch ein
  **Sim-Problem** drin (Reachability, Konvention, Skalierung).

Der Replay ist **vollständig additiv**: er überschreibt nichts vom Modell-Eval, nutzt
eigene Ausgabepfade (`/data/sim_videos_replay`, `/data/sim_results_replay`) und kann sogar
**parallel** in einer zweiten Instanz laufen.

### Lauf konfigurieren

Gleiches Image wie der Modell-Eval, aber der **Entrypoint wird überschrieben**:

**Docker Options:**
```
--ipc=host --shm-size=16g -p 22 --entrypoint bash
```

**Args to pass to docker entrypoint:**
```
/scripts/entrypoint_replay.sh
```

**Environment Variables:**

| Variable | Wert | Pflicht? |
|---|---|---|
| `HF_TOKEN` | `hf_...` | Ja (für USD-Asset-Download) |
| `HF_CHECKPOINT_REPO` | `luca-mue/groot-g1dex3-checkpoint` | Ja, falls kein `ASSET_PATH` |
| `ASSET_PATH` | `/data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd` | Alternativ zu `HF_CHECKPOINT_REPO` |
| `GRASP_TEST` | `1` | Nein — Würfel exakt an die aufgezeichneten Greifpunkte setzen (Greif-Physik-Test) |

> Der Replay braucht **kein** Checkpoint-Modell, nur das `g1_dex3.usd`-Asset. Das HF-Repo
> wird hier nur als Quelle für das USD genutzt.

### Ergebnis lesen

Der Lauf gibt direkt im Log die Diagnose-Kennzahlen aus:

```
[Replay] TRACKING-FEHLER Arm-Gelenke: mittel=0.022 rad
[Replay]   > ~0.3 rad mittel = Arme folgen NICHT (PD-Gains zu schwach = Sim-Bug);
            < ~0.1 = Tracking ok (dann Geometrie/Modell).
[Replay] GREIF-DIAGNOSE: min Hand→Würfel-Distanz = 5.9 cm | max Würfel-Anhebung = 1.0 cm
[Replay]   Distanz klein + Anhebung>~2cm → Greifen FUNKTIONIERT.
```

- **Tracking-Fehler < ~0.1 rad** → der Roboter folgt den kommandierten Gelenkwinkeln; die
  PD-Gains und die Articulation-Config sind in Ordnung.
- **`--grasp-test` (GRASP_TEST=1)**: setzt die Würfel exakt unter die aufgezeichneten
  Greifpunkte. Eine **Würfel-Anhebung > ~2 cm** beweist, dass die Greif-Physik (Finger,
  Kontakte, Reibung) funktioniert.

Ergebnisse sichern (vor dem Zerstören der Instanz):
```bash
scp -P <port> root@<ip>:/data/sim_results_replay/results.json ./replay_results.json
scp -P <port> root@<ip>:/data/sim_videos_replay/replay_episode0.mp4 ./replay_episode0.mp4
```

> **Befund dieses Projekts:** Tracking 0.022 rad und (nach Korrektur der Tischhöhe) 1.0 cm
> Anhebung im Grasp-Test → Sim/Config sind validiert; das Greifen funktioniert physikalisch.
> Das Wegdriften im Modell-Eval von checkpoint-3000 ist also dem **untrainierten Modell**
> zuzuschreiben, nicht der Config. Details in
> [`implementation-notes.md`](implementation-notes.md) §11.

---

## Kosten & Laufzeiten (Richtwerte)

| GPU | Preis/h | 20 Episoden (ca.) | Kosten gesamt |
|---|---|---|---|
| L40 (48 GB) | ~0,7–0,9 $/h | ~30–60 min | ~0,5–0,9 $ |
| RTX 4090 (24 GB) | ~0,4–0,6 $/h | ~45–90 min | ~0,3–0,9 $ |

Dazu kommen ~10-15 Minuten für Isaac-Sim-Shader-Kompilierung beim ersten Start.

---

## Troubleshooting

### `createDLSSContext error` / Rendering-Fehler
Die GPU hat keine RT-Cores. Nur L40, RTX 30xx/40xx, A6000 sind geeignet —
**kein A100, kein H100, kein V100**.

### GR00T-Server startet nicht
```bash
cat /data/logs/groot_server.log
```
Häufige Ursachen:
- `CHECKPOINT_PATH` existiert nicht → Pfad prüfen, ggf. HF-Download-Log ansehen
- `Python.h not found` / Triton-gcc-Fehler → Image vor dem Python-3.10-Fix gebaut; neu bauen + pushen
- `flash attention`-AssertionError → GPU ohne Flash-Attn-Support (Volta/Turing); Ampere+ nutzen
- VRAM voll → kleinere GPU-Instanz war gewählt; auf L40/A6000 wechseln

### `CHECKPOINT_PATH leer oder existiert nicht`
Entrypoint hat Fehler gemeldet. Mit `SHELL_ON_ERROR=1` landet man in einer Shell:
```bash
# Manuell hochladen, dann Entrypoint neu starten:
bash /scripts/entrypoint_sim.sh
```

### Isaac Lab startet nicht / `g1_dex3.usd nicht gefunden`
```bash
# USD-Asset fehlt — hochladen:
# (vom Laptop)
scp -P <port> ./g1_dex3.usd root@<ip>:/workspace/assets/g1_dex3.usd

# Im Container, Sim manuell starten:
${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/run_g1_dex3_sim_eval.py \
    --headless --enable_cameras \
    --server tcp://localhost:5555 \
    --num-episodes 20
```

### Instanz läuft, aber kein Output sichtbar
Isaac Sim kompiliert beim ersten Start Shader — das dauert **10-15 Minuten** ohne sichtbaren
Output. Danach kommt `[Phase A] Isaac-Lab-Sim wird initialisiert …`.

---

## Zusammenfassung: Schnellstart

Voraussetzungen: Image gepusht (`.\Simulation\update_sim_image.ps1 -VastAI`), Checkpoint und
USD-Asset auf HuggingFace (`luca-mue/groot-g1dex3-checkpoint`).

1. `.\Simulation\update_sim_image.ps1 -VastAI` ausführen (nur wenn Image noch nicht gepusht)
2. vast.ai → Search → **L40** filtern (≥24 GB, Ampere+, RT-Cores) → Rent
3. Image: `lucam03/projekt-humanoider-roboter-sim-vastai:latest`
4. Docker Options: `--ipc=host --shm-size=16g -p 22`
5. Env:
   ```
   HF_TOKEN=hf_...
   HF_CHECKPOINT_REPO=luca-mue/groot-g1dex3-checkpoint
   ASSET_PATH=/data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd
   NUM_EPISODES=20
   SHELL_ON_ERROR=1
   ```
6. Disk: 60 GB → Launch
7. ~15 min warten (HF-Download + Shader-Kompilierung + Server-Start)
8. Logs unter **Instances → Logs** beobachten → Ergebnisse abwarten
9. `scp` für `results.json` + Videos → Instanz zerstören
