# Projektarbeit Humanoider Roboter

Dieses Repo ist die Integrationsschicht, um NVIDIA GR00T mit der Unitree Isaac-Sim-Umgebung zu testen. Ziel ist, einen Unitree G1/G129 mit DEX3-Hand in Isaac Sim per Textprompt zu steuern, zum Beispiel: `pick up the cylinder`.

Die beiden Basis-Repos liegen als Submodules unter `repos/`:

- `repos/Isaac-GR00T`: NVIDIA GR00T Fork mit Server-Fix.
- `repos/unitree_sim_isaaclab`: Unitree IsaacLab Fork mit GR00T ActionProvider.

Der wichtige Ablauf ist:

1. GR00T laedt den Checkpoint im `groot-server` Container.
2. Isaac Sim laeuft im `unitree-sim` Container.
3. Der `GrootActionProvider` liest Kamera-, Joint-State- und Textdaten aus Isaac Sim.
4. Der Provider sendet die Observation per ZeroMQ/MsgPack an den GR00T PolicyServer.
5. Die zurueckkommenden Actions werden zuerst im Dry-Run geloggt und spaeter als Joint-Ziele in Isaac Sim geschrieben.

Mehr technische Details fuer Coding Agents stehen in [docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md).

## Projektaufbau

```text
projektarbeit_humanoider_roboter/
├── repos/
│   ├── Isaac-GR00T/
│   │   └── checkpoints/GR00T-N1.6-G1-PnPAppleToPlate/
│   └── unitree_sim_isaaclab/
├── docker/
│   ├── docker-compose.groot-unitree.yml
│   └── .env.groot-unitree.example
├── scripts/
│   ├── verify_layout.sh
│   └── run_module_checks.sh
└── docs/
    └── AGENT_HANDOFF.md
```

Docker-Services:

- `groot-server`: GR00T PolicyServer, idealerweise GPU 0.
- `unitree-sim`: Isaac Sim / Unitree Sim / GR00T ActionProvider, idealerweise GPU 1.

Die Container nutzen `network_mode: host`, damit die Sim den GR00T-Server ueber `127.0.0.1:5555` erreicht.

Fuer Vast.AI gibt es zusaetzlich ein experimentelles Single-Container-Setup
unter [docker/1-docker-setup](docker/1-docker-setup/README.md). Dieses Image
enthaelt beide Umgebungen, startet aber weiterhin zwei getrennte Prozesse:
GR00T PolicyServer und Isaac/Unitree Sim.

## Wichtig vorab

Der Checkpoint `GR00T-N1.6-G1-PnPAppleToPlate` ist ein HuggingFace/PyTorch-Checkpoint mit `.safetensors`. Das ist normal; es fehlt kein `.pt`-Modell.

Der Checkpoint wird nicht direkt in Isaac Sim geladen. Stattdessen laedt der GR00T PolicyServer den Checkpoint, und Isaac Sim fragt diesen Server ueber den `GrootActionProvider` ab.

GR00T funktioniert fuer diesen Checkpoint nicht sinnvoll auf CPU, weil FlashAttention CUDA braucht. Auf einer einzelnen RTX 5070 mit 12 GB VRAM passen GR00T und Isaac Sim nicht gemeinsam in den Speicher. Fuer den Institut-Test ist deshalb eine Workstation mit ausreichend VRAM sinnvoll, idealerweise mit zwei GPUs.

Immer zuerst mit Dry-Run testen:

```bash
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

## Voraussetzungen

Auf der Workstation:

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
git --version
git lfs version
```

Wenn der Docker-GPU-Test keine GPU zeigt, muss das NVIDIA Container Toolkit installiert oder konfiguriert werden.

## Setup

Repo mit Submodules klonen:

```bash
mkdir -p /home/<user>/projects
cd /home/<user>/projects

git clone --recurse-submodules https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
```

Falls ohne Submodules geklont wurde:

```bash
git submodule update --init --recursive
```

Checkpoint bereitstellen:

```text
/home/<user>/projects/projektarbeit_humanoider_roboter/repos/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

Die alten Geschwisterpfade unter `/home/<user>/projects/Isaac-GR00T/...` oder `/home/<user>/projects/unitree_sim_isaaclab/...` werden nach der Restrukturierung nicht mehr verwendet.

Konfiguration anlegen:

```bash
cp docker/.env.groot-unitree.example docker/.env.groot-unitree
```

Typische Konfiguration:

```bash
PROJECTS_DIR=..
GROOT_GPU=0
SIM_GPU=1
GROOT_PORT=5555
GROOT_HOST=127.0.0.1
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

`PROJECTS_DIR=..` ist korrekt, wenn `docker compose` aus dem Projektroot mit `-f docker/docker-compose.groot-unitree.yml` gestartet wird. Docker Compose loest relative Pfade relativ zur Compose-Datei im `docker/`-Ordner auf. Wenn Compose von woanders gestartet wird, `PROJECTS_DIR` auf den absoluten Projektroot setzen.

Layout pruefen:

```bash
./scripts/verify_layout.sh .
```

## Docker-Images bauen

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml build groot-server
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml build unitree-sim
```

Der erste Build dauert lange, weil GR00T und Isaac Sim grosse Dependencies haben.

## Testen

Die Tests sind absichtlich gestuft. Nicht direkt mit echten Roboterbewegungen starten.

### Test 0: Modulchecks ohne GPU

```bash
./scripts/run_module_checks.sh .
```

Erwartung:

```text
Module checks passed.
```

Dieser Test prueft Repo-Struktur, Python-Syntax der Integrationsdateien und die Docker-Compose-Konfiguration. Er startet weder GR00T noch Isaac Sim.

### Test 1: GR00T Server starten

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml up -d groot-server
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml logs -f groot-server
```

Erwartung:

```text
Server is ready and listening
```

### Test 2: GR00T isoliert abfragen

In einem zweiten Terminal:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml run --rm unitree-sim \
  conda run -n unitree_sim_env python3 tools/test_groot_server_action.py \
    --host 127.0.0.1 \
    --port 5555 \
    --timeout-ms 120000 \
    --prompt "pick up the cylinder"
```

Erwartung:

```text
GR00T server ping ok.
Action keys: [...]
```

Wenn dieser Test klappt, liefert das echte GR00T-Modell Actions. Wenn er fehlschlaegt, liegt das Problem noch nicht in Isaac Sim.

### Test 3: Isaac Sim im Dry-Run

In `docker/.env.groot-unitree`:

```bash
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

Dann:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml up unitree-sim
```

Wichtig: `unitree-sim` erst starten, nachdem `groot-server` im Log `Server is ready and listening` meldet.

Gute Logs:

```text
[GrootActionProvider] connected to GR00T server 127.0.0.1:5555
[GrootActionProvider][debug] observation prompt='...'
[GrootActionProvider][debug] GR00T action keys: [...]
[GrootActionProvider][debug] dry-run enabled: GR00T action is not applied
```

Dry-Run bedeutet: GR00T wird abgefragt, aber die Actions werden noch nicht auf den Roboter geschrieben.

### Prompt senden

In einem weiteren Terminal:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec unitree-sim \
  conda run -n unitree_sim_env python3 groot_prompt_terminal.py --fifo groot_prompt.pipe
```

Beispielprompts:

```text
pick up the cylinder
grasp the cube
pick up the block
```

Fallback ohne Prompt-Terminal:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec unitree-sim \
  bash -lc 'echo "pick up the cylinder" > /home/code/unitree_sim_isaaclab/groot_prompt.txt'
```

### Test 4: Echte Actions aktivieren

Erst aktivieren, wenn der Dry-Run Action-Keys liefert.

In `docker/.env.groot-unitree`:

```bash
UNITREE_EXTRA_ARGS=--groot_debug
```

Dann neu starten:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml up unitree-sim
```

Jetzt schreibt der ActionProvider die gemappten GR00T-Actions in Isaac Sim. Wenn der Roboter kippt oder stark springt, wieder auf `--groot_dry_run` zurueckgehen und die geloggten Action-Werte pruefen.

## Aktueller Stand des Action-Mappings

Der `UNITREE_G1`-Checkpoint erwartet:

- Video: `ego_view`
- State: `left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`, `left_hand`, `right_hand`
- Sprache: `annotation.human.task_description`
- Actions: `left_arm`, `right_arm`, `left_hand`, `right_hand`, `waist`, `base_height_command`, `navigate_command`

Aktuell angewendet werden:

- `left_arm`
- `right_arm`
- `waist`
- `left_hand`
- `right_hand`

`base_height_command` und `navigate_command` werden geloggt, aber noch nicht in IsaacLab-Locomotion-Control gemappt. Das ist ein wichtiger naechster Integrationspunkt, falls GR00T wirklich Whole-Body/Locomotion steuern soll.

## Nuetzliche Befehle

Logs:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml logs -f groot-server
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml logs -f unitree-sim
```

Stoppen:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml down
```

VRAM beobachten:

```bash
watch -n 0.5 nvidia-smi
```

Container-Shell:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec unitree-sim bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec groot-server bash
```

## Troubleshooting

- `flash_attn ... CPU backend`: GR00T wurde auf CPU gestartet. Der Checkpoint braucht CUDA/FlashAttention.
- `CUDA out of memory`: GR00T und Isaac sollten auf getrennten GPUs laufen oder die GPU muss genug VRAM haben.
- `Cannot connect to GR00T PolicyServer`: `groot-server` muss laufen und `Server is ready` melden.
- `No reader on FIFO`: Nicht kritisch. Das Prompt-Terminal schreibt dann in `groot_prompt.txt`.
- Sim startet, aber Roboter kippt: Zuerst Dry-Run verwenden. Danach Action-Keys und `applied_target_delta` Logs anschauen.
- Docker sieht keine GPU: NVIDIA Container Toolkit mit dem CUDA-Container-Test pruefen.
