# Projektarbeit Humanoider Roboter

Dieses Repo ist die Integrationsschicht fuer:

- NVIDIA Isaac-GR00T
- Unitree `unitree_sim_isaaclab`
- den GR00T-ActionProvider fuer Isaac Sim
- Docker/Compose-Setup fuer eine Workstation mit ausreichend VRAM

Ziel: Auf einer Instituts-Workstation soll nur dieses Repo geklont werden. Die beiden Basis-Repos liegen als Git-Submodules unter `repos/` und enthalten die benoetigten Integrationsaenderungen direkt in den Forks.

## Zielarchitektur

```text
projects/
└── projektarbeit_humanoider_roboter/
    ├── repos/
    │   ├── Isaac-GR00T/
    │   │   └── checkpoints/GR00T-N1.6-G1-PnPAppleToPlate/
    │   └── unitree_sim_isaaclab/
    ├── docker/
    │   ├── docker-compose.groot-unitree.yml
    │   └── .env.groot-unitree
    └── scripts/
```

Docker-Services:

- `groot-server`: GR00T PolicyServer, idealerweise GPU 0
- `unitree-sim`: Isaac Sim / Unitree Sim / GR00T ActionProvider, idealerweise GPU 1

Die beiden Container laufen mit `network_mode: host`. Dadurch kann die Sim den GR00T-Server ueber `127.0.0.1:5555` erreichen.

## Warum so?

Auf dem lokalen PC passten GR00T und Isaac Sim nicht gemeinsam in 12 GB VRAM. Gemessen wurden grob:

- GR00T allein: ca. 7.5 GB VRAM
- Isaac Sim allein: ca. 6-7.5 GB VRAM

Auf der Instituts-Workstation mit RTX PRO 6000 ist das Ziel deshalb:

- GR00T auf eine eigene GPU
- Isaac Sim auf eine eigene GPU
- erst Dry-Run mit Debug-Logs
- danach echte Actions aktivieren

## Enthaltene Submodules

- `repos/unitree_sim_isaaclab`: Fork mit GR00T ActionProvider, Prompt-Terminal, Testclient, `sim_main.py`-Flags und Minimal-Szene.
- `repos/Isaac-GR00T`: Fork mit GR00T-Server-Fix fuer DeepSpeed/CUDA_HOME und Docker-Build-Kontext.

Die eigentlichen Codeaenderungen liegen also in den Forks, nicht mehr als Overlay-Kopien in diesem Repo.

## Voraussetzungen auf der Workstation

Linux-Host empfohlen. Von Windows-Laptop aus am besten per SSH auf die Workstation gehen.

Auf der Workstation:

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Wenn der letzte Befehl keine GPU zeigt, NVIDIA Container Toolkit installieren/konfigurieren.

Außerdem sinnvoll:

```bash
git --version
git lfs version
```

## Repo klonen

Beispiel unter `/home/<user>/projects`:

```bash
mkdir -p /home/<user>/projects
cd /home/<user>/projects

git clone --recurse-submodules https://github.com/Docboter/projektarbeit_humanoider_roboter.git
```

Falls ohne `--recurse-submodules` geklont wurde:

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter
git submodule update --init --recursive
```

## Checkpoint bereitstellen

Der GR00T-Checkpoint muss hier liegen:

```text
/home/<user>/projects/projektarbeit_humanoider_roboter/repos/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

Die `.dockerignore` im GR00T-Fork verhindert, dass `checkpoints/` ins Docker-Image gebacken wird. Der Checkpoint wird stattdessen in den Container gemountet.

## Konfiguration

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter
cp docker/.env.groot-unitree.example docker/.env.groot-unitree
```

Layout pruefen:

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter
./scripts/verify_layout.sh .
```

## GPU-Zuweisung

In `docker/.env.groot-unitree`:

```bash
PROJECTS_DIR=.
GROOT_GPU=0
SIM_GPU=1
GROOT_PORT=5555
GROOT_HOST=127.0.0.1
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

Hinweis: In jedem Container sieht die freigegebene GPU intern als `cuda:0` aus. Deshalb startet GR00T im Container mit `--device cuda:0`, auch wenn auf dem Host `GROOT_GPU=1` gesetzt ist.

## Docker-Images bauen

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter

docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml build groot-server
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml build unitree-sim
```

Der erste Build dauert lange. GR00T und Isaac Sim haben große Dependencies.

## Test 1: GR00T Server starten

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml up -d groot-server
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml logs -f groot-server
```

Erwartung:

```text
Server is ready and listening
```

## Test 2: GR00T isoliert abfragen

In einem zweiten Terminal:

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter
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

Wenn das klappt, liefert das echte GR00T-Modell Actions.

## Test 3: Isaac Sim im Dry-Run

In `docker/.env.groot-unitree`:

```bash
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

Dann:

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml up unitree-sim
```

Wichtig: Starte `unitree-sim` erst, nachdem `groot-server` im Log `Server is ready and listening` meldet. `depends_on` startet den Container, wartet aber nicht auf die Modellinitialisierung.

Wichtige Logs:

```text
[GrootActionProvider] connected to GR00T server 127.0.0.1:5555
[GrootActionProvider] received prompt: ...
[GrootActionProvider][debug] observation prompt='...'
[GrootActionProvider][debug] GR00T action keys: [...]
[GrootActionProvider][debug] dry-run enabled: GR00T action is not applied
```

Dry-Run bedeutet: GR00T wird abgefragt, aber die Actions werden nicht auf den Roboter geschrieben. Das ist der sichere erste Integrationstest.

## Prompt senden

In einem weiteren Terminal:

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec unitree-sim \
  conda run -n unitree_sim_env python3 groot_prompt_terminal.py --fifo groot_prompt.pipe
```

Beispiel:

```text
pick up the cylinder
grasp the cube
pick up the block
```

Fallback ohne Terminal:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec unitree-sim \
  bash -lc 'echo "pick up the cylinder" > /home/code/unitree_sim_isaaclab/groot_prompt.txt'
```

## Test 4: Echte Actions aktivieren

Wenn der Dry-Run Action-Keys liefert, in `docker/.env.groot-unitree`:

```bash
UNITREE_EXTRA_ARGS=--groot_debug
```

Dann neu starten:

```bash
cd /home/<user>/projects/projektarbeit_humanoider_roboter
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml up unitree-sim
```

Jetzt schreibt der ActionProvider die gemappten GR00T-Actions in Isaac Sim.

## Aktueller Stand des Action-Mappings

Der Provider baut bereits eine GR00T-Observation mit:

- `video.ego_view`
- `state.left_leg`
- `state.right_leg`
- `state.waist`
- `state.left_arm`
- `state.right_arm`
- `state.left_hand`
- `state.right_hand`
- `language.annotation.human.task_description`

Aktuell angewendete GR00T-Action-Keys:

- `left_arm`
- `right_arm`
- `waist`
- `left_hand`
- `right_hand`

Wichtig: `base_height_command` und `navigate_command` werden geloggt, aber noch nicht in IsaacLab-Locomotion-Control gemappt. Genau das ist der naechste Integrationspunkt, wenn GR00T whole-body/locomotion wirklich stabil steuern soll.

## Minimal-Szene

Die Compose-Sim startet standardmaessig mit:

```bash
--minimal_scene
--camera_width 320
--camera_height 240
```

Dadurch werden schwere Warehouse-Assets entfernt und der USD-Tisch durch einen einfachen Cuboid-Tisch ersetzt. Uebrig bleiben:

- Roboter
- Boden
- einfacher Tisch
- Zylinder
- Licht
- Frontkamera

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

- `flash_attn ... CPU backend`: GR00T wurde auf CPU gestartet. Der Checkpoint braucht CUDA/FlashAttention. Im Container `--device cuda:0` verwenden.
- `CUDA out of memory`: GPU-Zuweisung pruefen. GR00T und Isaac sollten auf getrennten GPUs laufen oder die GPU muss genug VRAM haben.
- `Cannot connect to GR00T PolicyServer`: `groot-server` muss laufen und `Server is ready` melden. Mit Host-Netzwerk ist `127.0.0.1:5555` korrekt.
- `No reader on FIFO`: Nicht kritisch. Das Prompt-Terminal schreibt dann in `groot_prompt.txt`, der Provider liest die Datei beim naechsten Step.
- Sim startet, aber Roboter kippt: Zuerst Dry-Run verwenden. Danach Action-Keys und `applied_target_delta` Logs anschauen.
- Docker sieht keine GPU: NVIDIA Container Toolkit testen mit `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`.

## Was auf der Workstation getestet werden soll

1. GR00T-Server startet auf GPU.
2. Dummy-Observation liefert Action-Keys.
3. Isaac Sim verbindet sich mit GR00T.
4. Prompt kommt im ActionProvider an.
5. Dry-Run zeigt GR00T-Action-Keys.
6. Ohne Dry-Run sieht man in Isaac Sim, welche Bewegungen der Roboter bei Befehlen wie `pick up the cylinder` oder `grasp the cube` macht.
