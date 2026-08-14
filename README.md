# Projektarbeit Humanoider Roboter

Integrationsrepo fuer NVIDIA GR00T und die Unitree Isaac-Sim/IsaacLab-Umgebung.
Ziel ist, einen Unitree G1/G129 mit DEX3-Hand in Isaac Sim per Textprompt zu
steuern, zum Beispiel `pick up the cylinder`.

Dieses Root-README ist nur der Einstieg. Dauerhafte Agent-Anweisungen liegen im
Repository-Root; ausfuehrliche Anleitungen liegen zentral unter `docs/`:

- [Agent Instructions](AGENTS.md): kurze, automatisch geladene Regeln und
  Platzhalter zur Ausrichtung zukuenftiger Agents.
- [Project Context](docs/PROJECT_CONTEXT.md): Architektur,
  Implementierungsstand, technische Entscheidungen und Backlog.
- [Setup and Testing](docs/SETUP_AND_TESTING.md): normales Zwei-Service-Setup
  mit Docker Compose, Teststufen, Action-Mapping und Troubleshooting.
- [Remote Simulation](docs/REMOTE_SIMULATION.md): erster Meilenstein fuer den
  G1 vor Tisch und Wuerfeln auf einem IKR-Server mit WebRTC-Livebild.
- [Vast Single Container](docs/VAST_SINGLE_CONTAINER.md): experimentelles
  Ein-Container-Image fuer Vast.AI mit GR00T und Isaac/Unitree in getrennten
  Runtime-Prozessen.

## Projektbild

Die beiden Basis-Repos liegen als Submodules unter `repos/`:

- `repos/Isaac-GR00T`: NVIDIA GR00T Fork mit Server-Fix.
- `repos/unitree_sim_isaaclab`: Unitree IsaacLab Fork mit GR00T ActionProvider.

Der wichtige Ablauf:

1. GR00T laedt den Checkpoint im PolicyServer.
2. Isaac Sim erzeugt Kamera-, Joint-State- und Text-Observations.
3. `GrootActionProvider` sendet die Observation per ZeroMQ/MsgPack an GR00T.
4. GR00T liefert Actions zurueck.
5. Actions werden zuerst im Dry-Run geloggt und erst danach auf Isaac Sim
   angewendet.

## Schnellstart

```bash
git clone --recurse-submodules https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
cp docker/.env.groot-unitree.example docker/.env.groot-unitree
./scripts/verify_layout.sh .
./scripts/run_module_checks.sh .
```

Der Checkpoint wird erwartet unter:

```text
repos/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

Immer zuerst mit Dry-Run testen:

```bash
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

Fuer den aktuellen, GR00T-freien Simulations-Meilenstein:

```bash
./scripts/fetch_unitree_assets.sh .
cp docker/.env.simulation.example docker/.env.simulation
docker compose --env-file docker/.env.simulation \
  -f docker/docker-compose.simulation.yml up -d --build
```

## Struktur

```text
projektarbeit_humanoider_roboter/
|-- AGENTS.md
|-- docs/
|   |-- PROJECT_CONTEXT.md
|   |-- REMOTE_SIMULATION.md
|   |-- SETUP_AND_TESTING.md
|   `-- VAST_SINGLE_CONTAINER.md
|-- docker/
|   |-- docker-compose.simulation.yml
|   |-- docker-compose.groot-unitree.yml
|   |-- simulation/
|   |-- .env.groot-unitree.example
|   `-- 1-docker-setup/
|-- repos/
|   |-- Isaac-GR00T/
|   `-- unitree_sim_isaaclab/
`-- scripts/
    |-- verify_layout.sh
    `-- run_module_checks.sh
```
