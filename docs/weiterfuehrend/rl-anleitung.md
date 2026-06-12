# RL-Fine-tuning (FPO) auf vast.ai — Schritt-für-Schritt-Anleitung

Ziel: Den feingetunten GR00T-N1.6-**BC-Checkpoint** per **Reinforcement Learning (FPO)** in der
Isaac-Lab-Block-Stacking-Sim weiter verfeinern — auf einer **RT-Core-GPU** (L40 / RTX 4090 / A6000)
auf vast.ai. RL trainiert die Policy **in genau der Sim**, in der sie auch evaluiert wird, und
optimiert direkt auf **Aufgaben-Erfolg** statt nur Aktions-Nachahmung (Hintergrund:
[reinforcement-learning-plan.md](reinforcement-learning-plan.md)).

> ## ⚠️ Status: erste Implementierung — beim ersten Lauf zu verifizieren
>
> Der RL-Pfad ist **echt gebaut** (Shaped Reward, batched Obs, FPO-Trainer, Launch-Infra) und der
> **Algorithmus-Kern ist gegen die GR00T-API verifiziert** — aber **noch nicht end-to-end auf
> RT-Core-Hardware gelaufen**. Diese Anleitung ist daher der geplante Ablauf; an den mit
> **`# >>> LIVE-CHECK`** markierten Stellen in [`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py)
> muss beim ersten echten Lauf nachjustiert werden (Details in [Schritt 7](#schritt-7--live-check-was-beim-ersten-lauf-zu-prüfen-ist)).
> Nutze zuerst den **Smoke-Test** (`--check`) und ein **kleines `RL_NUM_ENVS`**.

---

## Überblick: Was läuft wo

```
vast.ai Instanz (L40 48 GB / RTX 4090 24 GB — RT-Cores PFLICHT)
└── Docker-Container: lucam03/projekt-humanoider-roboter-sim-vastai:latest
    └── entrypoint_rl.sh → rl_finetune.py
        ├── lädt BC-Checkpoint (Policy) + eingefrorene Referenz (für KL)
        ├── baut vektorisierte Isaac-Lab-Env (RL_NUM_ENVS, reward_mode="shaped")
        └── FPO-Schleife: Rollout → GAE → FPO-Loss (nur Action-Head) → Checkpoint
            ↳ W&B: Erfolgsrate + Reward-Komponenten
```

**Kein separater Server/Client** wie bei der Sim-Eval — der RL-Trainer lädt die Policy in-process
und steuert die Env direkt (Gradienten-fähig).

---

## Voraussetzungen

| Was | Wo |
|---|---|
| vast.ai-Account mit Credits | https://cloud.vast.ai |
| Docker-Hub-Login (`lucam03`) | `docker login` lokal |
| NGC-API-Key für `nvcr.io` | https://ngc.nvidia.com → API Key |
| HuggingFace-Token (`hf_...`) | https://huggingface.co/settings/tokens |
| **BC-Checkpoint** (RL-Startpunkt) | aus dem BC-Training, auf HF hochgeladen (→ Schritt 2) |
| `g1_dex3.usd` Asset | einmalig erzeugt (→ Schritt 3) |
| **GPU mit RT-Cores** | L40 / RTX 4090 / A6000 — **kein A100/H100** (kein RT-Core-Rendering) |

> **Warum RT-Cores Pflicht sind:** RL braucht pro Schritt das **kamerabasierte Rendering** der 4
> Beobachtungskameras (Isaac-Sim-Raytracing). A100/H100 haben keine RT-Cores → ungeeignet. Genau
> deshalb läuft RL **nicht auf KISSKI** (siehe [kisski-hpc.md](../training/kisski-hpc.md)).

---

## Schritt 1 — Image bauen & pushen (einmalig)

RL nutzt **dasselbe kombinierte Sim+GR00T-Image** wie die Closed-Loop-Eval — `rl_finetune.py`
und `entrypoint_rl.sh` sind bereits darin enthalten (`COPY g1_dex3_sim/`, `COPY scripts/` in
[`Dockerfile.vastai`](../../Simulation/Dockerfile.vastai)). Wenn das Image für die Sim-Eval schon
gepusht ist, **entfällt dieser Schritt**.

```powershell
docker login nvcr.io   # Username: $oauthtoken   Password: <NGC-API-Key>
.\Simulation\update_sim_image.ps1 -VastAI
```

Ergebnis: `lucam03/projekt-humanoider-roboter-sim-vastai:latest` auf Docker Hub.

---

## Schritt 2 — BC-Checkpoint als RL-Startpunkt bereitstellen

RL **verfeinert** einen BC-Checkpoint (es ersetzt BC nicht). Lade den besten BC-Checkpoint zu
HuggingFace hoch — identisch zur Sim-Eval, [vastai-anleitung.md Schritt 2](../simulation/vastai-anleitung.md#schritt-2--checkpoint-von-kisski-holen):

```bash
# Modell-Repo (einmalig, privat reicht)
huggingface-cli repo create groot-g1dex3-checkpoint --type model --private
# besten BC-Checkpoint hochladen
huggingface-cli upload luca-mue/groot-g1dex3-checkpoint ./checkpoint-20000/ --repo-type model
```

Im Container `HF_CHECKPOINT_REPO=<hf-user>/groot-g1dex3-checkpoint` setzen → der Entrypoint lädt
den Checkpoint nach `CHECKPOINT_PATH` (`/data/checkpoints/<repo-name>`).

> **Welcher Checkpoint?** Vorab per [Open-Loop-Eval](../training/train-test-split.md) auf dem
> **`test`-Split** den besten 1–2 BC-Checkpoints filtern und dessen **Closed-Loop-Erfolgsrate als
> Baseline** messen — RL wird daran gemessen (RL-Plan Gruppe 4/6).

---

## Schritt 3 — USD-Asset erzeugen (einmalig)

Identisch zur Sim-Eval — die RL-Env spawnt denselben Roboter. Vollständige Anleitung:
[vastai-anleitung.md Schritt 3](../simulation/vastai-anleitung.md#schritt-3--usd-asset-erzeugen-einmalig).
Kurz: `convert_urdf_to_usd.py` erzeugt `g1_dex3.usd`; am besten ins selbe HF-Repo wie den
Checkpoint legen, dann findet der Entrypoint es automatisch (`ASSET_PATH` default =
`$CHECKPOINT_PATH/g1_dex3.usd`).

---

## Schritt 4 — Instanz auf vast.ai konfigurieren

### 4a) GPU auswählen
https://cloud.vast.ai → Search → **L40** (48 GB, empfohlen) oder **RTX 4090** (24 GB) →
Min VRAM ≥ 24 GB, Disk ≥ 60 GB → **Rent**. **Kein A100/H100** (kein RT-Core-Rendering).

### 4b) Instance Configuration

**Image:**
```
lucam03/projekt-humanoider-roboter-sim-vastai:latest
```

**Docker Options** (Entrypoint auf RL überschreiben):
```
--ipc=host --shm-size=16g -p 22 --entrypoint bash
```

**Args to pass to docker entrypoint:**
```
/scripts/entrypoint_rl.sh
```

**Environment Variables:**

| Variable | Wert | Pflicht? |
|---|---|---|
| `HF_TOKEN` | `hf_...` | Ja (Checkpoint-/USD-Download) |
| `HF_CHECKPOINT_REPO` | `luca-mue/groot-g1dex3-checkpoint` | Ja (lädt BC-Checkpoint + USD) |
| `ASSET_PATH` | `/data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd` | Empfohlen (sonst cfg-Default) |
| `RL_NUM_ENVS` | `8` (erst klein!) | Nein (default 16) |
| `RL_ITERATIONS` | `500` | Nein |
| `RL_ROLLOUT_STEPS` | `32` | Nein |
| `RL_LR` | `1e-5` | Nein |
| `RL_KL_COEF` | `0.1` | Nein (KL gegen BC, gegen Reward-Hacking) |
| `RL_CLIP` | `0.2` | Nein |
| `WANDB_API_KEY` | `...` | Empfohlen (Erfolgsrate-Logging) |
| `SHELL_ON_ERROR` | `1` | **Empfohlen** (bei Fehler in Shell statt Container-Tod) |

> **Erst klein anfangen:** Der Render-Durchsatz mit 4 Kameras bestimmt das machbare `RL_NUM_ENVS`
> (RL-Plan Gruppe 0). Mit `RL_NUM_ENVS=2–4` starten, VRAM/FPS beobachten, dann hochskalieren.

**Disk Space:** ≥ 60 GB → **Launch**

---

## Schritt 5 — Smoke-Test vor dem echten Lauf (empfohlen)

Bevor du Stunden RL fährst: per SSH den **Aufbau** ohne Training prüfen (lädt Modell + Critic +
Env, läuft **kein** Rollout):

```bash
# SSH in die Instanz (vastai ssh <id>), dann:
cd /workspace/g1_dex3_sim
python rl_finetune.py --checkpoint /data/checkpoints/groot-g1dex3-checkpoint \
    --asset-path /data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd \
    --num-envs 2 --check
```

`--check` instanziiert Env + Policy + Critic und beendet sich („Aufbau OK"). Schlägt das fehl,
sind die Pfade/das Image das Problem — nicht der RL-Loop.

---

## Schritt 6 — RL überwachen

```bash
vastai ssh <instance-id>            # SSH-Befehl
```

**Trainer-Output** (Vordergrund-Prozess, auch unter vast.ai → Instances → Logs):
```
[rl] device=cuda  num_envs=8
[rl] trainierbare Tensoren: ...
[rl] iter 0000  reward=+0.123  success=0.000
[rl] iter 0001  reward=+0.187  success=0.000
...
[rl] Checkpoint gespeichert: /data/g1_dex3_rl/rl-checkpoint-50
```

**Hauptmetrik ist die Erfolgsrate, NICHT der Loss** (RL-Plan §5/Gruppe 3). In W&B
(Projekt `gr00t-g1-dex3-rl`):
- `success_rate` soll über die Iterationen **steigen** — das ist das Ziel (BC-Plateau → RL-Sprung).
- `reward_mean` steigt; bleibt `success_rate` bei 0, während `reward_mean` stark steigt →
  Verdacht auf **Reward-Hacking** (dann `RL_KL_COEF` erhöhen oder Reward-Gewichte anpassen).

**VRAM prüfen:** `watch -n 2 nvidia-smi`.

---

## Schritt 7 — LIVE-CHECK: was beim ersten Lauf zu prüfen ist

Diese Stellen sind im Code als `# >>> LIVE-CHECK` markiert und können erst auf echter Hardware
final verifiziert werden:

1. **Aktions-Injektion** ([`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py),
   `main` + `fpo_logprob_proxy`): Die gesampelte normalisierte Aktion wird als
   `inputs["action"]` an `model.forward` gegeben, damit `action_head.forward` sie als
   `action_input.action` sieht. Schlüsselname/Shape am echten collated-Input bestätigen.
2. **Obs→Inputs-Adapter** (`_obs_batched_to_policy_dict` / `_sample_action`): Zeitdimension/Dtype
   gegen den echten Processor prüfen.
3. **Minibatch-Slicing** (`_select_env`): bei verschachtelten `eagle_*`-Strukturen ggf. rekursiv.
4. **USD-Asset**: ist `ASSET_PATH` korrekt? Sonst nutzt die Env den cfg-Default aus
   [`g1_dex3_cfg.py`](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py).

> Tipp: Bei `SHELL_ON_ERROR=1` landest du nach einem Fehler in einer Shell und kannst die Stelle
> direkt nachziehen, ohne die Instanz neu zu mieten.

---

## Schritt 8 — RL-Checkpoints sichern

**Vor dem Zerstören der Instanz** herunterladen:

```bash
# RL-Checkpoints (alle save_every Iterationen geschrieben)
scp -P <port> -r root@<ip>:/data/g1_dex3_rl/ ./rl_checkpoints/
```

Der beste RL-Checkpoint wird anschließend genau wie ein BC-Checkpoint in der
[Closed-Loop-Sim](../simulation/vastai-anleitung.md) evaluiert — **RL- vs. BC-Erfolgsrate auf dem
`test`-Split** ist der eigentliche Vergleich (RL-Plan Gruppe 6).

---

## Troubleshooting

### `createDLSSContext error` / Rendering schlägt fehl
GPU ohne RT-Cores. Nur L40 / RTX 30xx-40xx / A6000 — **kein A100/H100/V100**.

### `isaaclab nicht importierbar`
Das Skript braucht das **kombinierte** Image (`Dockerfile.vastai`), nicht das BC-Trainingsimage.

### Erfolgsrate bleibt 0, Reward explodiert
Reward-Hacking — `RL_KL_COEF` erhöhen oder die Shaped-Reward-Gewichte (`rew_*` in
[`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py)) nachjustieren.

### OOM / zu langsam
`RL_NUM_ENVS` senken (Rendering + Modell + Backprop teilen sich den VRAM). Erst mit 2–4 Envs
stabilisieren, dann hochfahren.

---

## Schnellstart

1. Image gepusht (`.\Simulation\update_sim_image.ps1 -VastAI`), BC-Checkpoint + `g1_dex3.usd` auf HF.
2. vast.ai → **L40** (RT-Cores!) → Rent.
3. Image: `lucam03/projekt-humanoider-roboter-sim-vastai:latest`
4. Docker Options: `--ipc=host --shm-size=16g -p 22 --entrypoint bash` · Args: `/scripts/entrypoint_rl.sh`
5. Env:
   ```
   HF_TOKEN=hf_...
   HF_CHECKPOINT_REPO=luca-mue/groot-g1dex3-checkpoint
   ASSET_PATH=/data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd
   RL_NUM_ENVS=4
   WANDB_API_KEY=...
   SHELL_ON_ERROR=1
   ```
6. Disk 60 GB → Launch → per SSH **Smoke-Test** (`--check`) → dann echten Lauf beobachten.
7. `scp` für `/data/g1_dex3_rl/` → Instanz zerstören.
