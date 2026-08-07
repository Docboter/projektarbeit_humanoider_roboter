# RL-Fine-tuning (FPO) — Schritt-für-Schritt-Anleitung

Ziel: Den feingetunten GR00T-N1.6-**BC-Checkpoint** per **Reinforcement Learning (FPO)** in der
Isaac-Lab-Block-Stacking-Sim weiter verfeinern — auf einer **RT-Core-GPU** (L40 / RTX 4090 / A6000
oder ein eigener Server mit RT-Core-GPU, z. B. RTX PRO 6000 Blackwell). RL trainiert die Policy
**in genau der Sim**, in der sie auch evaluiert wird, und optimiert direkt auf **Aufgaben-Erfolg**
statt nur Aktions-Nachahmung (Hintergrund: [reinforcement-learning-plan.md](reinforcement-learning-plan.md)).

> ## ⚠️ Status: erste Implementierung — beim ersten Lauf zu verifizieren
>
> Der RL-Pfad ist **echt gebaut** (Shaped Reward, batched Obs, FPO-Trainer, Launch-Infra) und der
> **Algorithmus-Kern ist gegen die GR00T-API verifiziert** — aber **noch nicht end-to-end auf
> RT-Core-Hardware gelaufen**. Diese Anleitung ist daher der geplante Ablauf; an den mit
> **`# >>> LIVE-CHECK`** markierten Stellen in [`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py)
> muss beim ersten echten Lauf nachjustiert werden (Details in [Schritt 7](#schritt-7--live-check-was-beim-ersten-lauf-zu-prüfen-ist)).
> Nutze zuerst den **Smoke-Test** (`--check`) und ein **kleines `RL_NUM_ENVS`**.
>
> **⚠️ Image-Rebuild-Pflicht:** `lucam03/projekt-humanoider-roboter-sim-vastai:latest` wurde zuletzt
> am 2026-06-15 gepusht. Commit `8e01979` (2026-07-19) hat danach zwei für RL zwingende Fixes in
> `Dockerfile.vastai`/`entrypoint_rl.sh` nachgezogen (gr00t+flash-attn im Isaac-Sim-Python 3.11;
> Start über `isaaclab.sh -p` statt nacktem `python`). **Vor dem ersten Lauf neu bauen+pushen:**
> `./Simulation/update_sim_image.sh --vastai`. `server_rl_run.sh preflight` (Pfad B unten) weist das
> automatisch nach.
>
> **Hardware-Update (2026-08-05):** Der Server, der schon für den
> [RoboCasa-Referenz-Eval](robocasa-referenz-eval.md) genutzt wurde (2× RTX PRO 6000 Blackwell),
> **hat RT-Cores** — RL kann dort grundsätzlich laufen, keine vast.ai-Miete nötig. **Aber:** der erste
> `check`-Lauf dort ist mit einem Segfault in Isaac Sims RTX-Renderer abgestürzt (Treiber-Inkompatibilität
> zwischen Blackwell und dem installierten Treiber-Branch 610.43.02, kein Bug in unserem Code) —
> siehe [Troubleshooting](#segfault-in-librtxscenedbpluginso--carbonpluginstartup-beim-start-rtx-pro-6000-blackwell).
> Der Treiber auf diesem Server ist **fix, kann nicht geändert werden** — der offizielle Fix
> (Downgrade auf 580.65.06) entfällt damit. Nächster Kandidat: **Isaac Sim 6.0** (Details/Quellen im
> Troubleshooting-Eintrag), sonst Pfad A (vast.ai, L40/RTX 4090) als sicherer Fallback.

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

## Pfad B — eigener Docker-GPU-Server mit RT-Cores ★ empfohlen, wenn verfügbar

[`Simulation/server_rl_run.sh`](../../Simulation/server_rl_run.sh) fährt denselben kombinierten
Isaac-Lab+GR00T-Container wie vast.ai (`Dockerfile.vastai`), aber als langlebiger
„Workbench"-Container auf einem generischen Docker-GPU-Server — analog zu
[`server_robocasa_ref_run.sh`](../../Simulation/server_robocasa_ref_run.sh) (Pfad A2 im
[RoboCasa-Referenz-Eval](robocasa-referenz-eval.md)). Spart die vast.ai-Miete, **wenn** ein Server mit
RT-Core-GPU (z. B. die RTX PRO 6000 Blackwell aus dem RoboCasa-Lauf) bereits zur Verfügung steht.

```bash
./Simulation/server_rl_run.sh preflight              # Image-Frische (Commit-8e01979-Fixes) + GPU prüfen
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh setup  # BC-Checkpoint + USD von HF laden (einmalig)
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check  # LIVE-CHECK: Aufbau, 2 Envs, kein Training
HF_TOKEN=hf_... WANDB_API_KEY=... ./Simulation/server_rl_run.sh rl   # echter RL-Lauf
```

- **Image-Rebuild zuerst:** anders als beim RoboCasa-Server-Pfad ist hier ein Rebuild **nötig** (siehe
  Status-Callout oben) — `./Simulation/update_sim_image.ps1 -VastAI` bzw. `update_sim_image.sh --vastai`.
- **Isaac Sim auf Blackwell:** noch nicht offiziell verifiziert (Isaac Lab 2.3.2 ist älter als die
  RTX PRO 6000). `preflight` prüft Torch/flash-attn/gr00t; `check` ist der eigentliche Nachweis, dass
  Kamera-Rendering + Env-Konstruktion auf dieser GPU laufen (entspricht Schritt 5/7 unten).
- **GPU-Zuteilung:** `server_rl_run.sh` pinnt standardmäßig auf eine GPU (`RL_GPUS='"device=0"'`) —
  Rendering + Backprop teilen sich sonst unnötig zwei Karten; `RL_GPUS=all` überschreibt das.
- **Daten:** Checkpoint-Cache, RL-Checkpoints (`/data/g1_dex3_rl/`) und Isaac-Sim-Shader-Cache liegen
  alle unter dem gemounteten `/data` — bleiben zwischen Läufen erhalten, bis `clean`.

---

## Pfad A — vast.ai (Cloud-Miete)

Voraussetzungen und Schritte 1–8 unten sind der vast.ai-Ablauf. Ohne eigenen RT-Core-Server (Pfad B)
ist das der Standardweg.

### Voraussetzungen

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

### Schritt 1 — Image bauen & pushen (einmalig)

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

### Schritt 2 — BC-Checkpoint als RL-Startpunkt bereitstellen

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

### Schritt 3 — USD-Asset erzeugen (einmalig)

Identisch zur Sim-Eval — die RL-Env spawnt denselben Roboter. Vollständige Anleitung:
[vastai-anleitung.md Schritt 3](../simulation/vastai-anleitung.md#schritt-3--usd-asset-erzeugen-einmalig).
Kurz: `convert_urdf_to_usd.py` erzeugt `g1_dex3.usd`; am besten ins selbe HF-Repo wie den
Checkpoint legen, dann findet der Entrypoint es automatisch (`ASSET_PATH` default =
`$CHECKPOINT_PATH/g1_dex3.usd`).

---

### Schritt 4 — Instanz auf vast.ai konfigurieren

#### 4a) GPU auswählen
https://cloud.vast.ai → Search → **L40** (48 GB, empfohlen) oder **RTX 4090** (24 GB) →
Min VRAM ≥ 24 GB, Disk ≥ 60 GB → **Rent**. **Kein A100/H100** (kein RT-Core-Rendering).

#### 4b) Instance Configuration

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

### Schritt 5 — Smoke-Test vor dem echten Lauf (empfohlen)

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

### Schritt 6 — RL überwachen

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

### Schritt 7 — LIVE-CHECK: was beim ersten Lauf zu prüfen ist

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

### Schritt 8 — RL-Checkpoints sichern

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

### Segfault in `librtx.scenedb.plugin.so` / `carbOnPluginStartup` beim Start (RTX PRO 6000 Blackwell)
**Beobachtet am 2026-08-05** (`server_rl_run.sh check` auf `ikr-ki-server-01`, 2× RTX PRO 6000
Blackwell, Treiber 610.43.02/CUDA 13.3): Kit startet, lädt Extensions, stürzt ~7s später beim Anlegen
der ersten USD-Stage (`UsdContext::newStage` → Hydra-RTX-Engine-Aufbau) mit Segfault in
`librtx.scenedb.plugin.so!carbOnPluginStartup` ab — **vor** jeglichem Zugriff auf unseren Checkpoint
oder das `g1_dex3.usd`-Asset, also kein Fehler in unserem Code.

**Root Cause (bestätigt, [NVIDIA-Forum](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-shortly-after-startup-on-windows-server-2025-with-rtx-pro-6000-blackwell/370054),
[GitHub #651](https://github.com/isaac-sim/IsaacSim/issues/651)):** Isaac Sim 5.1.0 (in `isaac-lab:2.3.2`
gebündelt, hier als `5.1.0-rc.19`) ist offiziell nur gegen Treiber **580.65.06/580.88** validiert.
Neuere Treiber-Branches (595.x, 610.x) crashen beim RTX-Renderer-Start auf Blackwell-GPUs — bestätigt
für RTX 5060 Ti/5070/5080/5090 **und explizit RTX PRO 6000 Blackwell**.

**Bekannter Fix:** Host-Treiber auf **580.65.06 (Linux) / 580.88 (Windows)** downgraden.
⚠️ **Aber:** [GitHub #651](https://github.com/isaac-sim/IsaacSim/issues/651) (identischer Treiber
610.43.02, CUDA 13.3, sehr ähnliches Setup) berichtet, dass 580/570 auf neueren Kernels **gar nicht
erst booten** — vor einem Downgrade auf `ikr-ki-server-01` daher erst prüfen, ob Treiber 580 mit
dessen Kernel (7.0.0-28) kompatibel ist. Der Server ist außerdem geteilte Infrastruktur (RoboCasa-Eval
läuft dort mit Treiber 610 fehlerfrei) — ein Treiber-Downgrade ist eine Host-weite Änderung, keine
Container-Einstellung, und sollte nicht ohne Abwägung der Seiteneffekte auf andere Workloads passieren.
**Alternative ohne Host-Änderung:** Pfad A (vast.ai, L40/RTX 4090 — Ada Lovelace, nicht von diesem
Blackwell-spezifischen Bug betroffen).

**Update 2026-08-05 — Treiber auf `ikr-ki-server-01` kann nicht geändert werden.** Damit entfällt der
Downgrade; verbleibende Optionen ohne Host-Eingriff:

- **Isaac Sim 6.0 (seit Juni 2026 GA)** validiert gegen neuere Treiber (offiziell ≥580.95.05) und ein
  Community-Bericht bestätigt Treiber **610.74** als funktionierend (nahe an unserem 610.43.02) —
  [Diskussion #689](https://github.com/isaac-sim/IsaacSim/discussions/689). **Aber nicht garantiert:**
  ein separater, noch ungelöster Bericht zeigt Isaac Sim **6.0.1** auf **derselben RTX PRO 6000
  Blackwell** mit einem anderen Crash (`ERROR_DEVICE_LOST`, Treiber 595.71.05, ebenfalls "Treiber kann
  nicht geändert werden") —
  [NVIDIA-Forum](https://forums.developer.nvidia.com/t/isaac-sim-6-0-1-gpu-crash-error-device-lost-on-rtx-pro-6000-blackwell-driver-595-71-05-cannot-change-driver-on-shared-server/379255).
  Blackwell-Treiberprobleme sind mit 6.0 also gelindert, nicht sicher behoben.
- **Migrationsaufwand ist real, kein Tag-Bump:** Isaac Lab 3.0.0-beta2 (aktuell **Beta**, Stand
  2026-08-05) bündelt Isaac Sim 6.0.0/6.0.1, pinnt **Python 3.12** (statt 3.11) und PyTorch 2.10.0/
  CUDA 12.8. Der Commit-8e01979-Fix (gr00t + flash-attn ins Isaac-Sim-Python installieren) müsste für
  cp312-Wheels neu verifiziert werden — im Kern ein neuer `Dockerfile.vastai`-Port, keine
  Einzeiler-Änderung.
- **Vor dem Investieren des Migrationsaufwands:** kurz im laufenden NVIDIA-Forum-Thread
  (`.../379255`) nachfragen/mitlesen, ob sich für RTX PRO 6000 Blackwell + 6xx-Treiber inzwischen eine
  Lösung ergeben hat — spart ggf. die Portierung.

**Downgrade-Vorgehen (obsolet auf `ikr-ki-server-01` — Treiber dort fix, nicht änderbar; als Referenz
belassen, falls der Server-Constraint sich mal ändert oder für einen anderen Server relevant wird):**

1. **Vorher sichern, nichts löschen:**
   ```bash
   nvidia-smi --query-gpu=driver_version,name --format=csv   # aktuell: 610.43.02
   dpkg -l | grep nvidia-driver                                # exakter Paketname für Rollback
   uname -r                                                    # aktueller Kernel (7.0.0-28-generic)
   ```
2. **Verfügbarkeit von 580 prüfen, BEVOR etwas entfernt wird:**
   `apt-cache madison nvidia-driver-580-open` (gezielt die `-open`-Variante — neue Architekturen wie
   Blackwell werden zuverlässig nur über die offenen Kernel-Module unterstützt).
3. **Absicherung vor dem Wechsel:** Server-Konsole/IPMI-Zugriff sicherstellen (falls der Treiber nach
   dem Reboot nicht lädt, ist kein SSH über die GPU nötig, aber ein Fallback-Zugriffsweg schadet nicht).
   Kernel dabei **nicht** mitupdaten — nur den Treiber wechseln, um Variablen zu reduzieren.
4. **Wechsel:**
   ```bash
   sudo apt remove --purge 'nvidia-*'
   sudo apt install nvidia-driver-580-open
   sudo reboot
   ```
5. **Nach dem Reboot, in dieser Reihenfolge verifizieren:**
   - `nvidia-smi` lädt und zeigt `580.65.06` + die RTX PRO 6000 korrekt an?
   - **Regressionstest zuerst:** `./Simulation/server_robocasa_ref_run.sh preflight` — der bestehende,
     funktionierende RoboCasa-Workload darf durch den Treiberwechsel nicht kaputtgehen.
   - Erst danach: `./Simulation/server_rl_run.sh check` erneut.
6. **Falls 580 nicht bootet / GPU nicht erkannt wird:** zurück auf den in Schritt 1 notierten
   610er-Treiber (`sudo apt install nvidia-driver-610-open` o. ä.), RL-Pfad auf Pfad A (vast.ai)
   umstellen.

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
