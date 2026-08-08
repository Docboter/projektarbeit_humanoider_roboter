# RL-Fine-tuning (FPO) — Schritt-für-Schritt-Anleitung

Ziel: Den feingetunten GR00T-N1.6-**BC-Checkpoint** per **Reinforcement Learning (FPO)** in der
Isaac-Lab-Block-Stacking-Sim weiter verfeinern — auf einer **RT-Core-GPU** (L40 / RTX 4090 / A6000
oder ein eigener Server mit RT-Core-GPU, z. B. RTX PRO 6000 Blackwell). RL trainiert die Policy
**in genau der Sim**, in der sie auch evaluiert wird, und optimiert direkt auf **Aufgaben-Erfolg**
statt nur Aktions-Nachahmung (Hintergrund: [reinforcement-learning-plan.md](reinforcement-learning-plan.md)).

> ## ✅ Status: Pipeline läuft end-to-end — Lernwirkung offen, Greif-Physik ungeklärt
>
> **Seit 2026-08-08 auf echter RT-Core-Hardware durchgelaufen** (Pfad B: RTX PRO 6000 Blackwell,
> Isaac Sim 6.0). Eine vollständige Iteration — Rollout → GAE → FPO-Loss → `backward`/`optim.step`
> → Checkpoint — ist mit `RL_NUM_ENVS=2 RL_ITERATIONS=1 RL_ROLLOUT_STEPS=2` fehlerfrei terminiert
> und hat `rl-checkpoint-1` geschrieben. Die drei `# >>> LIVE-CHECK`-Stellen in
> [`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py) sind damit abgearbeitet
> (was jeweils zu korrigieren war: [Schritt 7](#schritt-7--live-check-auf-hardware-nachgezogene-stellen)).
>
> **Was das noch nicht zeigt:** dass RL die Policy *verbessert*. Ein Ein-Iterations-Smoke-Test
> sagt nichts über Lernverhalten. Offen bleiben deshalb:
> - **⛔ Greif-Physik-Blocker (Läufe 25–28, 2026-08-08):** Auch im Open-Loop-Replay **ohne
>   Modell** wird kein Würfel angehoben (`max_cube_lift` 0,0 cm) — obwohl die Hand die Würfel
>   nachweislich erreicht und berührt und die Finger vollständig schließen (2,09/2,10 rad).
>   Solange nichts angehoben werden kann, liefern weder `reward_mode=binary` noch die
>   `stack`/`height`-Terme des Shaped-Reward ein Signal — **ein RL-Lauf würde in diesem Zustand
>   laufen, ohne lernen zu können.** Ob Test-Platzierung oder Kontakt-Physik die Ursache ist,
>   entscheidet der noch ausstehende `GRASP_MODE=hold`-Lauf mit dem in Lauf 28 korrigierten
>   Fingerkuppen-Messpunkt (Details: [Läufe 25–28 unten](#lauf-28-der-messpunkt-war-zum-dritten-mal-falsch)).
> - **Lernkurve** — steigt `success` über viele Iterationen? (Hyperparameter noch ungetunt.)
> - **Durchsatz** — Render-FPS bei produktivem `RL_NUM_ENVS` mit 4 Kameras (Plan-Gruppe 0).
>
> Vorgehen daher weiterhin: erst **Smoke-Test** (`--check`), dann **kleines `RL_NUM_ENVS`**
> hochskalieren — und den ersten längeren Lauf mit `LIVE_VIEW=1` starten
> ([Schritt 6](#live-zusehen-live_view1--dringend-empfohlen-beim-ersten-großen-lauf)). `reward_mean`
> allein unterscheidet nicht zwischen „Hand nähert sich dem Block" und „Blöcke spawnen außer
> Reichweite"; das sieht man nur im Bild, und in einen laufenden Job lässt es sich nicht nachrüsten.
>
> **⚠️ Image-Voraussetzung:** Es wird das **Isaac-Sim-6.0-Image** gebraucht (`isaac-lab`
> 3.0.0-beta2-post1, gebaut 2026-08-07) — der davor gepushte Tag von
> `lucam03/projekt-humanoider-roboter-sim-vastai:latest` (2026-06-15) ist doppelt veraltet: ihm
> fehlen sowohl die RL-Fixes aus Commit `8e01979` (2026-07-19) als auch der 6.0-Port. Auf einer
> Maschine ohne dieses Image zuerst `./Simulation/update_sim_image.sh --vastai` bauen+pushen;
> `server_rl_run.sh preflight` (Pfad B unten) weist den Stand automatisch nach.
>
> **Hardware-Historie (2026-08-05/08):** Der Server, der schon für den
> [RoboCasa-Referenz-Eval](robocasa-referenz-eval.md) genutzt wurde (2× RTX PRO 6000 Blackwell),
> **hat RT-Cores** — RL läuft dort, keine vast.ai-Miete nötig. Der erste `check`-Lauf ist dort
> allerdings mit einem Segfault in Isaac Sims RTX-Renderer abgestürzt (bekannte Inkompatibilität
> zwischen Blackwell und Treiber-Branch 610.x, kein Bug in unserem Code); der Server-Treiber ist
> **nicht änderbar**. Reaktion: **Migration auf Isaac Sim 6.0** (`isaac-lab` 2.3.2 →
> 3.0.0-beta2-post1) — hat den Segfault behoben und ist seit 2026-08-08 im Betrieb bestätigt.
> Details und Änderungstabelle:
> [Troubleshooting](#segfault-in-librtxscenedbpluginso--carbonpluginstartup-beim-start-rtx-pro-6000-blackwell).
> Fallback bleibt Pfad A (vast.ai, L40/RTX 4090).

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
./Simulation/server_rl_run.sh preflight              # Python 3.12 + torch + flash-attn + gr00t auf der GPU
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh setup  # BC-Checkpoint + USD von HF laden (einmalig)
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check  # LIVE-CHECK: Aufbau, 2 Envs, kein Training
HF_TOKEN=hf_... WANDB_API_KEY=... ./Simulation/server_rl_run.sh rl   # echter RL-Lauf

# mit Live-Ansicht im Browser (empfohlen — siehe Schritt 6) + W&B-Video alle 10 Iterationen:
HF_TOKEN=hf_... WANDB_API_KEY=... LIVE_VIEW=1 RL_WANDB_VIDEO_EVERY=10 \
    RL_NUM_ENVS=4 ./Simulation/server_rl_run.sh rl
```

- **Image-Rebuild zuerst:** anders als beim RoboCasa-Server-Pfad ist hier ein Rebuild **nötig** (siehe
  Status-Callout oben) — `./Simulation/update_sim_image.ps1 -VastAI` bzw. `update_sim_image.sh --vastai`.
- **Isaac Sim auf Blackwell:** Isaac Sim 5.1 (isaac-lab 2.3.2) segfaultet auf der RTX PRO 6000 mit
  Treiber 610.x — daher der Port auf Isaac Sim 6.0 (siehe Troubleshooting). `preflight` prüft
  Torch/flash-attn/gr00t; `check` ist der eigentliche Nachweis, dass Kamera-Rendering +
  Env-Konstruktion auf dieser GPU laufen (entspricht Schritt 5/7 unten).
- **GPU-Zuteilung:** `server_rl_run.sh` reicht standardmäßig **beide** Karten durch
  (`RL_GPUS='"device=1,0"'`). Der Trainer ist kein Data-Parallel-Setup — er nutzt die zweite Karte
  gezielt für **ein** Stück Ballast: das eingefrorene Referenzmodell für die KL gegen den
  BC-Checkpoint. Das läuft nur unter `no_grad`, braucht also seine ~6–7 GB Gewichte plus einen
  transienten Forward, aber keinen Backward-Graphen — genau das gehört nicht auf die Karte, die
  sich Rendering, Policy, Optimizer-States und Aktivierungen ohnehin teilt.
  **Reihenfolge zählt:** die zuerst genannte Karte wird im Container zu `cuda:0` und trägt
  Rendering + Training. Physische GPU 1 steht vorn, weil dort am 2026-08-08 mehr frei war — vor
  einem langen Lauf `nvidia-smi` prüfen und ggf. auf `"device=0,1"` drehen.
  Einzelkarte: `RL_GPUS='"device=0"'`, dann rückt das Referenzmodell automatisch mit auf (altes
  Verhalten). Abschalten trotz zweier Karten: `RL_REF_DEVICE=same`.
- **Daten:** Checkpoint-Cache, RL-Checkpoints (`/data/g1_dex3_rl/`) und Isaac-Sim-Shader-Cache liegen
  alle unter dem gemounteten `/data` — bleiben zwischen Läufen erhalten, bis `clean`.
- **Live-Ansicht:** Der Port 8900 wird beim **Anlegen** des Containers gemappt (`-p` wirkt nur dort).
  Läuft bereits ein älterer Container ohne dieses Mapping, warnt `server_rl_run.sh` und man legt ihn
  einmalig per `clean` neu an — die Daten unter `/data` bleiben dabei erhalten.
- **Code-Änderungen ohne Rebuild:** `Simulation/g1_dex3_sim` ist in den Container **gemountet**,
  `Simulation/scripts` (Entrypoints) dagegen nicht. Änderungen am Trainer wirken darum nach einem
  `git pull` sofort; nur Entrypoint-Änderungen brauchen einen Image-Rebuild. Deshalb liest
  `rl_finetune.py` die `LIVE_VIEW*`-Variablen auch direkt aus der Umgebung statt über CLI-Flags
  des Entrypoints.

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

**Logs werden mitgeschrieben** (Pfad B). Jede Aktion außer `shell` spiegelt ihre komplette
Ausgabe in eine Datei — der Terminal-Scrollback ist bei einem Lauf über Stunden keine
verlässliche Quelle, und Ausgaben wie die Kamera-Pose-Tabelle aus `cams` will man später noch
lesen:

```bash
$RL_HOST_DATA_DIR/logs/rl-20260808-141530.log     # Host-Seite (server_rl_run.sh)
$RL_HOST_DATA_DIR/logs/entrypoint_rl.log          # Container-Seite (entrypoint_rl.sh)
tail -f "$RL_HOST_DATA_DIR"/logs/rl-*.log         # mitlesen
```

Beide liegen unter dem gemounteten `/data`, sind also ohne `docker cp` direkt auf dem Host
lesbar. `RL_HOST_DATA_DIR` ist standardmäßig `/home/lmuecke/project/data/RL`.

#### Live zusehen (`LIVE_VIEW=1`) — dringend empfohlen beim ersten großen Lauf

Mit `LIVE_VIEW=1` blendet der Trainer den laufenden Rollout als **MJPEG-Stream im Browser** ein
(„Spur B" aus dem [Livestream-Plan](livestream-plan.md), Modul
[`live_view.py`](../../Simulation/g1_dex3_sim/live_view.py)):

```
http://<server-ip>:8900/            # Bild + Live-Metriken (Iteration, reward_mean, success_rate)
ssh -L 8900:localhost:8900 <server> # falls nur SSH möglich → http://localhost:8900/
```

**Warum das nicht optional-nice, sondern beim ersten Lauf wichtig ist:** `reward_mean` ist ein
Distanzmaß, kein Erfolgsmaß. Ein flacher Reward-Verlauf sieht identisch aus, egal ob die Hand sich
dem Block nähert, die Blöcke außerhalb der Reichweite spawnen oder die Kamerabilder schwarz sind.
Diese Fehlerklassen erkennt man **visuell in Minuten** statt nach Stunden Rechenzeit. Und
nachrüsten geht nicht: ein bereits laufender Job wird nicht rückwirkend beobachtbar.

Eigenschaften: beliebig viele Zuschauer, zustandslos (Tab schließen und morgen wieder öffnen ändert
nichts am Lauf), reines HTTP (tunnelbar). Kostet **keinen zusätzlichen Render-Pass** — alle Kameras
werden ohnehin jeden Env-Step gerendert; dazu kommen nur ein GPU→CPU-Copy und die JPEG-Kodierung,
letztere nur solange tatsächlich jemand zuschaut. Drosseln mit `LIVE_VIEW_EVERY_N=3`. Bei
`LIVE_VIEW=0` (Default) ist der Codepfad ein reiner Early-Return.

**Welche Kameras?** Default sind `cam_left_high,cam_left_wrist` — zwei der vier **kalibrierten
Policy-Kameras**. Das ist bewusst so: nur diese vier wurden per Overlay gegen die
Dataset-Referenzframes justiert (die zwölf Iterationen sind in
[`g1_dex3_cfg.py`](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py) dokumentiert), und sie zeigen
exakt das, was das Modell als Eingabe bekommt. Für die Frage „nähert sich die Hand dem Würfel?"
ist das aussagekräftiger als eine Übersicht. Alle vier gleichzeitig:
`LIVE_VIEW_CAMS=cam_left_high,cam_right_high,cam_left_wrist,cam_right_wrist`.

> ⚠️ **`cam_scene` ist unvalidiert.** Die Übersichtskamera trägt im Code den Vermerk „nur fürs
> Video" und hat nie eine Kalibrierung gesehen. Am 2026-08-08 zeigte sie in der Live-Ansicht fast
> nur den hellen Dome-Hintergrund und eine Ecke Bodengitter — obwohl die konfigurierte Pose
> nachgerechnet Tisch, Würfel und Roboter vollständig erfassen müsste. Die Zahlen in
> `g1_dex3_cfg.py` sind also nicht die Ursache; die Abweichung entsteht zwischen Konfiguration und
> Render. Zum Nachmessen:
>
> ```bash
> HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
> ```
>
> Das gibt je Kamera die konfigurierte gegen die tatsächlich gerenderte Pose aus — inklusive der
> Pose **relativ zum Env-Ursprung**, weil die Kameras unter `{ENV_REGEX_NS}` hängen und je Env
> geklont werden — und schreibt ein PNG je Kamera nach `/data/cam_dump/`. Auf das Training hat das
> alles keinen Einfluss: `cam_scene` ist nicht Teil der Policy-Observation.

> ⚠️ Der Stream hat **keine Authentifizierung**. Im VPN/Institutsnetz vertretbar — auf einer
> öffentlichen vast.ai-IP nur per SSH-Tunnel nutzen, nicht den Port mappen.

**Zusätzlich archivierbar:** `RL_WANDB_VIDEO_EVERY=10` schneidet alle 10 Iterationen einen
kompletten Rollout ins W&B-Dashboard mit. Nicht live (eine Iteration Verzug), dafür dauerhaft
abrufbar und in der Projektarbeit zitierbar — der MJPEG-Stream ist flüchtig.

**Smoke-Test der Live-Ansicht** (billig, vor dem großen Lauf):

```bash
HF_TOKEN=hf_... LIVE_VIEW=1 ./Simulation/server_rl_run.sh check   # prüft Port-Bindung + Pillow
HF_TOKEN=hf_... LIVE_VIEW=1 RL_NUM_ENVS=2 RL_ITERATIONS=1 RL_ROLLOUT_STEPS=8 \
    ./Simulation/server_rl_run.sh rl                              # Browser zeigt bewegtes Bild
```

Startet der Server nicht (Port belegt, Pillow fehlt), schaltet sich die Live-Ansicht mit einer
`[live] …deaktiviert:`-Zeile ab — der RL-Lauf läuft in jedem Fall weiter.

---

### Schritt 7 — LIVE-CHECK: auf Hardware nachgezogene Stellen

Diese Stellen waren im Code als `# >>> LIVE-CHECK` markiert und ließen sich erst auf echter
Hardware final verifizieren. Alle drei sind am 2026-08-07 auf der RTX PRO 6000 Blackwell
(Isaac Sim 6.0) durchlaufen und korrigiert worden — hier als Nachschlagewerk, falls beim
Portieren auf eine andere Checkpoint-/Embodiment-Kombination Ähnliches auftritt:

1. **Aktions-Injektion** ([`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py),
   `main` + `fpo_logprob_proxy`): Die gesampelte normalisierte Aktion muss in den **inneren**
   Batch — der Collator liefert `BatchFeature({"inputs": batch})`, `model.forward` erwartet
   genau die `inputs`-Ebene. Zusätzlich muss `action_mask` mitgegeben werden, sonst mittelt der
   FPO-Proxy über Padding-Horizont und -Dimensionen.
2. **Obs→Inputs-Adapter** (`_obs_batched_to_policy_dict` / `_sample_action`): Der Sim-Pfad baut
   ein **flaches** Dot-Key-Dict (`video.ego_view`, …), `_unbatch_observation` erwartet aber ein
   **verschachteltes** (`{"video": {...}, "state": {...}}`). Konvertierung in `_flat_to_nested`.
   Beim Zurückschreiben liefert `decode_action` **bare** Gruppen-Namen (`left_arm`, …) — das
   `action.`-Präfix setzt erst der ZMQ-Wrapper, den dieses Skript umgeht.
3. **Minibatch-Slicing**: Ein einzelner Batch-Eintrag lässt sich aus den collated Eagle-Inputs
   **nicht** herausschneiden — `pixel_values` ist über Envs *und* Kameras auf Dim 0 gepackt
   (`(B*n_cams, C, H, W)`), `input_ids` dagegen `(B, L)`. Ein naives `[n:n+1]` gab 1 statt
   4 Bildern und Eagle brach mit `size of tensor a (324) must match tensor b (81)` ab.
   Gelöst durch Gruppieren nach Zeitschritt: ein Forward je `t` über den vollen Env-Batch,
   danach die benötigten Env-Indizes herausgreifen (`_select_env` ist entfallen).
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
Downgrade. Gewählter Weg: **Migration auf Isaac Sim 6.0** (umgesetzt 2026-08-07, siehe unten).

- **Isaac Sim 6.0 (seit Juni 2026 GA)** validiert gegen neuere Treiber (offiziell ≥580.95.05) und ein
  Community-Bericht bestätigt Treiber **610.74** als funktionierend (nahe an unserem 610.43.02) —
  [Diskussion #689](https://github.com/isaac-sim/IsaacSim/discussions/689). **Aber nicht garantiert:**
  ein separater, noch ungelöster Bericht zeigt Isaac Sim **6.0.1** auf **derselben RTX PRO 6000
  Blackwell** mit einem anderen Crash (`ERROR_DEVICE_LOST`, Treiber 595.71.05, ebenfalls "Treiber kann
  nicht geändert werden") —
  [NVIDIA-Forum](https://forums.developer.nvidia.com/t/isaac-sim-6-0-1-gpu-crash-error-device-lost-on-rtx-pro-6000-blackwell-driver-595-71-05-cannot-change-driver-on-shared-server/379255).
  Blackwell-Treiberprobleme sind mit 6.0 also gelindert, nicht sicher behoben.

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

### Isaac-Sim-6.0-Migration (2026-08-07) — implementiert, Hardware-Test offen

[`Dockerfile.vastai`](../../Simulation/Dockerfile.vastai) wurde von `isaac-lab:2.3.2` auf
**`isaac-lab:3.0.0-beta2-post1`** (= Isaac Sim 6.0) portiert. Die Bundle-Angaben unten sind per
pip-list-Inventar aus dem echten Image verifiziert (2026-08-07), nicht aus Release Notes übernommen —
die nannten fälschlich torch 2.11. Folgeänderungen:

| Was | 2.3.2 (Isaac Sim 5.1) | 3.0.0-beta2-post1 (Isaac Sim 6.0) |
|---|---|---|
| Container-User am Ende des Basis-Image | `root` | **`isaaclab` (uid 1000)** → `USER root` im Dockerfile ergänzt, sonst scheitern alle `RUN`-Schritte an Rechten |
| Isaac-Sim-Python | 3.11 | **3.12** (Build-Guard schlägt fehl, falls das nicht stimmt) |
| Gebündeltes PyTorch | 2.7.0+cu128 | **2.10.0+cu128** (torchvision 0.25.0, numpy 2.5.1) |
| flash-attn-Wheel | `v2.7.4.post1 … cu12torch2.7 … cp311` | **`v2.8.1 … cu12torch2.10 … cp312`** — trifft alle Achsen exakt |
| gr00t-Laufzeit-Deps im Kit-Python | pandas kam aus dem 5.1-Bundle | 6.0-Bundle hat **kein pandas** mehr → `pandas==2.2.3` explizit gepinnt (erster Build brach mit `No module named 'pandas'` ab) |

Unverändert bleiben: Ubuntu 24.04 als Basis (deadsnakes-Python-3.10 für die GR00T-venv funktioniert
weiter), `${ISAACLAB_PATH}/_isaac_sim` als Symlink auf `/isaac-sim` (alle Pfad-Referenzen halten), und
die Isaac-Lab-Python-API (`isaaclab.sensors.TiledCamera`, `DirectRLEnv`, `ArticulationCfg` — die
Kamera-Deprecation in 6.0 betrifft `isaacsim.sensors.camera`, **nicht** Isaac Labs eigene Wrapper).
scipy/termcolor/wandb sind im 6.0-Bundle weiterhin enthalten und werden bewusst **nicht** gepinnt
(die uv.lock-Versionen wären Downgrades; `scipy==1.15.3` würde wegen `numpy<2.5` sogar das
Bundle-numpy mit-downgraden).

**⚠️ Verbleibendes Hauptrisiko — numpy 2.5:** gr00t ist gegen numpy 1.26.4 (uv.lock) entwickelt,
das 6.0-Bundle bringt **numpy 2.5.1** (nicht ersetzbar, Isaac Sim hängt daran). Import-Brüche fängt
der Smoke-Test im Dockerfile; ob das numerische Verhalten (Processor/Statistiken) identisch bleibt,
zeigt erst der erste echte Lauf. Dazu bleibt Isaac Lab 3.0.0-beta2 eine **Beta** — API-Abweichungen
in der Env-Konstruktion tauchen erst beim `check` auf.

**Ablauf für den ersten Test:**
```bash
./Simulation/update_sim_image.sh --vastai   # Rebuild auf Isaac Sim 6.0 (~60 min; Smoke-Test bricht bei Dep-Lücken ab)
# auf ikr-ki-server-01, nach git pull:
./Simulation/server_rl_run.sh clean          # alten Workbench-Container (5.1-Image) entsorgen!
./Simulation/server_rl_run.sh preflight      # Python 3.12 + torch 2.10 + flash-attn + gr00t auf der GPU
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check   # der eigentliche Blackwell-Nachweis
```

### `createDLSSContext error` / Rendering schlägt fehl
GPU ohne RT-Cores. Nur L40 / RTX 30xx-40xx / A6000 — **kein A100/H100/V100**.

### Würfel landen nicht auf dem Tisch (`num_envs > 1`) — GELÖST, zwei unabhängige Ursachen

**Symptom (2026-08-08):** Im RL-Lauf mit 4 Envs konnte `success_rate` gar nicht steigen. Der
Kamera-Dump zeigte `block_0` bei `z = 0.025` — die halbe Würfelkante, die Würfel lagen also **auf
dem Boden**.

**Ursache 1 — fehlender Env-Versatz.** `_reset_idx()` übergab die env-lokalen `block_*_range`-Werte
direkt an `write_root_pose_to_sim()`, das **Weltkoordinaten** erwartet; `self.scene.env_origins`
fehlte. Die Würfel landeten damit am Weltursprung, wo bei `num_envs > 1` kein Tisch steht (jede Env
hat ihren eigenen bei `env_origin + (0.5, 0, …)`), und fielen durch.
**Mit `num_envs=1` ist der Fehler unsichtbar** (Env-Ursprung `(0,0,0)`) — deshalb fiel er in allen
Sim-Evals und Replay-Läufen nie auf und erst im RL-Lauf mit mehreren Envs.
Behoben mit `block_pos += self.scene.env_origins[env_ids]`; in `runs/20260808/09` nachgemessen:
Würfel bei Welt `(1.37, -1.17, 0.895)`, also auf dem Tisch von env 0 (Oberkante 0.87 + 0.025).

**Ursache 2 — überlappende Startpositionen.** Alle drei Würfel wurden unabhängig aus **demselben**
Rechteck gezogen (x 0.30–0.40, y −0.20–0.20). Zwei 5-cm-Würfel überlappen dort mit ~18 % je Paar,
bei drei Paaren also in **44,9 %** aller Resets (Monte-Carlo, 200 000 Ziehungen). PhysX löst die
Durchdringung auf, indem es die Würfel auseinanderschießt — in `runs/20260808/08` lagen sie danach
gut 1 m entfernt am Boden. Behoben mit disjunkten y-Bändern je Würfel plus 3 cm Rand:
Überlappungsrate 0,00 %, garantierter Mindestabstand 6 cm bei 5 cm Kantenlänge, x bleibt voll
randomisiert und die y-Gesamtspanne unverändert.

**Nicht betroffen:** Reward und Erfolgskriterium rechnen ausschließlich mit *Differenzen*
(`cdist` Hand↔Würfel, paarweise Würfelabstände, Höhendifferenz) und sind damit frame-unabhängig.
Ebenso die per `init_state` gespawnten Objekte — Tisch und Roboter sind nachgemessen korrekt
(`table` Welt `(1.5, -1, 0.435)`, `robot_root` `(1, -1, 0.85)` bei `env_origin (1, -1, 0)`), Isaac
Lab setzt die selbst env-relativ. Und die Juni-Auswertungen liefen mit `num_envs=1`, sind also
unberührt; der 0-%-Befund dort bleibt der Domain-Gap.

**Folge:** Alle RL-Läufe vor dem Fix mit `RL_NUM_ENVS` 2 oder 4 trainierten auf einer unlösbaren
Aufgabe — die Würfel lagen außerhalb der Reichweite am Boden. `reward_mean` bewegte sich trotzdem,
weil der Shaped Reward aus Gelenk- und Abstandsgrößen kommt.

### Kamerabilder gleichmäßig weiß — GELÖST (`runs/20260808/13` + `14`)

**Kurzfassung für Eilige:** Die Kamera-Prims standen in der USD-Stage **anders ausgerichtet** als
`cam.data` meldete — 95,6° bei den High-Cams, 102,8° an den Handgelenken, 136,8° bei `cam_scene`.
Die Kameras filmten den Himmel. `_setup_scene()` schreibt die Orientierung jetzt selbst
(`RL_CAM_USD_FIX=1`, Default an). In Lauf 14 zeigen `cam_left_high`, `cam_right_high` und
`cam_scene` wieder die Szene; die beiden Wrist-Cams wurden dort wegen eines Namensfehlers noch
übersprungen und sind seitdem nachgezogen. Herleitung unten.

**Stand 2026-08-08 nach `runs/20260808/09`:** Der Würfel-Fix oben ist wirksam, die Bilder sind
trotzdem gleichmäßig hell (min/median/max **245/248/249**, chroma 4,0, dunkel 0 %). Die Szene ist
damit als Ursache ausgeschlossen — und zwar gemessen, nicht vermutet:

| geprüft | Ergebnis |
|---|---|
| Objektposen zur Laufzeit | Tisch, Roboter, Würfel alle korrekt in ihrer Env |
| Würfel im Blickfeld? | `IM BILD` bei 0,66–0,71 m, u/v deutlich innerhalb des Frustums |
| Kamerapose + Konvention | alle fünf Kameras `quat_w_world / +X` bei **0,0°** Abweichung — ⚠️ **diese Zeile war die Falle**, siehe Lauf 13 |
| Clipping-Ranges | Objekte liegen in allen Kameras innerhalb |

**Belichtung: widerlegt** (`runs/20260808/10`, `RL_DOME_SWEEP=500,120,30,8`). Das Bild wird nur
dunkler, es kommt keine Struktur zum Vorschein:

| intensity | 2000 | 500 | 120 | 30 | 8 |
|---|---|---|---|---|---|
| min/median/max | 227/233/234 | 182/193/201 | 83/96/97 | 25/31/32 | 5/7/7 |
| Spannweite | 7 | 19 | 14 | 7 | 2 |

**Was tatsächlich gerendert wird** (Kantenenergie und dominanter Farbanteil je PNG):

| Kamera | Kanten x/y | eine Farbe | Befund |
|---|---|---|---|
| `cam_left_high` | 0.00 / 0.00 | 100,0 % | konstanter Puffer, **nichts** gerendert |
| `cam_right_high` | 0.00 / 0.00 | 100,0 % | konstanter Puffer |
| `cam_right_wrist` | 0.00 / 0.00 | 100,0 % | konstanter Puffer |
| `cam_left_wrist` | 0.29 / 0.14 | 91,0 % | **DEX3-Hand klar sichtbar**, Rest Hintergrund |
| `cam_scene` | 0.34 / 0.47 | 93,7 % | Bodengitter im Eck, Rest Hintergrund |

~~Ein fehlgerichtetes Objektiv in einer beleuchteten Szene zeigt *irgendetwas*. Ein über alle fünf
Belichtungen exakt einfarbiges Bild ist kein Blickwinkel-, sondern ein Renderproblem.~~
**Falsch, und dieser Fehlschluss hat drei Läufe gekostet.** Eine Kamera, die in den leeren Himmel
über einer Domelight-Kuppel blickt, zeigt eben *nicht* irgendetwas, sondern exakt eine Farbe. Der
Schluss „einfarbig ⇒ kein Blickwinkelproblem" war genau verkehrt herum (Lauf 13).

**Domain Randomization: widerlegt** (`runs/20260808/11`, `DR_ENABLED=0`). Die DR lieferte nur die
Chroma (5,76 → 0,00), nicht die Objekte. Schärfer noch, gemessen an der Zahl verschiedener
Grauwerte im ganzen 640×480-Bild:

| Kamera | Grauwerte | Befund |
|---|---|---|
| `cam_left_high` | **1** | nur Dome-Farbe (Lauf 12: kanalweise nachgewiesen) |
| `cam_right_high` | **1** | nur Dome-Farbe |
| `cam_right_wrist` | **1** | nur Dome-Farbe |
| `cam_left_wrist` | 210 | Hand sichtbar |
| `cam_scene` | 156 | Boden sichtbar |

Ein einziger Grauwert ist kein „zu wenig Kontrast", das ist ein Sensor, der nichts von der Szene
schreibt. (Damals als „unbeschriebener Puffer" gelesen — Lauf 12 zeigt, dass es die Dome-Farbe
ist, also ein korrekt gerendertes Bild von leerem Raum. Das verschiebt die Ursache vom Sensor auf
das, was der Renderer an dieser Stelle vorfindet.)
`cam_left_high` blickt *laut `cam.data`* mit −47,7° nach unten; zwischen x ≈ 0,9 m und 2,56 m
müsste derselbe globale Boden im Bild stehen, den `cam_scene` zeigt, dazu die eigenen Arme wie im
Dataset-Referenzbild. Nichts davon. Und `cam_left_wrist` und `cam_right_wrist` unterscheiden sich
in nichts außer dem Link, an dem sie hängen — trotzdem zeigt nur eine von beiden etwas.
~~Damit sind sowohl „Würfel rendern nicht" als auch jede Pose-Erklärung raus.~~ **Auch das war
verkehrt:** dass ausgerechnet eine der beiden baugleichen Wrist-Cams etwas zeigt, ist der
*stärkste* Hinweis auf ein Pose-Problem — die eine blickt in den Roboter, die andere von ihm weg
(Lauf 13).

**Sensorklasse: widerlegt** (`runs/20260808/12`, `RL_CAMERA_CLASS=camera`). Alle fünf Sensoren
liefen als gewöhnliche `Camera` statt `TiledCamera`, bei identischer Pose, Optik und Auflösung —
das Ergebnis ist dasselbe: `cam_left_high` und `cam_right_high` haben 2 Grauwerte (Spannweite 1,4
bei std 0,20 — Denoiser-Rauschen), `cam_right_wrist` genau 1. `TiledCamera` ist damit raus, der
Fehler sitzt in Szene oder Render-Setup.

Lauf 12 hat aber die Ursache eingekreist, weil die Bilder erstmals der Diagnostik **widersprechen**:

| | `cam_left_high` | `cam_scene` |
|---|---|---|
| Diagnostik sagt | Tisch `u=+0.02 v=−0.87`, 1,13 m, **IM BILD** | Tisch, Roboter, alle 3 Würfel **IM BILD**, 2,2–2,55 m |
| Bild zeigt | reine Dome-Farbe | Bodengitter im Eck, sonst reine Dome-Farbe |

Die „leeren" Frames sind kanalweise **exakt** die Dome-Farbe (R 246 / G 244 / B 241 bei
Intensität 2000, R 93 / G 85 / B 76 bei 120) — der 0,75-Graupunkt des `DomeLightCfg` durchs
Tonemapping. Das ist kein unbeschriebener Puffer, das ist ein korrekt gerendertes Bild **von
nichts**: der Renderer sieht an dieser Stelle Himmel. Und `cam_scene` zeigt es am deutlichsten —
Boden ja, Tisch und Roboter mitten im Bild nein. Rechnerisch müsste bei Pitch −22,3° und 47,2°
vertikalem Öffnungswinkel der Boden 97 % des Frames füllen; sichtbar ist er auf ~8 %.

Gleichzeitig **rendert** `cam_left_wrist` die DEX3-Hand sauber, inklusive UNITREE-Schriftzug — das
Roboter-USD ist also im Render-Graph. Nur eben nicht dort, wo `cam.data` es verortet.

**Auflösung: die USD-Stage als unabhängiger Zeuge** (`runs/20260808/13`). Bisher stammten
Soll-Richtung, Treffer-Matrix und Frustum-Rechnung alle aus `data.quat_w_world`; ein Widerspruch
zwischen zwei Ableitungen aus einer Quelle ist mit dieser Quelle nicht auflösbar.
`dump_camera_poses.py` liest deshalb zusätzlich direkt aus der Stage — und der erste Lauf
entscheidet die Sache:

| Kamera | Prim-Position | Blick laut Stage | Blick laut `cam.data` | Abweichung |
|---|---|---|---|---|
| `cam_left_high` | `[1.0, −0.92, 1.45]` | `[0.673, 0, 0.739]` | `[0.666, −0.097, −0.739]` | **95,6°** |
| `cam_right_high` | `[1.0, −1.08, 1.45]` | `[0.673, 0, 0.739]` | `[0.666, +0.097, −0.739]` | **95,6°** |
| `cam_left_wrist` | `[1.16, −0.851, 1.045]` | `[0, −0.882, 0.471]` | `[0.882, 0, −0.471]` | **102,8°** |
| `cam_right_wrist` | `[1.16, −1.149, 1.045]` | `[0, −0.882, 0.471]` | `[0.882, 0, −0.471]` | **102,8°** |
| `cam_scene` | `[2.8, 0.6, 1.7]` | `[0.925, 0, 0.38]` | `[−0.633, −0.675, −0.38]` | **136,8°** |

**Position und Optik stimmen exakt** — auch `focalLength`/`aperture` am Prim ergeben genau die
47,2° × 36,3°, die der Renderer meldet. Es ist ausschließlich die Rotation.

*Warum `cam.data` das nicht sehen kann:* Isaac Lab rechnet die Offset-Rotation beim Spawn von
`convention="world"` nach OpenGL um und beim Lesen wieder zurück. Ist diese Umrechnung fehlerhaft,
hebt der Rückweg sie auf — `quat_w_world` liefert brav die Config zurück, während der Renderer die
verdrehte Pose benutzt. Die Treffer-Matrix konnte diesen Fehler also **prinzipiell** nicht finden.

*Warum die Stage-Pose die gerenderte ist:* weil sie alle fünf Bilder erklärt, Zug um Zug, während
die gemeldete Pose keines erklärt.

| Kamera | Stage-Pose zeigt auf | Bild |
|---|---|---|
| `cam_left_high` / `cam_right_high` | +42,6° nach **oben** — Tisch liegt darunter | reine Dome-Farbe ✔ |
| `cam_right_wrist` | −Y, vom Roboter **weg**, +28° nach oben | genau **1** Grauwert ✔ |
| `cam_left_wrist` | −Y, in den Roboter **hinein**, 0,15 m | DEX3-Hand formatfüllend ✔ |
| `cam_scene` | +X und +22° nach oben, Frame reicht bis −1,3° unter den Horizont | schmaler Bodenkeil unten ✔ |

Besonders `cam_right_wrist` ist beweiskräftig: unter der *gemeldeten* Pose stünde `block_0` in
0,22 m Entfernung mitten im Bild — ein Würfel auf 22 cm rendert garantiert. Unter der Stage-Pose
ist dort leerer Raum. Das Bild hat genau einen Grauwert.

Bei den drei statischen Kameras hat der Fehler eine saubere Signatur: die gerenderte Blickrichtung
ist `(√(fx²+fy²), 0, −fz)` der gemeldeten — **der Yaw wird verworfen und das Vorzeichen des Pitch
gekippt**. Für die Handgelenke fällt das anders aus, weil dort die Link-Rotation dazwischenliegt.
Als feste Konventionsmatrix oder als Vertauschung der Quaternion-Komponenten ließ sich das nicht
nachbauen; der genaue Mechanismus in Isaac Lab 3.0-beta2 bleibt offen — für die Reparatur ist er
auch nicht nötig.

**Der Fix — die Orientierung selbst schreiben.** `_setup_scene()` ruft am Ende
`_force_camera_prim_orientations()`: für jede Kamera wird aus dem Config-Quaternion (Welt-Konvention)
die USD-Orientierung gebildet — `R_usd = R_welt · C` mit den Spalten von `C` als USD-Achsen
(`X_usd = −Y_welt`, `Y_usd = +Z_welt`, `Z_usd = −X_welt`) — und als `xformOp:orient` direkt auf das
Prim geschrieben. Der Op-Stack wird dabei neu aufgebaut (Translate + Orient), damit kein
konkurrierender `rotateXYZ` danebensteht. Die Umrechnung ist offline gegen alle vier Config-Posen
geprüft: Rückweg über die Quaternion trifft die Soll-Blickachse auf < 0,03°.

Abschalten mit `RL_CAM_USD_FIX=0` — dann bleibt die Isaac-Lab-Variante stehen, für den direkten
Vergleich im selben Dump.

Zusätzlich behoben: `ComputePurpose()` nimmt in USD 25.11 keine `TimeCode` mehr, der Geometrieblock
des Dumps ist daran abgestürzt (Zeile 212). Deshalb fehlen `visibility`/`purpose`/BBox in Lauf 13
noch; beide Signaturen werden jetzt bedient.

```bash
git pull
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
```

**Bestätigt in `runs/20260808/14`** — und zwar an den Bildern gemessen, nicht an der Diagnostik:

| Kamera | min/median/max | chroma | dunkel % | Bild |
|---|---|---|---|---|
| `cam_left_high` | 0 / 229 / 241 | 6,93 | 26,9 % | Tisch, Würfel, eigene Hand |
| `cam_right_high` | 0 / 232 / 242 | 7,83 | 18,7 % | dito, gespiegelt |
| `cam_scene` | 10 / 129 / 243 | 7,75 | 32,1 % | ganze Szene: Roboter, Tisch, drei Würfel |
| Juni-Referenz (Isaac Sim 4.x) | 32 / 229 / 239 | 6,0 | 11–22 % | — |

Die High-Cams treffen die Referenz von vor der Migration praktisch exakt. Der Geometrieblock lief
diesmal durch und ist unauffällig: Boden, Tisch, alle drei Würfel, Band und Roboter stehen
`inherited` / `purpose=default` mit korrekten Welt-BBoxen in der Stage — die Hypothesen
„ausgeblendet" und „keine renderbare Geometrie" sind damit beide erledigt.

**Zwei Nachträge aus Lauf 14:**

1. `cam.data` meldet seit dem Fix eine *falsche* Blickrichtung (`cam_left_high`: 174,4° neben dem
   Prim), während das Prim exakt auf der SOLL-Richtung sitzt — dieselbe kaputte Umrechnung, nur
   rückwärts, und damit ein zweiter unabhängiger Beleg. Praktische Folge: **Frustum-Rechnung und
   Treffer-Matrix im Dump sind jetzt unbrauchbar**, Referenz ist das Bild. Der Stage-Abgleich prüft
   deshalb neu gegen die *lokale* Config-Rotation im selben Elternframe (Spalte `SOLL:`); die
   `cam.data`-Spalte bleibt nur noch als Anzeige der Isaac-Lab-Umrechnung stehen.
2. Die beiden Wrist-Cams wurden übersprungen — sie heißen in `CAMERA_CFG`
   `cam_left_wrist_local` / `cam_right_wrist_local`, der Fix suchte unter dem Sensornamen. Er liest
   die Offset-Rotation jetzt aus `cam.cfg.offset`, also aus genau der Quelle, aus der auch Isaac Lab
   beim Spawn liest, und deckt damit alle fünf Kameras unabhängig von der Benennung ab.

**Abgeschlossen mit `runs/20260808/15`.** 20 Prims (5 Kameras × 4 Envs) neu ausgerichtet,
`SOLL: OK (0.0°)` bei allen fünf, und die Bilder bestätigen es:

| Kamera | Graustufen | Kantenenergie | Bild |
|---|---|---|---|
| `cam_left_high` | 248 | 2,12 | Tisch, Würfel, beide Hände |
| `cam_right_high` | 240 | 2,11 | dito, dazu das `stack_band` |
| `cam_left_wrist` | 156 | 1,25 | Unterarm + Hand, gelber und grüner Würfel |
| `cam_right_wrist` | 206 | 1,10 | Unterarm + Hand, Tischkante |
| `cam_scene` | 223 | 5,06 | ganze Szene |

Zum Vergleich Lauf 12 (leer): 1–3 Graustufen bei Kantenenergie 0,00–0,31. Die Kameras sind damit
repariert.

**Die Selbstbewertung des Dumps musste dafür ausgetauscht werden.** Sie meldete in Lauf 15 drei
intakte Kameras als „kein Kontrast, Bild praktisch leer", weil ihr Kriterium der Anteil dunkler
Pixel war (`dunkel < 5 %`). Ein weißer Roboterarm vor einer weißen Tischplatte hat 0,0 % dunkle
Pixel und ist trotzdem vollständig korrekt. Der Test läuft jetzt über **Graustufenzahl und
Kantenenergie** — die messen, *ob* etwas gerendert wurde, statt *wie hell* es ist. An den echten
Daten trennt das sauber: leere Frames 0,00–0,44, intakte 1,10–5,06. `dunkel%` und `chroma` bleiben
als beschreibende Spalten stehen, ohne Urteil.

Offen bleibt eine **Kalibrierfrage, kein Renderfehler**: `cam_right_wrist` blickt an der Tischkante
vorbei ins Bodengitter, während `cam_left_wrist` die Würfel im Bild hat. Das deckt sich mit der
schon in `g1_dex3_cfg.py` notierten Asymmetrie (Iteration 9: „~45°-Diagonal-Versatz rechts = reale
Pose-Differenz"). Die konnte bisher niemand nachjustieren, weil die Kamera gar nichts gerendert
hat — jetzt geht es.

> **Konsequenz für alle bisherigen Ergebnisse:** Jeder Sim-Lauf seit der Isaac-Sim-6.0-Migration
> hat die Policy auf Himmelsbildern laufen lassen. Erfolgsraten, Reward-Kurven und
> Domain-Gap-Zahlen aus dieser Zeit sind gegenstandslos und müssen neu erhoben werden.

**Bestätigt nebenbei** (beides in Lauf 11 im Log): Die Renderer-Intrinsik meldet 47,2° × 36,3° für
die High-Cams — genau das, was die vorher angenommene `horizontal_aperture` ergab, die
Frustum-Rechnung war also korrekt. (Dass diese 47,2° *selbst* falsch waren — projektiert sind 75° —
fiel erst bei der Kalibrierung auf, siehe Iteration 13 weiter unten.) Und der Roll ist 0,0° bei beiden High-Cams und `cam_scene`
(−90° an den Handgelenken, dort armposenabhängig) — beides allerdings wieder aus `cam.data` und
damit nach Lauf 13 hinfällig, solange der Fix nicht bestätigt ist. Die Diagonale in `cam_scene`
hielt ich für die Kante der endlichen Bodenplatte; sie ist der streifende Blick knapp über den
Horizont — sichtbar sind nur ~8 % Boden statt der rechnerischen 97 %.

> **Nebenbefund:** `_randomize_visuals()` würfelt die Dome-Intensität pro Episode neu
> (`uniform(1000, 3800)`). `RL_DOME_INTENSITY` wirkt daher nur bei `DR_ENABLED=0` dauerhaft.

**Drei Lehren aus dieser Fehlersuche:**

1. Der Dump druckte die Objektposen env-relativ unter der Überschrift „WELTPOSITION". Das hat zu
   einer kompletten Fehldiagnose geführt („alle Objekte am Weltursprung"), obwohl die Szene
   korrekt war. Er gibt jetzt **beide** Spalten aus.
2. Die Treffer-Matrix („`quat_w_world/+X` bei 0,0°") vergleicht die *gerenderte* Quaternion gegen
   eine Soll-Richtung aus **derselben** Config mit **derselben** Hilfsfunktion. Sie beweist, dass
   Isaac Lab umsetzt, was verlangt wird — nicht, dass das Verlangte stimmt. Und sie prüft nur die
   *Blickachse*, nicht die Drehung um sie. Der Dump gibt deshalb jetzt zusätzlich den **Roll**
   gegen Welt-Oben aus und stützt die Frustum-Rechnung auf `data.intrinsic_matrices` statt auf
   eine angenommene `horizontal_aperture`.
3. Und das ist die eigentliche Lehre: **eine Größe, die durch einen Hin- und Rückweg derselben
   Umrechnung läuft, kann diese Umrechnung nicht prüfen.** `quat_w_world` meldete fünf Läufe lang
   0,0° Abweichung, während der Renderer 96–137° danebenlag. Vier Hypothesen (Belichtung, DR,
   Anti-Aliasing, Sensorklasse) wurden widerlegt, bevor jemand die Stage selbst gefragt hat. Wenn
   Diagnostik und Beobachtung sich widersprechen, ist die nächste Messung nicht die fünfte
   Variante der Diagnostik, sondern **eine zweite Quelle**.

### Historisch: DLSS-Upscaling (nicht die Ursache)

**Symptom (2026-08-08, RTX PRO 6000 Blackwell):** Alle fünf Kameras liefern nahezu einfarbige
Bilder. `cam_left_high` spannte über das gesamte Bild nur die Helligkeitsstufen **244–249**; die
Chroma fiel von 6,0 (Juni, mit farbigen Würfeln) auf 2,0. Die Wrist-Kameras erwischten noch einen
Streifen Hand am Bildrand, wo im Juni die Hand formatfüllend war. Zum Vergleich der Juni-Frame:
`Simulation/old_videos/12/_debug_obs_cam_left_high.png` (Tisch, drei Würfel, beide Hände).

**Was es NICHT ist** — drei per Messung ausgeschlossene Verdächtige:

| Verdacht | Widerlegt durch |
|---|---|
| Kamera-Pose / -Orientierung falsch | `server_rl_run.sh cams`: für alle fünf Kameras Position deckungsgleich mit dem Offset, Blickrichtung deckungsgleich mit der Config, `quat_w_world/+X` bei **0,0°** |
| Falsche Konvention (`convention="world"` als ROS/OpenGL interpretiert) | Treffer-Matrix im Dump: `quat_w_world/+X` gewinnt bei 0,0°; ROS und OpenGL liegen 22–130° daneben |
| `TiledCamera`-Slicing bei mehreren Envs | Lauf mit `RL_NUM_ENVS=1` zeigt dasselbe Bild wie mit 4 |
| Überbelichtung / Sättigung | Kein Kanal erreicht 255 (Max 247–249), und `cam_left_wrist` hat min=10 — Kontrast ist vorhanden |
| DLSS-Upscaling unter Mindestauflösung | `RL_AA_MODE=DLAA` beseitigte die Warnung, verschlechterte das Bild aber (Chroma 5,93 → 0,96); `RL_AA_MODE=Off` ändert nichts |
| Render-Konvergenz (zu wenige Frames) | `RL_SETTLE_STEPS=60` macht die Läufe deterministisch (Median stabil 232) — aber weiterhin leer |
| Clipping-Range zu eng | High-Kameras: `(0.1, 20.0)`, Tisch bei 0,79 m; Wrist: `(0.01, 5.0)`; Szene: `(0.1, 30.0)` — alles bequem drin |

**Was es ist (Kit-Log):**

```
[Warning] [omni.rtx] DLSS increasing input dimensions:
    Render resolution of (320, 240) is below minimal input resolution of 300.
```

DLSS rendert intern auf halber Auflösung (640×480 → 320×240) und liegt damit unter seinem
eigenen Minimum. Die Env setzte bis dahin **keine** Render-Konfiguration, lief also auf den
Isaac-Sim-6.0-Defaults.

**Auch das war es nicht.** `RenderCfg(antialiasing_mode="DLAA")` ließ sich sauber setzen (alle
Felder akzeptiert, die DLSS-Warnung verschwand), machte die Bilder aber **schlechter**: Chroma
5,93 → 0,96, Wertebereich auf 246–248 geschrumpft. DLAA ist selbst ein temporales Verfahren.
Der Modus ist deshalb **kein Default mehr**, sondern ein Messhebel (`RL_AA_MODE`).

**Die zwei belastbarsten Spuren** — beide aus den Messreihen, nicht aus Logzeilen:

1. **Die dunklen Bildbereiche fehlen.** Im Juni waren 11–22 % der Pixel dunkler als Helligkeit
   100 (Hintergrund und Schatten), heute 0 %. Ein weißer Tisch vor weißem Hintergrund ist
   unsichtbar, egal wie exakt die Kamera steht. Das deutet auf Beleuchtung/Hintergrund.
2. **Die Läufe schwanken.** Bei identischer Szene ergaben aufeinanderfolgende Läufe deutlich
   verschiedene Helligkeiten und Chroma-Werte. Ein konvergierter Render ist deterministisch —
   das deutet auf zu wenige Render-Frames vor der Messung (temporaler Denoiser).

**Drei Hebel zum Eingrenzen**, je ein `cams`-Lauf (~2 min):

```bash
# 1) Render-Konvergenz: viel länger settlen lassen
#    ERLEDIGT: settle=60 macht die Läufe reproduzierbar (median stabil 232), bleibt aber weiß.
RL_SETTLE_STEPS=60 HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
# 2) Anti-Aliasing-Modus durchprobieren (Off|FXAA|DLAA|DLSS|TAA; leer = Isaac-Default)
#    ERLEDIGT für DLAA: griff sauber, machte die Bilder aber SCHLECHTER (chroma 5,93 -> 0,96).
RL_AA_MODE=Off RL_SETTLE_STEPS=60 HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
# 3) Belichtung: mehrere Dome-Intensitäten in EINEM Lauf (aktuell der beste Verdacht)
RL_DOME_SWEEP=500,120,30,8 HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
```

`cams` gibt selbst eine Kennzahlen-Tabelle aus (min/median/max, Graustufen, Kantenenergie, chroma,
dunkel-%) samt der Juni-Referenzwerte und markiert strukturlose Bilder mit
`<-- keine Struktur, nichts gerendert`. Damit ist der Vergleich direkt im Log ablesbar, ohne die
PNGs auszuwerten.

> **Folge fürs Training:** Solange die Policy-Kameras leer sind, sieht das Modell nichts — RL
> optimiert dann gegen ein blindes Modell, während `reward_mean` sich weiter bewegt (der Shaped
> Reward kommt aus Gelenkpositionen). Erst `cams` grün, dann RL starten. Die Juni-Auswertungen
> sind unberührt: die liefen auf korrekt gerenderten Kameras.

### Kopfkameras kalibrieren — Iteration 13 (2026-08-08), Renderprüfung offen

Nachdem die Kameras wieder rendern (`runs/20260808/16`), zeigt der Overlay gegen
`Simulation/camera_reference/`, dass die Sim-Ansicht **nicht** die Ansicht des Datensatzes ist.
Drei Abweichungen, alle gegen eine unabhängige Quelle geprüft statt geschätzt:

| Befund | Beleg | Korrektur |
|---|---|---|
| **Sichtfeld 47,2° statt 75°** | `hfov_deg = 75.0` stand seit jeher in `g1_dex3_cfg.py`, wurde aber **nirgends gelesen** — gespawnt wurde die fest eingetragene `focal_length=24.0` | `focal_high/_wrist/_scene` aus `hfov_*_deg` berechnet, Env liest sie. 75° → 13,65 mm |
| **Kopf verdeckt ein Bilddrittel** | Kameras auf `z=1.45`, `y=±0.08` — neben und über dem Kopf. Der echte G1 trägt sie laut URDF im `d435_link`, pelvis-relativ `(0.0537, 0.0175, 0.4739)` | Montagepunkt env-lokal `(0.0537, 0.0175, 1.3239)`; Nahebene 0,1 → 0,15 m als Absicherung gegen die Kopfschale |
| **Schräge Tischkante** | Beide Kameras zielten auf **einen** Punkt → ±8,3° Gierwinkel. Ein reales Stereopaar blickt parallel | Ziel je Kamera auf der eigenen y-Linie, Gier exakt 0,00° |

Die **Stereobasis** kam aus den Referenzbildern selbst: Querversatz 40 px (SAD-Minimum über die
Tischplatte), Würfelkante 40–46 px bei bekannten 5 cm → Motivabstand ~0,57 m → Basis ~4,7 cm. Der
Wert hängt nicht am angenommenen FOV, weil Abstand und Winkel gemeinsam mitskalieren; er deckt sich
mit den 50 mm einer RealSense D435. Statt ±0,08 (16 cm) also **±0,025**.

Vorausberechnung der neuen Bildaufteilung (Blickachse −55,0°, Gier 0,00°, 75° × 59,8°):

| Punkt | Bildzeile von 480 | |
|---|---|---|
| Tisch-Hinterkante | 29 | Tisch vollständig im Bild wie in der Referenz |
| Würfel (Mitte) | 240 | exakt Bildmitte |
| rechte / linke Hand | 322 / 355 | von unten ins Bild, wie in der Referenz |
| Tisch-Vorderkante | 473 | gerade noch drin |

**Die Handgelenkskameras bleiben unangetastet** — bewusst. Der Stage-Abgleich zeigt für beide
dieselbe lokale Blickrichtung (`[0.882, 0, −0.471]`), sie sind also identisch konfiguriert. Dass
`cam_right_wrist` die Würfel verfehlt und `cam_left_wrist` nicht, kommt somit von der Armpose, nicht
von der Kamera. Eine einseitige Korrektur würde die Symmetrie zerstören, die der reale Roboter hat.

Nächster Schritt: rendern und den Overlay erneut ansetzen.

```bash
git pull
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
# Frames herunterladen, dann lokal:
python Simulation/scripts/overlay_camera_check.py --sim-dir Simulation/runs/<datum>/<nr>/cam_dump
```

Im Overlay muss der Tisch beide Bilder füllen, die Hinterkante annähernd waagerecht liegen und der
eigene Kopf verschwunden sein. Taucht der Kopf weiter auf, sitzt der `d435`-Punkt innerhalb der
Kopfschale — dann `left_high_eye`/`right_high_eye` um 2–3 cm in +X schieben.

**Ergebnis in `runs/20260808/17`:** Der Kopf ist weg, der Tisch füllt symmetrisch das Bild, beide
Hände kommen von unten herein, das Log meldet 75,0° × 59,8°. Zwei Messungen aus dem Overlay:

* **Sichtfeld bestätigt.** Die 5-cm-Würfel messen real 45 × 50 px und in der Sim 45 × 53 px — bei
  vergleichbarem Motivabstand (0,52–0,54 m gegenüber ~0,57 m). Mit den alten 47,2° wären sie rund
  1,6-fach zu groß gewesen.
* **Das Projektionsmodell trägt.** Rechnet man die im Log protokollierten Würfelpositionen durch die
  neue Kamera, landen sie im gerenderten Bild auf ±wenigen Pixeln (grün exakt, mittlere Abweichung
  −6/+6 px; die Ausreißer sind der von der Hand halb verdeckte rote Würfel und der
  Schwerpunkt-Bias der sichtbaren Würfelfläche). Framing lässt sich damit **vorausrechnen**, statt
  es zu errendern — jede weitere Iteration kostet keinen Renderlauf mehr.

**Iteration 14 (aus Lauf 17):** Die Stereobasis wird um `y=0` zentriert statt um die `y=0.0175` des
`d435_link`. In der Reset-Pose stehen die Handgelenke fast symmetrisch (`y=+0.158` / `−0.144`), und
im Referenzbild liegen beide Hände symmetrisch um die Bildmitte — die reale Kamera sitzt also auf
der Mittellinie. Das `d435_link` ist der Montageflansch eines Moduls, nicht der Mittelpunkt
zwischen zwei Bildsensoren. `x` und `z` bleiben beim URDF-Wert, die sind eindeutig. Rechnerisch
rückt die Hand-Mitte damit von 359 auf 341 px (rechte Kamera spiegelbildlich 307 → 289), das Paar
liegt also symmetrisch um die Bildmitte statt 13 px daneben.

**Bestätigt in `runs/20260808/18`.** Die Stereobasis ist damit nicht mehr behauptet, sondern
gemessen — an drei unabhängigen Stellen:

| Prüfung | Ergebnis |
|---|---|
| Vorhersage vs. Rendering, grüner Würfel | links Δ = (−0, −0) px, rechts Δ = (−2, −0) px |
| Disparität links/rechts, gerendert | +44,1 px (rot) / +43,9 px (grün) |
| Disparität aus dem Modell | +42,1 px — **Referenzbilder real: +40 px** |

Damit stimmen Sichtfeld, Montagepunkt und Basis. Der rote Würfel liegt in beiden Kameras um dieselben
−17/−20 px daneben: das ist der Schwerpunkt-Bias der halb von der Hand verdeckten Fläche, kein
Kameraversatz — ein Kamerafehler wäre nicht in beiden Bildern identisch.

**Damit sind die Kopfkameras kalibriert.** Was im Overlay noch verschieden aussieht, sind keine
Kameraparameter:

* **Armpose.** Das Referenzbild zeigt die Arme mitten in der Aufgabe (Hände erhoben, Finger nach
  oben), die Sim steht in der Reset-Pose. Handpositionen sind zwischen beiden nicht vergleichbar.
* **Würfelplatzierung.** Die Sim würfelt sie pro Episode neu. In Lauf 17 lag ihr Schwerpunkt 28 px
  über dem der Referenz, in Lauf 18 — bei identischer Kamera — 6 px darunter. Der Versatz misst den
  Zufallsgenerator, nicht die Pose.
* **Abstand zum Tisch.** Der Tisch überspannt in der Sim vertikal 52,1°, im Referenzbild nur 42,3°.
  Bei gleicher Tischtiefe entspricht das einem Roboter, der **~15 cm weiter vom Tisch weg** sitzt.
  Das ist Szenenlayout, und es zu ändern verschiebt den erreichbaren Greifraum, der aus den
  Replay-Daten abgeleitet wurde (`block_x_range`, Kommentar in `g1_dex3_blockstack_env.py`).
  Notiert als Entscheidung, nicht als Fix.

Nächster Schritt ist damit nicht die nächste Overlay-Runde, sondern **Schritt 2: den Domain-Gap neu
messen** (`Simulation/scripts/measure_domain_gap.py`). Der liefert eine Zahl statt eines
Augenmaßes; die letzten Werte (Mittel 0,26, linke Wrist-Cam 0,43) stammen vom 4. Juni unter
Isaac Sim 4.x und sagen über den heutigen Stand nichts.

### Schritt 2 — Domain-Gap neu messen (`server_rl_run.sh gap`)

Die Junizahlen beschreiben ein Rendering, das es nicht mehr gibt: sie stammen von **vor** dem
Isaac-Sim-6.0-Port, vor dem Orientierungs-Fix (die Kameras filmten danach den Himmel) und vor der
Kalibrierung aus Iteration 13/14. Die Messung wird deshalb wiederholt — mit unverändertem Skript und
unveränderten Referenzbildern, damit der Vergleich trägt.

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams   # erzeugt /data/cam_dump/cam_*.png
./Simulation/server_rl_run.sh gap                    # misst dagegen
```

`gap` legt Referenzbilder und Messskript per `docker cp` in den laufenden Container — kein
Image-Rebuild und kein `clean` nötig, obwohl `/scripts` ins Image gebacken ist und
`Simulation/camera_reference/` dort gar nicht existiert. Gerechnet wird im Isaac-Sim-Kit-Python
(dort liegt `transformers`); der SigLIP-Download (~1,6 GB) landet unter `/data/hf_cache` und
überlebt damit ein `clean`. Ergebnisse: Tabelle auf stdout, dazu
`/data/cam_dump/domain_gap_results.json`.

Gemessen wird die Kosinus-Distanz der Bild-Embeddings durch `google/siglip-so400m-patch14-224` —
identisch mit GR00Ts Vision-Backbone, weil das BC-Fine-tuning mit `tune_visual=false` lief. Das
Skript stellt jeder Kamera ihren Juniwert und das Delta daneben.

| Kamera | 2026-06-04 (Isaac Sim 4.x) | Bewertung damals |
|---|---|---|
| `cam_left_high` | 0,1477 | moderat |
| `cam_right_high` | 0,2136 | groß |
| `cam_right_wrist` | 0,2491 | groß |
| `cam_left_wrist` | **0,4275** | kritisch |
| **Mittel** | **0,2595** | groß |
| Grundlinie real↔real (andere Kamera) | 0,2726 | — |

**Was die Zahl nicht kann.** Sie vergleicht ganze Bilder und enthält damit auch den Szeneninhalt:
Das Referenzbild zeigt die Arme mitten in der Aufgabe, der Kamera-Dump die Reset-Pose, die Würfel
liegen woanders und der Tisch ist ~15 cm näher (siehe oben). Ein Teil der Distanz ist also nicht
Renderqualität. Maßstab dafür ist die Grundlinie real↔real: 0,27 zwischen zwei *echten* Kameras
derselben Szene. Liegt der Real→Sim-Wert darunter, ist der Erscheinungs-Gap kleiner als der
Blickwinkelunterschied zweier realer Kameras.

**Entscheidungsregel — vorher festgelegt, damit die Zahl nicht nachträglich passend gedeutet wird:**

| Mittelwert | Konsequenz |
|---|---|
| < 0,20 | direkt weiter zu Schritt 3 (BC-Erfolgsrate), kein ViT-Eingriff |
| 0,20–0,35 | Schritt 3 trotzdem fahren, aber `TUNE_VISUAL=1` bzw. stärkere Domain-Randomisierung einplanen |
| > 0,35 | der visuelle Gap dominiert; RL auf diesem BC-Checkpoint trainiert gegen eine Policy, die die Szene nicht erkennt — erst Sehen reparieren |

Besonders zu beobachten ist `cam_left_wrist`: mit 0,4275 war sie im Juni der einzige kritische Wert
und ist zugleich die Kamera, die der Orientierungs-Fix am stärksten verändert hat.

#### Ergebnis (`runs/20260808/19`) — der Gap sitzt in der Belichtung, nicht in der Geometrie

| Kamera | jetzt | Juni | Delta | Bewertung |
|---|---|---|---|---|
| `cam_left_high` | 0,1885 | 0,1477 | +0,0408 | moderat |
| `cam_right_high` | 0,1733 | 0,2136 | −0,0403 | moderat |
| `cam_left_wrist` | 0,4284 | 0,4275 | +0,0009 | kritisch |
| `cam_right_wrist` | 0,3422 | 0,2491 | +0,0931 | groß |
| **Mittel** | **0,2831** | 0,2595 | +0,0236 | groß |
| Grundlinie real↔real | 0,2726 | 0,2726 | ±0,0000 | — |

**Die Messapparatur ist validiert.** Die Grundlinie real↔real kommt auf 0,27256725 — der Juniwert
auf vier Stellen identisch, bei anderem Container, anderem Python, anderer Isaac-Sim-Version. Die
Referenzbilder gehen also unverändert durch dieselbe Rechnung. Damit ist der Juni-Vergleich
belastbar: Unterschiede in den Real→Sim-Werten liegen an den Sim-Bildern, nicht am Messweg.

**Die Kalibrierung hat genau das getan, was sie sollte — und nichts darüber hinaus.** Der Mittelwert
der beiden Kopfkameras liegt bei 0,1809 gegen 0,1807 im Juni, also unverändert; ihre *Spreizung*
fällt von 0,066 auf 0,015, ein Faktor 4,3. Vorher schauten die beiden auf unterschiedliche Dinge
(konvergierende Stereobasis, ±8,3° Gierwinkel), jetzt sind sie ein symmetrisches Paar. Iteration
13/14 hat Geometrie repariert, nicht Erscheinung — und die Zahlen zeigen genau das.

**Der verbleibende Gap ist Überbelichtung.** Bildstatistik der vier Kamerapaare (auf 224 px, also
in der Auflösung, die der ViT sieht):

| Kamera | Helligkeit real → sim | Kontrast real → sim | Chroma real → sim | Pixel ≥ 245 in sim |
|---|---|---|---|---|
| `cam_left_high` | 130 → 194 | 59 → 65 | 3,6 → 1,7 | 1,1 % |
| `cam_right_high` | 128 → 194 | 59 → 65 | 4,7 → 1,6 | 1,1 % |
| `cam_left_wrist` | 109 → **232** | 65 → **28** | 5,3 → 2,8 | **16,1 %** |
| `cam_right_wrist` | 104 → **235** | 68 → **28** | 4,0 → **0,7** | **30,6 %** |

Die Sim ist überall 64–131 Graustufen heller als die Referenz. Bei den Kopfkameras überlebt der
Kontrast das noch (65 gegen 59), das Bild bleibt lesbar. Bei den Wrist-Kameras bricht er auf 28 ein,
und 16 % bzw. 31 % aller Pixel liegen bei ≥ 245, sind also nach Weiß abgeschnitten. `cam_right_wrist`
hat mit Chroma 0,71 praktisch keine Farbe mehr. **Die beiden Kameras mit dem größten Domain-Gap sind
exakt die beiden mit abgeschnittenen Pixeln** — das ist die Erklärung, und es ist auch der Grund,
warum die Kalibrierung hier nichts bewirken konnte: sie ändert, wohin die Kamera schaut, nicht wie
hell die Szene ist. Der Kontrastverlust ist Folge des Clippings, also durch weniger Licht umkehrbar.

**Zwei Einschränkungen, die den Juni-Vergleich betreffen:**

* **Die Frames entstanden unter zufälliger Beleuchtung.** Der Dump lief mit `DR=1`, und
  `_randomize_visuals()` würfelt die Dome-Intensität je Episode aus [1000, 3800]. Der Wert 0,2831 ist
  damit eine Stichprobe, kein Betriebspunkt. Für vergleichbare Zahlen `DR_ENABLED=0` setzen.
* **Der Szeneninhalt ist nicht derselbe wie im Juni.** Die Junimessung nutzte
  `_debug_obs_cam_*.png` aus einem laufenden Eval-Rollout (Arme mitten in der Aufgabe), heute kommen
  die Frames aus `dump_camera_poses.py` in der Reset-Pose. Das trifft vor allem `cam_right_wrist`:
  in der Reset-Pose sieht sie Hand und leeren Tisch, im Referenzbild die Würfel. Ein Teil der
  +0,093 ist deshalb Inhalt, nicht Belichtung. Innerhalb *eines* Sweeps entfällt der Effekt, weil
  alle Stufen dieselbe Pose zeigen.

**Entscheidung nach der vorab festgelegten Regel:** 0,2831 liegt im Band 0,20–0,35, also Schritt 3
fahren und einen visuellen Eingriff einplanen. Als Eingriff kommt zuerst die Belichtung dran, nicht
`TUNE_VISUAL=1`: der Hebel existiert bereits (`RL_DOME_INTENSITY`), die Messung dauert einen
Dump-Lauf, und sie trifft genau die zwei Kameras, die durchfallen — ein ViT-Retraining kostet ein
Vielfaches und würde dem Modell beibringen, mit weißgeclippten Bildern zu leben, statt sie zu
vermeiden.

```bash
DR_ENABLED=0 RL_DOME_SWEEP=1000,500,200,80 ./Simulation/server_rl_run.sh cams
./Simulation/server_rl_run.sh gap     # misst Basis + alle Sweep-Stufen in einem Lauf
```

`gap` erkennt die `__dome<wert>`-Varianten selbst und stellt sie als Tabelle nebeneinander, inklusive
der Stufe mit dem kleinsten Gap. Zielgröße: Wrist-Helligkeit von ~233 auf ~105 und der Anteil
geclippter Pixel gegen 0. Bleibt der Gap auch bei der besten Stufe über 0,35, ist es nicht die
Belichtung, und dann ist `TUNE_VISUAL=1` dran.

#### Sweep-Ergebnis (`runs/20260808/20`) — Belichtung ist NICHT der Hebel

Der Sweep lief mit `DR=0` (feste Beleuchtung) über 2000 → 1000 → 500 → 200 → 80, also einen
Faktor 25:

| Variante | left_high | right_high | left_wrist | right_wrist | Mittel |
|---|---|---|---|---|---|
| Basis (2000) | 0,1296 | 0,1586 | 0,4493 | 0,3098 | 0,2618 |
| dome1000 | **0,1154** | **0,1375** | 0,4553 | 0,2973 | 0,2514 |
| dome500 | 0,1346 | 0,1440 | 0,4472 | 0,2968 | 0,2556 |
| dome200 | 0,1424 | 0,1477 | 0,4254 | 0,2796 | 0,2488 |
| dome80 | 0,1450 | 0,1544 | **0,4169** | **0,2776** | **0,2485** |

**Die Hypothese ist widerlegt.** 25-fach weniger Licht bringt 0,0133 — 5 %. Die Helligkeit wurde
dabei nachweislich getroffen: die Wrist-Kameras fallen von 219 auf 119 Graustufen (real 108,5), die
Kopfkameras liegen bei `dome500` mit 131,6 exakt auf dem Referenzwert 130,3, und Clipping
verschwindet komplett. Der Hebel *wirkt*, er bewegt den Gap nur nicht. Noch deutlicher: die
Kopfkameras werden bei `dome1000` **besser**, obwohl sie dort mit 152 zu hell sind, und bei
`dome500` schlechter, obwohl die Helligkeit dort stimmt. Gap und Helligkeitstreffer laufen nicht
einmal in dieselbe Richtung. Damit ist Belichtung als Erklärung erledigt — sie war die naheliegende
Vermutung aus der Pixelstatistik und hat der Messung nicht standgehalten.

#### Was es stattdessen ist: Albedo und Hintergrund

Der Blick auf die Bilder beantwortet es sofort:

| | Referenz (real) | Sim |
|---|---|---|
| DEX3-Hand | **schwarz**, glänzend, Schrauben und Kanten sichtbar | **weiß**, mattes Plastik ohne Details |
| Unterarm | silbern/metallisch | weiß |
| Hinter dem Tisch | heller Laborboden, weiße Wand | **schwarzes Raster** (Isaac-Default-Boden) |

Das erklärt jede gemessene Zahl:

* **Wrist-Kontrast 44 statt 65.** Die Hand füllt den Großteil des Wrist-Bildes. Real ist das
  schwarz auf weißem Tisch — maximaler Kontrast. In der Sim weiß auf weiß. Kein Dimmen der Welt
  ändert ein Verhältnis: dunkler wird beides gleichzeitig.
* **Chroma 1,0–2,0 statt 3,6–5,3.** Weißer Roboter, weißer Tisch, graues Dome-Licht — die Szene ist
  fast unbunt, bis auf drei Würfel.
* **Der Gap ist über 25-fache Beleuchtung stabil.** Albedo ist eine Materialeigenschaft, keine
  Beleuchtungsfrage. Genau das erwartet man, wenn die Ursache in der Oberfläche liegt.
* **Der schwarze Rasterboden** füllt in den Kopfkameras ~45 % und in den Wrist-Kameras ~25 % des
  Bildes — eine große, kontrastreiche Fläche, die im Referenzbild schlicht nicht vorkommt.

Zwei Hebel dafür, beide ungesetzt wirkungslos (nichts ändert sich still):

| Hebel | Beispiel | Wirkung |
|---|---|---|
| `BLACK_HANDS=1` (Default) | — | `server_rl_run.sh` erzeugt und benutzt `g1_dex3_blackhands.usd` |
| `RL_GROUND_COLOR` | `0.35,0.35,0.36` | hellt den Isaac-Default-Boden auf |

```bash
DR_ENABLED=0 RL_GROUND_COLOR=0.35,0.35,0.36 ./Simulation/server_rl_run.sh cams
./Simulation/server_rl_run.sh gap
```

**Erwartung, damit sie prüfbar bleibt:** Wrist-Kontrast von 44 Richtung 65 und `cam_left_wrist`
unter 0,35. Passiert das nicht, ist auch Albedo nicht die Erklärung, und dann ist `TUNE_VISUAL=1`
an der Reihe — dann ist es Textur und Materialcharakteristik, und dagegen hilft nur, dem ViT die
Sim-Optik beizubringen.

**Einordnung, die dabei nicht untergehen soll:** Der Mittelwert liegt mit 0,2618 (bzw. 0,2485 bei
`dome80`) **unter** der Grundlinie real↔real von 0,2726. Im Schnitt ist ein Sim-Bild seinem realen
Gegenstück also näher, als zwei *echte* Kameras derselben Szene einander sind. Das Problem ist nicht
das Mittel, sondern die Verteilung: Kopfkameras 0,13–0,16 (unauffällig), `cam_left_wrist` 0,42.

#### Ergebnis (`runs/20260808/21`) — halb gemessen, und die Hälfte war schon gelöst

Der Lauf sollte beide Albedo-Hebel prüfen. Gemessen wurde nur einer:

```
[Env] Bodenfarbe -> (0.75, 0.73, 0.7) (RL_GROUND_COLOR)
[DR] Handfarbe -> (0.05, 0.05, 0.05) an 0 Materialien (RL_HAND_COLOR).
[DR] WARNUNG: kein Hand-Material getroffen — Bindungen prüfen
```

**Der Boden allein macht es nicht besser, sondern schlechter.** Gap-Mittel 0,2618 → 0,2655,
`cam_left_high` 0,1296 → 0,1542, `cam_right_wrist` 0,3098 → 0,3397; nur `cam_left_wrist` fällt
(0,4493 → 0,4113). Die Pixelstatistik sagt, warum: Der Wert war zu hell gewählt und hat den
Kontrast von einer Überschreitung in eine Unterschreitung gekippt.

| Kopfkameras | real | Lauf 20 (schwarzer Boden) | Lauf 21 (Boden 0,75) |
|---|---|---|---|
| Helligkeit | 130,3 | 169,5 | 203,3 |
| Kontrast | 59,4 | 79,3 | **39,4** |
| Chroma | 9,1 | 2,7 | **23,3** |

Ein Zwischenwert (`0.35,0.35,0.36`) ist der nächste Versuch — der schwarze Rasterboden bleibt
falsch, 0,75 war nur die falsche Richtung von zu wenig zu zu viel.

**Der Hand-Hebel konnte nicht funktionieren, und er war überflüssig.** Zwei Gründe:

1. *Er kann es nicht.* Die `/visuals`-Prims des Roboters sind `instanceable`; die Meshes liegen in
   USD-Prototypen. `Usd.PrimRange` läuft dort nicht hinein, und eine Bindung am Instance-Root
   komponiert nicht in den Prototyp. Daher 0 Materialien. Der Prim-Pfad-Filter war zusätzlich
   wirkungslos: der Roboter-Root heißt `g1_29dof_with_hand_rev_1_0` und enthält `_hand_` selbst,
   also passte *jeder* Pfad darunter.
2. *Es gab ihn schon.* [`recolor_hands_black.py`](../../Simulation/g1_dex3_sim/recolor_hands_black.py)
   löst seit Juni exakt dieses Problem — de-instanziert die Hand-`/visuals` und bindet je Mesh mit
   `strongerThanDescendants`. `data/g1_dex3_blackhands.usd` liegt seit dem 04.06. im Repo,
   `g1_dex3_cfg.py` zeigt per Default darauf, `kisski_sim_submit.sh` erzeugt es automatisch.

Warum die Hände auf dem Server trotzdem weiß sind: `server_rl_run.sh` setzte
`ASSET_PATH=$CHECKPOINT_PATH/g1_dex3.usd` — das Original. Der KISSKI-Launcher zog das
schwarzhändige Asset, dieses Skript nicht. Behoben: `BLACK_HANDS=1` ist Default, `ensure_black_hands`
erzeugt das Asset bei Bedarf im Container und fällt bei Fehlschlag aufs Original zurück. Der
Laufzeit-Hebel `RL_HAND_COLOR` ist entfernt — ein zweiter Mechanismus für dieselbe Sache, der
nachweislich nicht greift.

**Zwei Werkzeugfehler, die der Lauf offengelegt hat:**

* Die Sweep-Zeilen in Lauf 21 sind **byteidentisch mit Lauf 20** (`md5sum` geprüft) — alte
  `__dome*.png` überlebten im Container, obwohl `sweep=<aus>` lief. `cams` löscht sie jetzt vorher.
* Das Sweep-Mittel wurde über *vorhandene* Kameras gebildet und gegen ein 4-Kamera-Mittel gestellt.
  Deshalb stand da „Keine Sweep-Variante schlägt die Basis" — auf denselben zwei Kameras gerechnet
  war es umgekehrt. `measure_domain_gap.py` vergleicht jetzt ein Delta gegen dieselben Kameras und
  markiert unvollständige Zeilen mit `*`.

**Stand der Vorhersage:** Das Kriterium (`cam_left_wrist` < 0,35) ist mit 0,4113 nicht erfüllt — der
Test der eigentlichen Hypothese hat aber nie stattgefunden. Damit daraus kein Nachbessern bis zum
Erfolg wird: **ein** Lauf mit schwarzen Händen und Boden 0,35. Bleibt `cam_left_wrist` dann über
0,35, ist Albedo widerlegt und `TUNE_VISUAL=1` die Konsequenz — ohne weiteren Zwischenversuch.

#### Ergebnis (`runs/20260808/22`) — Albedo bestätigt, Kriterium um 0,0056 verfehlt

Diesmal liefen beide Hebel, nachweisbar im Log:

```
OK: schwarzes Material an 16 Hand-/visuals-Roots gebunden (de-instanziert, strongerThanDescendants)
Asset erzeugt: /data/checkpoints/groot-g1dex3-checkpoint/g1_dex3_blackhands.usd
[Env] Bodenfarbe -> (0.35, 0.35, 0.36) (RL_GROUND_COLOR)
```

| Kamera | Juni | Lauf 20 | Lauf 21 | **Lauf 22** | Δ zu Juni |
|---|---|---|---|---|---|
| `cam_left_high` | 0,1477 | 0,1296 | 0,1542 | **0,1121** | −0,0356 |
| `cam_right_high` | 0,2136 | 0,1586 | 0,1567 | **0,1310** | −0,0826 |
| `cam_left_wrist` | 0,4275 | 0,4493 | 0,4113 | **0,3556** | −0,0719 |
| `cam_right_wrist` | 0,2491 | 0,3098 | 0,3397 | **0,2930** | +0,0439 |
| **MITTEL** | 0,2595 | 0,2618 | 0,2655 | **0,2229** | −0,0366 |

Alle vier Kameras verbessern sich gegenüber Lauf 21, drei von vier gegenüber Juni. Das Mittel liegt
mit 0,2229 unter der Grundlinie real↔real (0,2726) und praktisch auf der sim-internen Streuung
(0,2163) — real→sim ist damit das 0,8-fache des real→real-Abstands.

Die Pixelstatistik zeigt, dass der Mechanismus der vermutete war. Der Kontrast der Kopfkameras
trifft jetzt den realen Wert fast exakt, nachdem er in Lauf 20 zu hoch und in Lauf 21 zu niedrig war:

| Kopfkameras | real | Lauf 20 | Lauf 21 | **Lauf 22** |
|---|---|---|---|---|
| Helligkeit | 130,3 | 169,5 | 203,3 | 184,9 |
| Kontrast | 59,4 | 79,3 | 39,4 | **59,0** |
| Chroma | 9,1 | 2,7 | 23,3 | 19,7 |

Der Wrist-Kontrast, der Auslöser der ganzen Hypothese, geht von 36,1 (Lauf 20) auf **68,9** bei real
65,0 — die schwarze Hand auf hellem Tisch stellt genau das Verhältnis her, das im Datensatz steht.

**`RL_GROUND_COLOR` macht den Boden nicht grau, sondern entdimmt Isaacs blaue Rastertextur.**
Ein nahezu neutraler Wert (0,35/0,35/0,36) rendert sichtbar blau, und die Chroma steigt vom
Isaac-Default 2,7 auf 19,7 bei real 9,1. Ein neutraler Eingang kann kein farbiges Ergebnis
erzeugen, wenn er die Albedo direkt setzt — er wirkt also als Tint auf eine Textur (vermutlich
`Looks/theGrid.inputs:diffuseColor` des Default-Bodens; in der Quelle nicht geprüft, Isaac Lab liegt
nur im Container). Praktische Folge: Aufhellen des Bodens erhöht zwangsläufig die Sättigung. Für
einen wirklich neutralen Boden müsste das Material ersetzt statt getönt werden. Betroffen sind vor
allem die Kopfkameras — und die sind mit 0,11/0,13 ohnehin unauffällig.

**Stand der Vorhersage:** `cam_left_wrist` liegt bei **0,3556**, das Kriterium war **< 0,35**. Um
0,0056 verfehlt. Das wird hier nicht gerundet: Albedo erklärt einen großen, messbaren Teil des Gaps
— es reicht aber nicht, um die Wrist-Kamera aus der Ausreißerrolle zu holen (sie ist weiterhin das
1,3-fache der real↔real-Grundlinie). Nach der Regel gilt damit: **kein weiterer Albedo-Versuch**,
`TUNE_VISUAL=1` ist die Konsequenz.

Bevor dafür ein H100-Lauf über ~48 h gebucht wird, ist Schritt 3 (BC-Erfolgsrate in der Sim) fällig
— er stand ohnehin als Nächstes an, kostet eine Sim-Eval statt eines Trainings und misst die
Größe, die `TUNE_VISUAL=1` verbessern soll, direkt statt über den Proxy. Ist die Erfolgsrate 0,
ist die Entscheidung bestätigt. Das ist ausdrücklich **kein** weiterer Renderversuch.

Offen und in derselben Kamera wirksam: die bekannte Abweichung von ~15 cm im Roboter-Tisch-Abstand.
Der Modulkopf von `measure_domain_gap.py` warnt selbst davor, dass die Zahl Szeneninhalt enthält;
die Wrist-Kamera ist dafür die empfindlichste. Das ist für Schritt 3 ohnehin zu klären, weil eine
unerreichbare Tischplatte jede Erfolgsrate auf 0 nagelt, unabhängig von der Optik.

### Schritt 3 — BC-Erfolgsrate in der Sim (`server_rl_run.sh eval`)

Der Domain-Gap ist ein Proxy. Diese Zahl ist die Zielgröße selbst: **schafft die BC-Policy die
Aufgabe in der Simulation überhaupt?** Sie entscheidet zwei Dinge auf einmal — ob sich ein
`TUNE_VISUAL=1`-Lauf über ~48 h lohnt, und was der Nullpunkt für jeden späteren RL-Vergleich ist.
Ohne sie lässt sich „RL hat geholfen" nicht belegen, egal wie die RL-Kurve aussieht.

```bash
HF_TOKEN=hf_... NUM_EPISODES=20 EPISODE_LENGTH_S=120 DR_ENABLED=0 \
    ./Simulation/server_rl_run.sh eval
```

Was hier läuft, ist **nicht** `rl_finetune.py`, sondern die vollständige Closed-Loop-Pipeline
(`entrypoint_sim.sh`: GR00T-Policy-Server über ZMQ + Isaac-Lab-Client). Der Unterschied ist für die
Zahl wesentlich: der Client führt 8 Schritte eines 16er-Chunks aus, während der RL-Rollout jeden
Step neu plant und 15 von 16 Vorhersagen wegwirft. Gemessen werden soll der Betriebsmodus.

| Parameter | Wert | Begründung |
|---|---|---|
| `NUM_EPISODES` | 20 | Auflösung siehe unten |
| `EPISODE_LENGTH_S` | 120 | 3× die menschliche Demo. `replay_episode0.npz` hat 1173 Steps = 39 s bei 30 Hz. Der cfg-Default von 300 s ist das 7,7-fache und kostet nur Laufzeit |
| `EXECUTION_HORIZON` | 8 (Default) | wie im Deployment |
| `DR_ENABLED` | 0 | feste Beleuchtung, sonst ist jede Episode eine andere Szene |

Laufzeit: bis zu 20 × 120 s × 30 Hz = 72 000 Env-Steps mit 5 gerenderten Kameras, dazu ~9000
Policy-Aufrufe. Grobe Schätzung 1–2 h; erfolgreiche Episoden brechen früher ab. Ergebnisse landen in
`$HOST_DATA_DIR/sim_results/results.json`, Videos je Episode in `$HOST_DATA_DIR/sim_videos/`.

**Entscheidungsregel — vorher festgelegt, damit die Zahl nicht nachträglich gedeutet wird:**

| Erfolgsrate | Konsequenz |
|---|---|
| **0/20** | Die Policy löst die Aufgabe in der Sim nicht. **Erst Geometrie ausschließen** (s. u.), dann ist `TUNE_VISUAL=1` bestätigt — RL auf einem Nullpunkt-Reward kann nichts lernen, dem fehlt das Startsignal |
| **1–3/20** | Schwaches, aber echtes Signal — der beste Startpunkt für RL. Kein `TUNE_VISUAL`-Lauf, direkt RL, denn FPO braucht genau diesen seltenen Erfolg als Gradient |
| **> 3/20** | BC funktioniert in der Sim. RL ist reine Verbesserung, `TUNE_VISUAL` erübrigt sich |

**Was 0/20 statistisch heißt.** Bei 20 Episoden und null Erfolgen liegt die obere 95-%-Grenze der
wahren Rate bei rund 14 %. „0/20" schließt also eine schwache Fähigkeit nicht aus, es schließt nur
eine brauchbare aus. Für die Unterscheidung 0 % gegen 5 % wären ~60 Episoden nötig — das ist erst
interessant, wenn überhaupt ein Erfolg auftritt.

**Vor der Interpretation zu prüfen — sonst misst der Lauf etwas anderes als gedacht:**

* **Erreicht die Hand den Tisch?** Die bekannte Abweichung von ~15 cm im Roboter-Tisch-Abstand
  nagelt jede Erfolgsrate auf 0, unabhängig von der Optik. Die Videos in `sim_videos/` zeigen das
  unmittelbar: greift der Roboter ins Leere oder daneben, ist das Ergebnis kein Aussagewert über das
  Sehen. **Dieser Punkt ist die einzige zulässige Erklärung für eine 0, die nicht `TUNE_VISUAL=1`
  auslöst** — und er ist vor dem Lauf offen, nicht danach erfunden.
* **Sieht die Policy etwas?** Der Runner legt in Episode 1/Step 0 `_debug_obs_cam_*.png` in
  `sim_videos/` ab — vier Bilder, genau die Modell-Eingabe. Sind die leer, weiß oder schwarz, ist
  der Lauf ungültig und keine Aussage über die Policy.

#### Zwischenstand (runs/20260808/23, erste 2 von 20 Episoden)

Der Lauf startete 16:19 Uhr mit genau den Werten oben; das Log wurde bei Episode 3 kopiert. Beide
fertigen Episoden: **misslungen, volle 3600 Steps**, kein früher Abbruch. Beide Vorprüfungen sind
damit beantwortet — und keine der beiden erklärt das Ergebnis:

* **Die Policy sieht die Szene.** `_debug_obs_cam_left_high.png` zeigt Tisch, alle drei Würfel und
  beide schwarzen Hände scharf und mittig. Kein leeres oder überstrahltes Bild.
* **Die Hand erreicht den Tisch.** Im Szenenvideo liegt die linke Hand über weite Strecken auf
  Tischhöhe direkt neben dem roten Würfel. Die 15-cm-Abweichung äußert sich **nicht** als
  „greift ins Leere". Damit fällt die einzige zugelassene Nicht-Sehen-Erklärung für eine 0 weg.

Der eigentliche Befund steht aber nicht im Erfolgszähler, sondern im Video: **die Würfel bewegen
sich in 7200 Steps kein einziges Mal.** Anfangs- und Endbild beider Episoden zeigen sie
pixelgenau an derselben Stelle (die Startlagen sind zwischen den Episoden randomisiert, die
Endlagen also nicht durch eine feste Szene erklärt). Die Arme bewegen sich dabei durchgehend: die
Frame-zu-Frame-Differenz im Tischausschnitt liegt konstant bei 0,35–0,41 über alle sechs
20-s-Fenster, ohne jede Phasenstruktur. Kein Anfahren, kein Greifen, kein Anheben — eine
gleichförmige Bewegung, die den Würfel nie berührt.

Das ist informativer als die Erfolgsrate und war mit dem bisherigen Instrumentarium nur per Auge
am Video zu sehen. Deshalb protokolliert der Runner seit diesem Lauf zwei Zahlen je Episode
(`env.get_reach_diagnostics()`, kostet keinen zusätzlichen Render-Pass):

| Feld in `results.json` | Bedeutung |
|---|---|
| `min_reach_m` / `min_reach_step` | kleinster Abstand Hand↔Würfel in der Episode, und wann |
| `block_shift_max_m` | größte Verschiebung eines Würfels gegenüber dem Reset-Layout |

Damit trennt der nächste Lauf die beiden Lesarten quantitativ: kommt die Hand auf wenige
Zentimeter heran und greift trotzdem nicht, ist es Wahrnehmung/Politik und `TUNE_VISUAL=1` steht;
bleibt der Abstand groß, ist es Geometrie und ein ViT-Lauf wäre verschwendet.

#### Erste Messung (runs/20260808/24) — und warum sie so noch nichts entscheidet

Zwei Episoden à 40 s (1200 Steps), 0/2, mit aktiver Diagnose:

| Episode | `min_reach_m` | `min_reach_step` | `block_shift_m` (drei Würfel) |
|---|---|---|---|
| 1 | 0,1476 | 13 | 0,0189 / 0,0165 / 0,0200 |
| 2 | 0,1473 | 18 | 0,0209 / 0,0140 / 0,0200 |

Die naheliegende Lesart — „15 cm Abstand, genau die bekannte Tischabweichung, also Geometrie" —
hält der Prüfung **nicht** stand. Beide Spalten messen etwas anderes als gedacht:

* **Der Abstand war am falschen Ende der Hand gemessen.** `_get_hand_positions()` liefert
  `left/right_wrist_yaw_link`, also die Handwurzel. Laut URDF liegen zwischen ihr und dem letzten
  Fingergelenk 0,0415 + 0,0777 + 0,0458 = **16,5 cm**, die Fingerspitze noch etwas weiter. Ein
  Handwurzel-Abstand von 14,7 cm liegt damit *innerhalb der eigenen Handgeometrie* und ist
  mehrdeutig: er beschreibt „Würfel liegt in der Greiföffnung" genauso gut wie „Würfel 15 cm
  daneben". Die Zahl kann die Frage, für die sie eingebaut wurde, in dieser Form nicht beantworten.
* **Die Würfel-Verschiebung misst den Solver, nicht den Roboter.** Alle drei Würfel verschieben
  sich um 1,4–2,1 cm, in beiden Episoden, auch die, in deren Nähe nie eine Hand war (der dritte
  in beiden Episoden auf 4 Stellen identisch: 0,0200). Das ist das Einschwingen des Kontakts nach
  dem Reset. Das Kriterium „>1 cm = angefasst" meldete deshalb 2/2 — ein reiner Fehlalarm.

Belastbar ist dagegen `min_reach_step`: **13 bzw. 18 von 1200.** Die größte Annäherung der
gesamten Episode fällt in die erste halbe Sekunde und wird in den folgenden 40 s nie wieder
unterboten. Daraus schien zu folgen, dass sich die Arme nie auf einen Würfel zubewegen — **auch
das war ein Artefakt des Messpunkts** (s. Lauf 25): die Handwurzel bleibt zurück, während sich die
Finger nach vorn strecken, ihr Minimum liegt deshalb früh. Von Lauf 24 bleibt am Ende nur, dass
seine Zahlen die Frage nicht beantworten konnten.

Konsequenz im Code (alles in `get_reach_diagnostics` / `run_g1_dex3_sim_eval.py`):

| Änderung | Grund |
|---|---|
| Messkörper = Fingerspitzen (`*_hand_index_1_link`, `*_middle_1_link`, `*_thumb_2_link`), Fallback Handfläche → Handwurzel | nur an der Kontaktfläche heißt „klein" auch „am Würfel" |
| `reach_frame` in `results.json` + Startzeile | die Zahl darf nie ohne ihren Bezugsrahmen gelesen werden |
| `reach_start_m` zusätzlich zu `min_reach_m` | erst die Differenz zeigt, ob sich der Roboter überhaupt angenähert hat |
| Würfel-Grundlage erst nach 30 Steps Karenz | schneidet das Einschwingen ab, das sonst als „angefasst" zählt |
| `finger_span_max_rad` (Action-Dims 14:28) | zeigt, ob überhaupt ein Griff kommandiert wurde |

#### Die eigentliche Messung (runs/20260808/25) — die Hand ist am Würfel

> **Nachtrag Lauf 28:** die „Fingerspitzen"-Abstände unten sind ab dem *distalen Gelenk* gemessen,
> die Kuppe liegt 5,2 cm weiter. Alle Zahlen sind also um diesen Betrag zu groß — die Schlussfolgerung
> „die Hand ist am Würfel" wird dadurch stärker. Details und Fix im Abschnitt zu Lauf 28.

Derselbe Lauf mit korrigiertem Messpunkt, wieder 2 × 40 s, wieder 0/2:

| Episode | `reach_start_m` → `min_reach_m` | `min_reach_step` | `block_shift_m` |
|---|---|---|---|
| 1 | 0,135 → **0,0396** | 58 | 0,0147 / 0,0 / 0,0 |
| 2 | 0,127 → **0,0420** | 1014 | 0,0570 / 0,0 / 0,0 |

Das kehrt den Befund aus Lauf 24 um:

* **Der Roboter nähert sich.** 9,5 cm Annäherung gegenüber der Startpose, in beiden Episoden auf
  den Millimeter gleich. Das ist kein Zufallsprodukt einer wackelnden Bewegung.
* **Die Fingerspitzen erreichen den Würfel.** 4,0 bzw. 4,2 cm zum Würfel*mittelpunkt*, bei 5 cm
  Kantenlänge also rund **1,5 cm zur Oberfläche**.
* **Es gibt echten Kontakt.** Genau ein Würfel verschiebt sich (1,5 bzw. 5,7 cm), die beiden
  anderen exakt 0,0 — die Karenzzeit funktioniert, und der bewegte Würfel wurde angefasst.

Damit ist die Vorprüfung „Geometrie" endgültig erledigt: die Hand kommt hin, berührt den Würfel
und schiebt ihn. Was fehlt, ist der Griff. Dafür bleiben zwei Erklärungen, und die vorregistrierte
Regel („in Reichweite, greift nicht → `TUNE_VISUAL=1`") unterscheidet sie nicht:

1. **Die Politik versucht keinen Griff** — die Finger bleiben starr, der Würfel wird nur
   angestoßen. Das wäre Wahrnehmung/Politik und `TUNE_VISUAL=1` wäre richtig.
2. **Der Griff rutscht** — die Finger schließen, aber Reibung/Kontaktparameter halten den Würfel
   nicht. Das wäre Sim-Physik, und ein 48-h-ViT-Lauf wäre verschwendet.

Ein 5,7-cm-Schub spricht eher für (1), beweist es aber nicht. Deshalb protokolliert der Runner
zusätzlich `finger_span_max_rad`: die größte Spannweite, die ein Fingergelenk-**Kommando**
(Action-Dims 14:28) über die Episode durchläuft. Nahe 0 heißt „die Finger wurden nie bewegt" und
entscheidet (1) direkt am Kommando, noch vor jeder Physikfrage.

**Nächster Schritt** — beides, in dieser Reihenfolge, zusammen unter 15 Minuten:

```bash
# a) Greif-Physik isoliert: Würfel exakt an die aufgezeichneten Greifpunkte, echte
#    Dataset-Aktionen, kein Modell. Wird ein Würfel angehoben?  (neue Aktion 'grasp')
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh grasp

# b) Eval erneut, jetzt mit Fingerspur
HF_TOKEN=hf_... NUM_EPISODES=2 EPISODE_LENGTH_S=40 DR_ENABLED=0 \
    ./Simulation/server_rl_run.sh eval
```

Hebt (a) den Würfel (`max_cube_lift_cm` > ~2) und ist die Fingerspanne in (b) klein →
Politik/Wahrnehmung, `TUNE_VISUAL=1` steht. Hebt (a) nichts → zuerst die Kontaktparameter, ViT
später. Zum Kontext: die Finger-Positionsgrenzen waren schon einmal die Ursache eines
fallengelassenen Würfels; `_widen_finger_joint_limits()` weitet sie seit Juni auf die echte
Dataset-Range, dieser Pfad ist also bereits abgedeckt.

#### Antwort (runs/20260808/26): der Griff scheitert auch ohne Modell

`grasp` beantwortet (a) eindeutig — und zwar in die zweite Richtung:

| Größe | Wert | Lesart |
|---|---|---|
| Arm-Tracking, mittel | **0,019 rad** | der Sim führt die aufgezeichneten Aktionen sauber aus (Schwelle 0,1) |
| worst arm joints | 0,10–0,12 rad | keine zu schwachen PD-Gains |
| min Handfläche→Würfelmitte | 6,8 cm | die Hand ist am Würfel |
| **max Würfel-Anhebung** | **0,0 cm** | **kein Würfel wird angehoben** |

Aufbau: echte Dataset-Aktionen, kein Modell, kein Server, und die Würfel per `--grasp-test`
exakt an den aufgezeichneten Greifpunkten. Unter diesen Bedingungen ist die Politik vollständig
aus der Kette entfernt — **und der Griff scheitert trotzdem.**

Damit ist entschieden:

* **Es liegt nicht am Modell und nicht an der Wahrnehmung.** Der Fehler reproduziert sich ohne
  Modell in der Schleife. `TUNE_VISUAL=1` ist vorerst vom Tisch; ein 48-h-ViT-Lauf würde gegen ein
  Ziel trainieren, das die Sim physikalisch nicht hergibt.
  *(Eingeschränkt durch Lauf 27, siehe unten: die Würfel-Positionen dieses Tests sind geschätzt,
  ein ausbleibendes Anheben kann auch an der Platzierung liegen. Die Reihenfolge — erst Sim, dann
  ViT — bleibt davon unberührt.)*
* **Die BC-Erfolgsrate misst derzeit nicht die Policy.** Die 0/2 aus Lauf 25 sind kein Befund über
  den Checkpoint. Schritt 3 taugt als „Nullpunkt für jeden RL-Vergleich" erst, wenn ein Würfel
  überhaupt angehoben werden kann.
* **RL wäre in diesem Zustand wirkungslos.** Sowohl `reward_mode=binary` als auch die
  `stack`/`height`-Terme des Shaped-Reward setzen ein Anheben voraus. FPO bekäme aus diesen
  Komponenten nie ein Signal — der Lauf liefe, ohne lernen zu können. Das erklärt den bisher
  unbewiesenen Lerneffekt zwanglos.

Erste Spur, noch nicht beweisend: der tiefste **Handflächen**punkt liegt bei z = 0,953 (links)
bzw. 0,943 (rechts), die Würfel-Oberkante bei 0,940 — die Handflächen bleiben also über dem
Würfel. Da die Finger von der Handfläche aus nach vorn zeigen, ist damit noch nicht gesagt, ob sie
die Seiten umschließen. (Die Zeile war bis Lauf 26 als „Würfel-Oberseite z≈0,915" beschriftet,
tatsächlich ist das die Würfel-*Mitte* — dieselbe Sorte Beschriftungsfehler wie bei der Handwurzel
in Lauf 24, jetzt korrigiert.)

Deshalb misst `grasp` ab sofort zusätzlich, ob die Finger überhaupt schließen:

| Feld in `sim_results_replay/results.json` | Bedeutung |
|---|---|
| `mean_finger_tracking_error_rad` | folgen die Fingergelenke ihrem Kommando? |
| `finger_span_commanded_rad` / `finger_span_achieved_rad` | Greifbewegung kommandiert vs. tatsächlich gefahren |

**Nächster Schritt:** `grasp` noch einmal (rund 5 Minuten, kein Modell nötig). Bleibt
`finger_span_achieved_rad` deutlich hinter `finger_span_commanded_rad` zurück, klemmt eine
Gelenkgrenze die Greifbewegung ab und `_widen_finger_joint_limits()` greift nicht wie gedacht.
Sind beide Spannen groß, schließen die Finger — dann liegt es an Kontakt/Reibung oder daran, dass
der Würfel nicht zwischen den Fingern liegt. Gegen reine Reibung spricht dabei die Konfiguration:
die Würfel wiegen 50 g bei `static_friction=3.0` / `dynamic_friction=2.5`.

#### Nachmessung (runs/20260808/27): die Finger schließen — die Gelenkgrenze ist es nicht

| Größe | Wert | Lesart |
|---|---|---|
| Finger-Tracking, mittel | 0,043 rad | die Fingergelenke folgen ihrem Kommando |
| Fingerspanne kommandiert / erreicht | **2,09 / 2,10 rad** | die Greifbewegung wird vollständig gefahren |
| Finger-Tracking, max | 0,77 rad | einzelner Ausschlag — Kontakt oder Transiente, nicht dauerhaft |
| min Handfläche→Würfelmitte | 6,8 cm | unverändert |
| max Würfel-Anhebung | 0,0 cm | unverändert |

Damit ist der erste Zweig der Entscheidungsregel erledigt: `erreicht ≥ kommandiert`, also klemmt
**keine Gelenkgrenze** die Greifbewegung ab, und `_widen_finger_joint_limits()` arbeitet wie
vorgesehen. Die Hand öffnet und schließt über volle 2,1 rad.

Übersehen wurde bis hierher die aussagekräftigste Zeile des Laufs — die Würfel-XY am Ende:

| Würfel | gesetzt | am Ende | verschoben |
|---|---|---|---|
| links | (0,35 / 0,20) | (0,35 / 0,17) | 3,0 cm |
| rechts | (0,37 / −0,16) | (0,37 / −0,11) | 5,0 cm |
| Mitte | (0,35 / 0,00) | (0,37 / 0,03) | 3,6 cm |

Alle drei werden **angefasst und weggeschoben**, keiner wird angehoben — in Lauf 26 und 27
millimetergleich. Das ist die Signatur einer Hand, die den Würfel im Vorbeifahren wegstößt, nicht
die einer Hand, aus der er herausrutscht.

**Korrektur zu Lauf 26.** Die dortige Formulierung „der Griff scheitert auch ohne Modell" ist
schwächer, als sie klingt. Die Würfel-Positionen des Greif-Tests sind eine *Schätzung*
(tiefster Handflächenpunkt + 5 cm in +x, hart kodiert in
[`run_g1_dex3_replay.py`](../../Simulation/g1_dex3_sim/run_g1_dex3_replay.py)); das Dataset
speichert keine Objekt-Posen. Der Würfel liegt außerdem die ganze Episode dort, obwohl die Hand
den Punkt nur einmal passiert. Ein ausbleibendes Anheben kann deshalb genauso gut an Ort und
Zeitpunkt der Platzierung liegen wie an der Greif-Physik. Belastbar aus Lauf 26/27 bleibt: Arme
und Finger fahren die aufgezeichnete Trajektorie sauber ab, und die Würfel werden berührt.

#### Der Test ohne Platzierungs-Annahme: `GRASP_MODE=hold`

Um die beiden Erklärungen zu trennen, setzt der Replay den Würfel jetzt wahlweise **im Moment des
Zugreifens genau zwischen die drei Fingerspitzen**. Der Mittelpunkt des Fingerdreiecks *ist* die
Greiföffnung, und der Auslöser ist die Greifbewegung selbst (Öffnung fällt unter 70 % ihrer
bisher größten Weite) — Ort und Zeitpunkt sind damit per Konstruktion richtig, statt geschätzt.

```bash
HF_TOKEN=hf_... GRASP_MODE=hold ./Simulation/server_rl_run.sh grasp
```

| Feld in `sim_results_replay/results.json` | Bedeutung |
|---|---|
| `hold_ratio` | Anteil der Steps nach dem Einsetzen, in denen der Würfel < 4 cm am Fingerdreieck bleibt |
| `hold_rise_cm` | maximale Höhe über dem Einsetzpunkt — die Hand trägt ihn nach oben |
| `hold_final_z` | Endhöhe; ≈ 0,915 heißt: auf den Tisch gefallen |
| `min_fingertip_cube_dist_cm` / `min_fingertip_step` | Abstand ab den **Fingerspitzen** (nicht der Handfläche), mit Zeitpunkt |
| `finger_spread_min_cm` / `finger_close_step` | engste Greiföffnung je Hand und wann sie eintritt |

**Entscheidungsregel:** `hold_ratio` > ~0,5 → die Greif-Physik trägt, und das Problem ist die
Platzierung des Tests (bzw. im Closed Loop: die Politik trifft den Würfel nicht). `hold_ratio` ≈ 0
bei Endhöhe ≈ Tischauflage → der Würfel rutscht aus der geschlossenen Hand, dann sind Kontakt und
Reibung dran. Gegen letzteres spricht weiterhin die Konfiguration: 50 g bei `static_friction=3.0` /
`dynamic_friction=2.5`.

Unabhängig vom Modus laufen `min_fingertip_step` und `finger_close_step` ab sofort mit. Fallen die
beiden Zeitpunkte weit auseinander, greift die Hand ins Leere — dann ist die Reihenfolge
„erst Platzierung, dann Physik" ohnehin die richtige.

#### Lauf 28: der Messpunkt war zum dritten Mal falsch

`GRASP_MODE=hold` meldete für beide Hände „kein Zugreifen erkannt — Finger schließen nie unter
70 % ihrer größten Öffnung", bei einer engsten Greiföffnung von 7,4 cm (links) und 7,6 cm
(rechts). Das widerspricht der Fingerspanne aus demselben Lauf: 2,09 rad kommandiert, 2,09 rad
erreicht. Eine Hand, deren Beugegelenke 120° durchfahren, kann ihre Fingerkuppen nicht nahezu
still halten — also stimmte die Messung nicht.

Die Gegenprobe lief ohne Sim, direkt auf der mitgelieferten Aktionsdatei
[`replay_episode0.npz`](../../Simulation/g1_dex3_sim/replay_episode0.npz):

| | linke Hand | rechte Hand |
|---|---|---|
| stärkste Beugung (aufgezeichnet) | Step 330 | Step 186 |
| Beugung > 80 % | Steps 320–831 | Steps 102–203 |
| engste Greiföffnung (gemessen) | Step 36 | Step 109 |

Links liegen 294 Steps (rund 10 s) zwischen dem stärksten Zugreifen und dem, was die Diagnose als
engste Öffnung ausgab. Die Ursache steht in der URDF: der Frame eines distalen Fingerglieds sitzt
**im Gelenk**, und das Beugen dieses Gelenks **dreht den Frame nur** — sein Ursprung wandert nicht.
Die Kollisionsmesh reicht von dort noch **5,2 cm** weiter (STL-Bounding-Box, `index_1`/`middle_1`
lokal +x, `thumb_2` lokal ∓y). Gemessen wurde also das letzte Fingergelenk, und der Griff selbst
war in den Zahlen unsichtbar.

Das ist derselbe Fehler zum dritten Mal — Handwurzel (Lauf 24), Handfläche (Lauf 26), distales
Gelenk (Lauf 25/28). **Regel: ein Körper-Frame ist keine Kontaktfläche.** Wer den Abstand zu einem
5-cm-Würfel misst, darf nicht 5,2 cm vor der Kuppe anfangen.

**Was dadurch neu zu lesen ist:** die „Fingerspitzen"-Abstände aus Lauf 25 (4,0 / 4,2 cm zur
Würfelmitte) sind Abstände ab dem distalen Gelenk. Um die Kuppenlänge korrigiert liegen die echten
Kontaktflächen im Würfel — die Politik war also in Kontaktreichweite, nicht 1,5 cm davor. Die
Aussage „die Hand ist am Würfel" wird dadurch stärker, nicht schwächer; die 0,0 cm Anhebung bleibt
das Rätsel.

**Behoben in** [`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py):
`get_contact_points_w()` liefert jetzt die Kuppen — lokaler Versatz, mit der Körper-Orientierung
mitgedreht (`quat_apply`), sonst zeigte er beim gebeugten Finger in die falsche Richtung. Eval,
Replay und Greif-Test ziehen ihre Kontaktpunkte aus dieser einen Quelle, damit nicht wieder drei
Stellen drei verschiedene Punkte messen. `results.json` bekommt zusätzlich
`finger_spread_max_cm`, weil „nie unter 70 % des Maximums" ohne das Maximum nicht lesbar ist.

**Nächster Schritt:** beide Messungen mit dem korrigierten Bezugspunkt wiederholen —

```bash
HF_TOKEN=hf_... GRASP_MODE=hold ./Simulation/server_rl_run.sh grasp
```

Erst wenn `finger_spread_min_cm` mit der Beugung aus der Tabelle oben zusammenfällt, misst die
Diagnose, was sie behauptet. Danach entscheidet `hold_ratio` wie oben beschrieben.

#### Falscher Alarm „Sim-Eval ohne Erfolgsmarker beendet"

Lauf 24 meldete das direkt nach einem sauberen `[eval] fertig.` — der Marker war da. Ursache war
`… | tee /dev/stderr | grep -q "\[eval\] fertig"` unter `set -o pipefail`: `grep -q` steigt beim
ersten Treffer aus, `tee` bekommt beim nächsten Schreibversuch SIGPIPE, und `pipefail` reicht
dessen Status 141 als Pipeline-Ergebnis durch. Isaac Sim schreibt nach dem Marker noch
Shutdown-Zeilen, deshalb traf es `eval` und nicht `dump`/`gap`, wo der Marker die letzte Zeile
ist. `grep -c … >/dev/null` liest bis EOF und kann nicht früher schließen; alle drei Stellen in
[`server_rl_run.sh`](../../Simulation/server_rl_run.sh) sind umgestellt.

### `isaaclab nicht importierbar`
Das Skript braucht das **kombinierte** Image (`Dockerfile.vastai`), nicht das BC-Trainingsimage.

### Erfolgsrate bleibt 0, Reward explodiert
Reward-Hacking — `RL_KL_COEF` erhöhen oder die Shaped-Reward-Gewichte (`rew_*` in
[`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py)) nachjustieren.

### `torch.OutOfMemoryError` im Update-Schritt

**Zuerst prüfen, wer sonst noch auf der GPU liegt** — `nvidia-smi`. Auf `ikr-ki-server-01` belegte
am 2026-08-08 ein fremder `llama-server` dauerhaft **41 GB auf GPU 0** und 37 GB auf GPU 1; von den
95 GB blieben also nur ~56 übrig. Die Fehlermeldung nennt das als „non-PyTorch memory" und ist
leicht zu überlesen. Größter Hebel ist immer, diesen Prozess zu beenden. Sonst: die freiere Karte
in `RL_GPUS` **nach vorn** stellen (sie wird zu `cuda:0` und trägt Rendering + Training) — das
Referenzmodell landet dann auf der anderen.

**`RL_NUM_ENVS` zu senken half hier nicht — es machte es schlimmer.** Der Grund steckt in der
Struktur des Update-Schritts: ein Minibatch besteht aus `(t, env)`-Paaren, und pro Zeitschritt
darin läuft ein eigener Forward. Bei kleinerem `RL_NUM_ENVS` passen *mehr Zeitschritte* in
dasselbe Minibatch, also laufen mehr Forwards. Mit `RL_MINIBATCH_SIZE=64` und `RL_NUM_ENVS=4`
sind das 16 Zeitschritte × `RL_FPO_MC_SAMPLES=4` = **64 Forward-Graphen**, die bis zum `backward()`
gleichzeitig im Speicher lagen. Seit 2026-08-08 rechnet der Trainer jede Zeitschritt-Gruppe
einzeln zurück (Gradienten-Akkumulation, mathematisch identisch) — das senkt den Spitzenbedarf
um den Faktor `RL_MINIBATCH_SIZE / RL_NUM_ENVS`.

Reicht das nicht, in dieser Reihenfolge drehen:

| Variable | Notfallwert | Wirkung |
|---|---|---|
| `RL_FPO_MC_SAMPLES` | `2` statt `4` | Halbiert Speicher **und** Rechenzeit. Die K Ziehungen müssen bis zur Bildung von `exp(mean_K(new) − old)` alle im Graphen bleiben, lassen sich also nicht einzeln zurückrechnen — deshalb der erste Hebel |
| `RL_MINIBATCH_SIZE` | `16` oder `8` | Weniger Paare je Update-Schritt |
| `RL_NUM_ENVS` | `2` | Kleinere Batches beim Rendern und im Rollout — wirkt auf den Rollout, **nicht** auf den Update-Peak |
| `RL_EPOCHS_PER_ITER` | `1` | Halbiert die Update-Arbeit je Iteration (kostet Sample-Effizienz) |
| `RL_GPUS` | `'"device=1,0"'` | Zweite Karte durchreichen — das Referenzmodell zieht dorthin um und gibt ~6–7 GB auf der Trainingskarte frei (Default; `RL_REF_DEVICE=same` schaltet es ab) |

`PYTORCH_ALLOC_CONF=expandable_segments:True` setzt `server_rl_run.sh` inzwischen selbst — der
OOM-Traceback empfahl es (1,13 GB waren reserviert, aber unbenutzt: Fragmentierung).

### Zu langsam
Wanduhrzeit zwischen zwei `[rl] iter`-Zeilen messen. Eine Iteration kostet grob
`rollout_steps × (1 + K)` Forwards im Rollout plus `epochs × minibatches × Zeitschritte × 2K`
im Update — `RL_FPO_MC_SAMPLES` und `RL_EPOCHS_PER_ITER` sind daher auch hier die stärksten Hebel.

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
