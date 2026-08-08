# RL-Fine-tuning (FPO) — Schritt-für-Schritt-Anleitung

Ziel: Den feingetunten GR00T-N1.6-**BC-Checkpoint** per **Reinforcement Learning (FPO)** in der
Isaac-Lab-Block-Stacking-Sim weiter verfeinern — auf einer **RT-Core-GPU** (L40 / RTX 4090 / A6000
oder ein eigener Server mit RT-Core-GPU, z. B. RTX PRO 6000 Blackwell). RL trainiert die Policy
**in genau der Sim**, in der sie auch evaluiert wird, und optimiert direkt auf **Aufgaben-Erfolg**
statt nur Aktions-Nachahmung (Hintergrund: [reinforcement-learning-plan.md](reinforcement-learning-plan.md)).

> ## ✅ Status: Pipeline läuft end-to-end — Lernwirkung noch offen
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

### Kamerabilder praktisch leer (Isaac Sim 6.0) — DLSS-Upscaling

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

**Was es ist (Kit-Log):**

```
[Warning] [omni.rtx] DLSS increasing input dimensions:
    Render resolution of (320, 240) is below minimal input resolution of 300.
```

DLSS rendert intern auf halber Auflösung (640×480 → 320×240) und liegt damit unter seinem
eigenen Minimum. Die Env setzte bis dahin **keine** Render-Konfiguration, lief also auf den
Isaac-Sim-6.0-Defaults.

**Fix:** `_make_sim_cfg()` in
[`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) setzt
`RenderCfg(antialiasing_mode="DLAA", …)` — DLAA glättet Kanten in **Native-Auflösung**, das
Minimum entfällt damit. Die Kamera-Auflösung bleibt bei 640×480, weil sie an den Datensatz
gebunden ist; sie hochzudrehen wäre der falsche Hebel. Weil Isaac Lab 3.0 Beta ist, setzt die
Funktion nur Felder, die `RenderCfg` wirklich hat, und meldet jeden Fehlschlag **laut** — falsche
Kit-Settings schluckt Kit sonst kommentarlos.

**Verifizieren:** `./Simulation/server_rl_run.sh cams` und die PNGs unter `/data/cam_dump/`
ansehen. Erwartet wird die Startzeile `[Env] RenderCfg gesetzt: {...}` und ein Bild wie der
Juni-Referenzframe. Bleibt es leer, ist der nächste Test die Kamera-Auflösung: `CAMERA_CFG.width`
/`.height` testweise auf 1280×960 (intern dann 640×480, weit über dem Minimum) — bestätigt oder
erledigt die Auflösungs-These, auch wenn der AA-Schalter nicht gegriffen hat.

> **Folge fürs Training:** Solange die Policy-Kameras leer sind, sieht das Modell nichts — RL
> optimiert dann gegen ein blindes Modell, während `reward_mean` sich weiter bewegt (der Shaped
> Reward kommt aus Gelenkpositionen). Erst `cams` grün, dann RL starten. Die Juni-Auswertungen
> sind unberührt: die liefen auf korrekt gerenderten Kameras.

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
