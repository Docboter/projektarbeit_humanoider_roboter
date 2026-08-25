# Baseline-Closed-Loop-Test — stock Unitree G1 + Dex1-Greifer (UNITREE_G1)

> **TL;DR:** Vergleichs-Baseline zum DEX3-Fine-Tune: Wie gut stapelt das un-finetunte Basismodell
> `GR00T-N1.6-3B` mit dem stock-G1-Dex1-Greifer (`SIM_MODE=baseline`) denselben
> Block-Stacking-Task, in einer zur DEX3-Eval parallelen Pipeline? Vorbereitung — erster Lauf
> steht aus.

Status: Vorbereitung — erster Lauf steht aus (Stand 2026-08-18)

Vergleichs-Baseline zum DEX3-Fine-Tune: Wie gut stapelt das **un-finetunte Basismodell
`nvidia/GR00T-N1.6-3B`** mit dem **stock Unitree-G1-Greifer** (Dex1-Parallelgreifer,
vorregistriertes Embodiment `UNITREE_G1`) auf **demselben** Block-Stacking-Task?

Die Pipeline ist **vollständig parallel** zur DEX3-Eval: eigene Env, eigener Client, eigener
Entrypoint. Der bestehende DEX3-Closed-Loop bleibt **unverändert** (Default `SIM_MODE=dex3`).

---

## 1. Warum das eine OOD-Baseline ist (Erwartungsmanagement)

Das `UNITREE_G1`-Embodiment in GR00T N1.6 ist ein **Whole-Body-Loco-Manip-Setup**, kein
Tabletop-Setup:

- **State:** `left_leg, right_leg, waist, left_arm, right_arm, left_hand, right_hand`
- **Action:** `left_arm(7, RELATIVE), right_arm(7, RELATIVE), left_hand(1, ABS binär),
  right_hand(1, ABS binär), waist(ABS), base_height_command(ABS), navigate_command(ABS)`
- **Video:** genau **eine** Kamera `ego_view`
- NVIDIAs offizielle G1-Sim ist MuJoCo/robosuite + Whole-Body-Controller und kennt **kein**
  Block-Stacking (nur Apple-to-Plate / Bottle-PnP).

Quelle: `app/Groot-1.6/gr00t/configs/data/embodiment_configs.py`, Schlüssel `unitree_g1`.

Damit der Vergleich auf **demselben Task** möglich ist, wird die bestehende Isaac-Lab-
Block-Stacking-Szene wiederverwendet und der stock-G1-Greifer + das Basismodell
hineingesetzt. Konsequenzen, die im Bericht stehen müssen:

- Das Basismodell läuft **zero-shot** (kein Fine-Tuning auf Block-Stacking).
- Ein Loco-Manip-Embodiment (Beine/Waist/Navigation, eine Ego-Kamera) wird auf eine
  **fixierte Tabletop-Szene** gezwungen → stark **out-of-distribution**.
- **Realistische Erwartung: Erfolgsrate ≈ 0 %.** Das ist als *unterer Baseline-Wert* genau
  die gesuchte Vergleichszahl — **keine** Aussage „das Modell kann grundsätzlich nicht stapeln".

---

## 2. Architektur & neue Dateien

```
GR00T-Server (--embodiment-tag UNITREE_G1)  ←─ ZMQ :5555 ─→  Isaac-Lab-Sim-Client
  Modell: nvidia/GR00T-N1.6-3B (Basis)                        run_g1_gripper_sim_eval.py
```

| Datei | Zweck |
|---|---|
| `Simulation/g1_gripper_sim/g1_gripper_cfg.py` | Articulation: 14 Arm- + 4 Greifer-Joints policy-gesteuert; Beine/Waist per PD auf Default fixiert; eine `ego_view`-Kamera am `torso_link`. |
| `Simulation/g1_gripper_sim/g1_gripper_blockstack_env.py` | **Identische** Tisch-/Würfel-/Erfolgs-Logik wie DEX3 (vergleichbare Metrik); 16-DOF-Action (14 Arm + 2 Hand). |
| `Simulation/g1_gripper_sim/client_g1.py` | `UNITREE_G1`-Modality-Keys; legs/waist zero-gefüllt; Loco-Manip-Actions (waist/base_height/navigate) verworfen. |
| `Simulation/scripts/dump_unitree_g1_dims.py` | Liest die exakten Pro-Gruppe-Dims **zur Laufzeit** aus den Checkpoint-Statistiken (nichts hartkodiert). |
| `Simulation/g1_gripper_sim/run_g1_gripper_sim_eval.py` | Eval-Loop; gleiches Results-JSON-Format wie DEX3 → direkter Zahlenvergleich. |
| `Simulation/scripts/entrypoint_baseline.sh` | Server (`UNITREE_G1`) + Dim-Dump + Sim-Client; **kein** Hand-Recolor. |

**Greifer-Mapping:** Der Dex1-Greifer hat je Hand 2 symmetrische prismatische Finger-Joints
(`*_dex1_finger_joint_1/2`, Limit `[-0.02, 0.0245]`). Das `UNITREE_G1`-Embodiment behandelt
die Hand als 1-DOF-Binärsignal → beide Finger werden mit demselben (geklemmten) Target gefahren.

**Robuste Dimensionen:** Der `Gr00tSimPolicyWrapper` erwartet pro State-Gruppe ein Array der
korrekten Dimension `D`. `D` steht **nicht** in der Modality-Config, sondern ergibt sich aus den
Normalisierungs-Statistiken im Checkpoint. `dump_unitree_g1_dims.py` liest `D` zur Laufzeit aus
und schreibt es nach `/data/g1_baseline_dims.json`; der Sim-Client liest diese JSON.

---

## 3. ✅ TODO — vor dem ERSTEN Run zwingend erledigen

Diese Schritte sind **noch nicht** automatisiert/erledigt und müssen vor dem ersten Lauf passieren:

- [ ] **`data/unitree_ros`-Submodul auschecken** (URDF-Quelle):
  ```bash
  git submodule update --init data/unitree_ros
  ```
  (Bereits geschehen, falls `data/unitree_ros/robots/g1_description/` existiert.)

- [x] **USD-Asset `g1_gripper.usd` einmalig erzeugen.** ✅ **Erledigt** — das Asset liegt im Repo
  (`data/g1_gripper.usd` plus `data/configuration/g1_gripper_{base,physics,robot,sensor}.usd`,
  Base-Datei ~27 MB). Die Anleitung bleibt für eine Neuerzeugung stehen.
  Ausgangslage war: es existierte **kein** vorgefertigtes
  stock-Greifer-Asset — nur DEX3-USDs. Der Konverter braucht `isaaclab.sh`, das **nur im
  Sim-Container** existiert → **nicht** auf dem nackten Host ausführen, sondern in einem
  **lokal gestarteten Docker-Container** (genau wie das DEX3-Asset, siehe
  [vastai-anleitung.md](vastai-anleitung.md) Schritt 3b). Im Repo-Root:
  ```bash
  docker run -it --rm --gpus all --ipc=host --shm-size=8g \
    --entrypoint bash \
    -v "$(pwd)/data:/data" \
    lucam03/projekt-humanoider-roboter-sim-vastai:latest
  ```
  Im Container (nur Mesh-Import, ~2–5 min, eine lokale GPU reicht — keine RT-Cores nötig):
  ```bash
  unset VIRTUAL_ENV   # sonst nutzt isaaclab.sh das GR00T-venv statt Isaacs Python-Bundle
  ${ISAACLAB_PATH}/isaaclab.sh -p \
      /workspace/g1_dex3_sim/convert_urdf_to_usd.py \
      --headless \
      --urdf  /data/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf \
      --output /data/g1_gripper.usd
  ```
  Ergebnis liegt danach auf dem Host unter `data/g1_gripper.usd` (`/data` ist gemountet).
  Voraussetzung: `data/unitree_ros` ist ausgecheckt (vorheriger TODO-Punkt) — der Mount stellt
  die `g1_description`-Meshes im Container bereit. Das fertige `g1_gripper.usd` anschließend so
  verteilen wie das DEX3-Asset (mit ins Image baken, per Volume mounten oder ins HF-Repo legen)
  und beim Run als `ASSET_PATH` setzen.

- [ ] **Kamera-Pose `ego_view` prüfen/justieren.** Die Pose in `g1_gripper_cfg.py`
  (`ego_view_local`) ist eine **Startschätzung** (Kopfhöhe, Blick nach vorne-unten auf die
  Tischmitte). Nach dem ersten Lauf gegen `/data/sim_videos/_debug_obs_ego_view.png` abgleichen
  und ggf. nachziehen (analog zum DEX3-Kamera-Tuning, siehe umsetzungsnotizen.md).

- [ ] **Torso-/Pelvis-Höhe & Reichweite sichten.** Der Torso ist fixiert (`fix_root_link=True`,
  `pos.z=0.85`); die Beine hängen kosmetisch (Tabletop-Aufgabe). Im ersten Video prüfen, dass
  die Arme den Tisch (`z≈0.89`) erreichen; ggf. `pos.z` in `g1_gripper_cfg.py` anpassen.

- [ ] **Greifer-Richtung verifizieren.** `GRIPPER_OPEN=0.0245` / `GRIPPER_CLOSE=-0.02` als
  offen/zu angenommen. Im Debug-Video prüfen, ob „schließen" die Finger zusammenführt; bei
  invertierter Konvention die beiden Konstanten tauschen.

- [ ] **Image bauen & pushen** (kopiert `g1_gripper_sim/` + Entrypoint ins Image):
  ```bash
  ./Simulation/update_sim_image.sh --vastai
  ```

- [ ] **Dim-Dump-Vorabtest (optional, empfohlen).** Einmal lokal/Container im GR00T-venv prüfen,
  dass das Basismodell `unitree_g1`-Statistiken enthält:
  ```bash
  /app/Groot-1.6/.venv/bin/python /scripts/dump_unitree_g1_dims.py \
      --model-path /data/checkpoints/GR00T-N1.6-3B --embodiment-tag unitree_g1 \
      --out /tmp/dims.json && cat /tmp/dims.json
  ```
  Schlägt das fehl (Embodiment nicht in den Stats), bricht der Run sauber mit Hinweis ab.

---

## 4. Run ausführen (vast.ai)

GPU-Anforderung wie bei der DEX3-Eval: **Ampere+ mit RT-Cores** (L40, RTX 4090, A6000 …) —
siehe [README.md](README.md) §GPU.

Container-Env-Vars:

| Variable | Wert | Zweck |
|---|---|---|
| `SIM_MODE` | `baseline` | Aktiviert die Baseline-Pipeline (Default `dex3`). |
| `HF_TOKEN` | `hf_...` | HuggingFace-Token (Pflicht für Download). |
| `HF_CHECKPOINT_REPO` | `nvidia/GR00T-N1.6-3B` | Basismodell. |
| `ASSET_PATH` | `/workspace/assets/g1_gripper.usd` | Dex1-Greifer-USD (siehe TODO oben). |
| `NUM_EPISODES` | `20` | Eval-Episoden. |
| `SHELL_ON_ERROR` | `1` | Bei Fehler in Shell fallen (empfohlen für den ersten Lauf). |

Beispiel (Docker-Options: `--ipc=host --shm-size=16g`, `/data`-Volume mounten):

```bash
docker run --name groot-baseline --gpus all --ipc=host --shm-size=16g \
  -e SIM_MODE=baseline \
  -e HF_TOKEN=hf_... -e HF_CHECKPOINT_REPO=nvidia/GR00T-N1.6-3B \
  -e ASSET_PATH=/workspace/assets/g1_gripper.usd \
  -e NUM_EPISODES=20 -e SHELL_ON_ERROR=1 \
  -it lucam03/projekt-humanoider-roboter-sim-vastai:latest
```

Der Entrypoint (`entrypoint_baseline.sh`) führt aus: Checkpoint-Download → Asset-Check →
Dim-Dump → GR00T-Server (`UNITREE_G1`) → Isaac-Lab-Sim-Client.

---

## 5. Ergebnisse & Vergleich

| Artefakt | Pfad |
|---|---|
| Success-Rate-JSON | `/data/sim_results/results.json` (Feld `"embodiment": "unitree_g1"`, `"baseline": true`) |
| Rollout-Videos | `/data/sim_videos/episode_XXXX.mp4` |
| Debug-Ego-Frame | `/data/sim_videos/_debug_obs_ego_view.png` |
| Server-Log | `/data/logs/groot_server.log` |

Das JSON-Format ist **identisch** zur DEX3-Eval → Baseline- und Fine-Tune-Success-Rate lassen
sich direkt nebeneinanderstellen. Beim Reporting die OOD-Caveats aus §1 mitnennen.

**Regression-Sicherheit:** Ohne `SIM_MODE` (bzw. `SIM_MODE=dex3`) verhält sich der Container
exakt wie zuvor — der DEX3-Closed-Loop ist unverändert.
