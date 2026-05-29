# Closed-Loop-Simulation für G1 + Dex3 in Isaac Lab — Implementierungsplan

Ziel: Den feingetunten GR00T-N1.6-Checkpoint in einer **Closed-Loop-Simulation**
(Isaac Lab / Isaac Sim) auf der Block-Stacking-Aufgabe evaluieren — d.h. die
Policy steuert den simulierten G1+Dex3 in der Schleife, statt nur Aktionen offline
mit Ground-Truth zu vergleichen.

## 0. Ausgangs-Spec (aus `examples/G1_DEX3/g1_dex3_config.py`)

Die Sim muss exakt diese Observation/Action-Schnittstelle bedienen:

**Observation (pro Schritt):**
- **4 RGB-Kameras:** `cam_left_high`, `cam_right_high`, `cam_left_wrist`, `cam_right_wrist`
- **State (28-dim):** `left_arm[0:7]`, `right_arm[7:14]`, `left_dex3[14:21]`, `right_dex3[21:28]`
- **Language:** `annotation.human.task_description` (z. B. „stack the blocks")

**Action (28-dim, Chunk = 16 Schritte @ 30 Hz ≈ 0.53 s):**
- `left_arm` / `right_arm`: **RELATIVE** Joint-Position-Deltas (auf aktuelle Gelenkstellung addieren)
- `left_dex3` / `right_dex3`: **ABSOLUTE** Joint-Position-Targets (direkte Winkel in rad)

Joint-Reihenfolge (muss im Sim-Actuator-Mapping exakt übereinstimmen):
```
left_arm : ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw
right_arm: (analog)
left_dex3: Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1
right_dex3: Thumb0, Thumb1, Thumb2, Index0, Index1, Middle0, Middle1   ← Reihenfolge rechts ≠ links!
```

> ⚠️ **Domain-Gap:** Das Modell wurde auf **echten Teleoperations-Daten** trainiert.
> Closed-Loop-Sim-Erfolg hängt stark davon ab, wie nah Kamera-Optik, -Posen und
> -Intrinsics an der realen Trainingsverteilung liegen. Erwartet keine 1:1-Übertragung;
> die Sim ist v. a. ein qualitatives/relatives Bewertungswerkzeug.

## 1. Architektur

Zwei entkoppelte Prozesse, kommunizieren über ZMQ (localhost):

```
┌─────────────────────────────┐         ZMQ :5555          ┌──────────────────────────────┐
│  GR00T-Policy-Server         │ ◀───── obs (4 imgs+state) ─│  Isaac-Lab-Sim-Client        │
│  run_gr00t_server.py         │ ─────▶ action chunk (16×28)│  g1_dex3_blockstack_env.py    │
│  (euer SIF, GPU)             │                            │  (Isaac Sim 4.x, eigene Env)  │
└─────────────────────────────┘                            └──────────────────────────────┘
```

- **Policy-Server** = bestehendes [gr00t/eval/run_gr00t_server.py](app/Groot-1.6/gr00t/eval/run_gr00t_server.py),
  unverändert. Lädt euren Checkpoint mit `--embodiment-tag new_embodiment`
  `--embodiment-config-module examples.G1_DEX3.g1_dex3_config`.
- **Sim-Client** = neu zu schreiben. Nutzt `gr00t.policy.server_client` als Client.

Vorteil: Isaac Lab und GR00T haben **inkompatible Dependency-Stacks** (Isaac Sim
bringt eigenes PyTorch/USD-Universum mit). Trennung in zwei Prozesse/Container
vermeidet die Dependency-Hölle. Derselbe Server bedient später auch den echten Roboter.

## 2. Laufumgebung auf KISSKI

Isaac Lab ist **nicht** im aktuellen SIF. Optionen:

| Variante | Aufwand | Empfehlung |
|---|---|---|
| NVIDIA-Isaac-Lab-Apptainer-Image (`nvcr.io/nvidia/isaac-lab`) → SIF pullen | mittel | ✅ bevorzugt |
| Isaac Lab via pip/uv in frische venv auf scratch | hoch (Treiber/Vulkan-Fummelei) | nur Fallback |

- **Headless-Rendering** auf A100/H100: EGL-Pfad. Das mitgelieferte
  [scripts/eval/check_sim_eval_ready.py](app/Groot-1.6/scripts/eval/check_sim_eval_ready.py)
  zeigt das nötige Vulkan-/EGL-ICD-Setup (`nvidia_icd.json`, `10_nvidia.json`) —
  als Vorlage für die Sim-Umgebung verwenden.
- Beide Prozesse im selben SLURM-Job auf demselben Node starten (Server im
  Hintergrund, dann Client), Kommunikation über `localhost:5555`.

## 3. Roboter-Asset (kritischer Pfad)

Isaac Lab liefert eine `G1`-Konfiguration, aber i. d. R. **mit Standardhänden, nicht Dex3**.

1. **Dex3-Hand-USD beschaffen/bauen:** Dex3-URDF (aus `unitree_ros` / `unitree_sdk2`)
   per Isaac-URDF-Importer → USD konvertieren.
2. **An G1 attachen:** Dex3-Hände an die Wrist-Links des G1-USD montieren →
   kombiniertes `g1_dex3.usd`.
3. **Articulation-Config** (`G1_DEX3_CFG`) in Isaac Lab anlegen: Actuator-Gruppen,
   PD-Gains, Joint-Limits. Arm-Joints positionsgeregelt.
4. **Unterkörper fixieren:** Für Tabletop-Manipulation Beine/Waist locken oder
   `fix_root_link=True` (stehender Roboter am Tisch). Kein Whole-Body-Balancing nötig.

## 4. Kameras (zweitkritischer Pfad)

4 Kameras matchen, Auflösung/FOV/Posen aus dem realen Dataset ableiten:

- `cam_left_high`, `cam_right_high`: feste externe/Kopf-Kameras → aus den Dataset-Videos
  Blickwinkel rekonstruieren.
- `cam_left_wrist`, `cam_right_wrist`: an die jeweiligen Wrist-Links parenten.
- **Intrinsics/Auflösung** auf die Trainingsbilder abstimmen (sonst sieht das VLA
  eine fremde Verteilung). Referenz: ein Frame jeder Kamera aus
  `G1_Dex3_BlockStacking_Dataset` ziehen und Sim-Kamera daran kalibrieren.

## 5. Szene & Task

- Tisch + N Würfel (z. B. 3) mit randomisierten Startpositionen in Reichweite.
- **Erfolgsmetrik:** Turm gestapelt = Würfel-Schwerpunkte vertikal ausgerichtet,
  Höhe > Schwellwert, Geschwindigkeit ~0 (stabil). Pro Episode bool success.
- Reset: Würfel neu samplen, Roboter in Home-Pose, State zurücksetzen.
- Episodenlänge: z. B. 600 Steps (20 s @ 30 Hz).

## 6. Control-Loop (Kern des Sim-Clients)

```
reset() → home pose, sample blocks
loop bis done:
    obs = {4× RGB rendern, state = 28 Joint-Pos lesen, language}
    chunk = client.get_action(obs)         # (16, 28) vom GR00T-Server
    for t in range(execution_horizon):     # z. B. 8 von 16 Schritten ausführen, dann re-plan
        target_arms  = current_arm_pos + chunk[t, 0:14]     # RELATIVE → addieren
        target_hands = chunk[t, 14:28]                      # ABSOLUTE → direkt
        set_joint_position_targets(target_arms, target_hands)
        physics_step()  × (sim_dt-Verhältnis, z. B. 200 Hz physics / 30 Hz policy)
    check_success(); record_video_frame()
```

- **Re-plan-Horizont** (`execution_horizon`) tunen: 8 ist ein guter Startwert
  (halber Chunk). Server-Parameter, muss zu Client-Loop passen.
- PD-Targets, nicht Torque: Arme positionsgeregelt, Hände positionsgeregelt.

## 7. Eval-Harness wiederverwenden

Statt von Null: die Struktur der vorhandenen Sim-Envs als Vorlage nehmen —
[gr00t/eval/sim/LIBERO/libero_env.py](app/Groot-1.6/gr00t/eval/sim/LIBERO/libero_env.py)
und die Wrapper:
- [wrapper/multistep_wrapper.py](app/Groot-1.6/gr00t/eval/sim/wrapper/multistep_wrapper.py)
  — Action-Chunk-Ausführung
- [wrapper/video_recording_wrapper.py](app/Groot-1.6/gr00t/eval/sim/wrapper/video_recording_wrapper.py)
  — automatische Video-Aufzeichnung

Die G1-Dex3-Env als Gym-Env mit derselben Schnittstelle bauen → Success-Rate über
N Episoden + Rollout-Videos fallen automatisch an.

## 8. Vorgeschlagene Datei-/Code-Struktur (im Fork `lucam06/Isaac-GR00T`)

```
app/Groot-1.6/gr00t/eval/sim/G1_DEX3/
├── setup_isaaclab.sh          # Isaac-Lab-Env + EGL/Vulkan-Setup (Vorlage: check_sim_eval_ready.py)
├── assets/
│   ├── g1_dex3.usd            # kombiniertes Robotermodell
│   └── g1_dex3_cfg.py         # G1_DEX3_CFG Articulation-Config (Actuators, Gains, Limits)
├── g1_dex3_blockstack_env.py  # Gym-Env: Szene, Kameras, Success, reset/step
└── run_g1_dex3_sim_eval.py    # Client: startet/verbindet Server, Rollout-Loop, Metriken
```

## 9. Phasen & Reihenfolge (mit Abnahmekriterium je Phase)

| Phase | Inhalt | Fertig wenn … |
|---|---|---|
| **A** | Isaac-Lab-Env auf KISSKI lauffähig (headless EGL) | leere Szene rendert ein RGB-Bild |
| **B** | `g1_dex3.usd` importiert, Articulation lädt | Roboter steht, Arme per Skript fahrbar |
| **C** | 4 Kameras + Joint-State-Reader | obs-Dict hat korrektes Format/Shapes |
| **D** | GR00T-Server ↔ Client end-to-end | `client.get_action(obs)` liefert (16,28) |
| **E** | Control-Loop + relative/absolute-Mapping | Roboter bewegt sich plausibel auf Policy-Aktionen |
| **F** | Block-Szene + Success-Metrik + Video-Wrapper | Success-Rate + Rollout-Video über N Episoden |
| **G** | Domain-Gap-Tuning (Kamera-Kalibrierung, Licht, ggf. Domain-Randomization) | Success-Rate stabilisiert |

**Realistische Einschätzung:** Phasen A–C (Asset + Kameras) sind der Großteil des
Aufwands. D–E sind dank vorhandenem Server/Client klein. F–G sind iterativ.

## 10. Voraussetzung vor Phase A

Zuerst **Open-Loop-Eval** ([gr00t/eval/open_loop_eval.py](app/Groot-1.6/gr00t/eval/open_loop_eval.py))
auf dem Checkpoint laufen lassen. Wenn die per-Joint-MSE schon schlecht ist, ist
das Modell nicht reif für eine Sim und der Aufwand A–G lohnt nicht.
