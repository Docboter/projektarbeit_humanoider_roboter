# TL;DR: Articulation-Config (Joints, Aktuatoren) für G1+Dex3; Kamera-Teil aus camera_geometry.py.
"""
Articulation-Konfiguration für G1 + Dex3-Hand in Isaac Lab.

Joint-Reihenfolge muss EXAKT mit g1_dex3_config.py aus dem GR00T-Fork
(examples/G1_DEX3/g1_dex3_config.py) übereinstimmen:

    left_arm  [0:7]  : ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow,
                       WristRoll, WristPitch, WristYaw
    right_arm [7:14] : (analog)
    left_dex3 [14:21]: Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1
    right_dex3[21:28]: Thumb0, Thumb1, Thumb2, Index0, Index1, Middle0, Middle1
                       ↑ Achtung: Reihenfolge rechts ≠ links (Quelle: dataset info.json)!

Voraussetzung:
    Das kombinierte USD-Asset g1_dex3.usd muss vorhanden sein.
    Pfad: /workspace/assets/g1_dex3.usd  (oder per --asset-path übergeben).
    Erstellung: Unitree-Dex3-URDF → Isaac-URDF-Importer → USD, dann an G1-Wrists anhängen.
    Siehe ISAAC_LAB_SIM_PLAN.md Abschnitt 3.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Kameraposen, Intrinsics und das Pinhole-Modell liegen in camera_geometry.py — sie
# brauchen kein Isaac Lab, und nur deshalb lässt sich das Kameramodell außerhalb des
# Containers gegen ein gerendertes Bild prüfen. Weiter-exportiert, damit
# `from g1_dex3_cfg import CAMERA_CFG` unverändert funktioniert.
from camera_geometry import (  # noqa: F401
    CAMERA_CFG,
    G1Dex3CameraCfg,
    PinholeCamera,
    look_at_world_quat,
    quat_to_matrix,
)

# ---------------------------------------------------------------------------
# Joint-Namen (muss mit dem USD-Asset übereinstimmen)
# ---------------------------------------------------------------------------

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Dex3-Gelenke: exakte Reihenfolge aus g1_dex3_config.py
LEFT_DEX3_JOINTS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]

RIGHT_DEX3_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",    # Achtung: rechts Index VOR Middle (≠ links)
    "right_hand_index_1_joint",    # Quelle: g1_dex3_config.py + dataset info.json
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

# Alle 28 Joints in der korrekten Reihenfolge (entspricht dem Action-Vektor)
ALL_JOINTS_ORDERED = (
    LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_DEX3_JOINTS + RIGHT_DEX3_JOINTS
)

# Roboter-Startpose = observation.state aus Frame 0 / Episode 0 des Datasets
# (unitreerobotics/G1_Dex3_BlockStacking_Dataset). Übernimmt die exakte reale Teleop-
# Startstellung, damit der erste Sim-Frame der Trainingsverteilung entspricht
# ("adjust the scene to closely match the first frame of the dataset"). Reihenfolge =
# ALL_JOINTS_ORDERED (left_arm[0:7], right_arm[7:14], left_dex3[14:21], right_dex3[21:28]).
DATASET_INIT_STATE = [
    -0.05662,  0.31445,  0.12487, -0.28736,  0.20897,  0.13095, -0.09767,  # left_arm
    -0.40259, -0.34180, -0.01584,  0.17989,  0.00290, -0.09826,  0.29865,  # right_arm
    # left_dex3 [thumb0,thumb1,thumb2, middle0,middle1, index0,index1]
    # middle_0/index_0: negiert (USD-Achse invertiert vs. Dataset-Konvention, s.u.)
    -0.59676,  1.01160,  0.05485, -0.169,   -0.01227, -0.163,   -0.01201,
    # right_dex3 [thumb0,thumb1,thumb2, index0,index1, middle0,middle1]
    # index_0/middle_0: negiert (s.u.)
    -0.70991, -1.01463, -0.21031,  0.171,    0.01391,  0.142,    0.03354,
]
# middle_0/index_0 beider Hände: USD-Achse invertiert vs. Dataset.
# Dataset: links middle_0/index_0 positiv = schließen; USD: [-1.571, 0], d.h. negativ = schließen.
# Rechts umgekehrt: Dataset negativ = schließen, USD [0, 1.571] positiv = schließen.
# Fix: _pre_physics_step negiert diese 4 Aktionen (Policy-Indices 17, 19, 24, 26)
# und die Init-Pose wurde entsprechend mit negiertem Dataset-Wert gesetzt.

# ---------------------------------------------------------------------------
# Articulation-Konfiguration
# ---------------------------------------------------------------------------

G1_DEX3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Pfad zum kombinierten G1+Dex3 USD-Asset (Default: schwarzhändiges Asset, Domain-Gap-Fix).
        # Wird beim Env-Start per asset_path-Argument überschrieben; die Launcher (entrypoint_sim.sh,
        # kisski_*_submit.sh) stellen das schwarzhändige USD per Recolor sicher bzw. fallen aufs
        # Original zurück. Dieser Default greift nur, wenn die Env ohne --asset-path gestartet wird.
        usd_path="/data/assets/g1_dex3_blackhands.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            # Unterkörper fixieren: Beine/Waist werden gesperrt; Roboter steht am Tisch.
            # fix_root_link=True  → stellt den Torso fest (kein Balance-Controller nötig)
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        # Startpose 1:1 aus dem Dataset (Frame 0) — Hände greifen bereits Richtung Tisch,
        # statt der früheren generischen Ruhepose. Siehe DATASET_INIT_STATE oben.
        joint_pos=dict(zip(ALL_JOINTS_ORDERED, DATASET_INIT_STATE)),
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Arme: positionsgeregelt (PD-Controller), ABSOLUTE Targets — wie die Hände.
        # Der GR00T-Checkpoint trägt zwar use_relative_action=true, aber der Server
        # dekodiert das bereits per processor.decode_action() gegen den beobachteten
        # State zurück. _pre_physics_step() setzt die 28 Werte deshalb DIREKT als
        # Positions-Target — kein Aufaddieren auf joint_pos (das würde die
        # Verschiebung verdoppeln und die Arme wegdriften lassen).
        # Gains gemessen, nicht geschätzt: der Open-Loop-Replay mit echten
        # Dataset-Aktionen (Läufe 26/27, docs/ergebnisse/diagnose-chronik.md) ergibt
        # 0,019 rad mittleren Arm-Regelfehler bei Schwelle 0,1 — die Sim folgt den
        # aufgezeichneten Trajektorien. Wer hier K erhöht, muss D mit √K mitziehen,
        # sonst kippt der Regler ins Unterdämpfte (Jitter statt besserem Tracking).
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=LEFT_ARM_JOINTS,
            effort_limit=300.0,
            velocity_limit=20.0,
            stiffness=100.0,
            damping=10.0,
            # armature (reflektierte Rotorträgheit, Wert aus Unitree IsaacLab G1_CFG):
            # stabilisiert den impliziten PD-Regler bei hoher Stiffness numerisch
            # (verhindert Jitter), ohne die Tracking-Treue zu beeinträchtigen.
            armature=0.01,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_ARM_JOINTS,
            effort_limit=300.0,
            velocity_limit=20.0,
            stiffness=100.0,
            damping=10.0,
            armature=0.01,
        ),
        # Hände: positionsgeregelt, ABSOLUTE Targets (siehe Arme oben).
        # stiffness=60 / effort_limit=20 N·m: reale Dex3-Finger müssen ~50g Würfel gegen
        # Schwerkraft halten; mit stiffness=20/effort=5 schließen die Distal-Joints nicht
        # vollständig (per_joint_max_error Index 18/27 war 0.74/0.88 rad im Replay).
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=LEFT_DEX3_JOINTS,
            effort_limit=20.0,
            velocity_limit=3.0,
            stiffness=60.0,
            damping=4.0,
            armature=0.001,  # kleiner als Arme (Finger-Hardware), analog Unitree IsaacLab
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_DEX3_JOINTS,
            effort_limit=20.0,
            velocity_limit=3.0,
            stiffness=60.0,
            damping=4.0,
            armature=0.001,
        ),
    },
)

# ---------------------------------------------------------------------------
# Kamera-Parameter — siehe camera_geometry.py
# ---------------------------------------------------------------------------
# Ausgelagert am 2026-08-17, unverändert übernommen. Dort steht auch, warum eine
# Rückprojektion (Bild → Weltkoordinate) erst gilt, wenn die Vorwärtsrichtung gegen ein
# GERENDERTES Bild geprüft ist: der Pose-Bug aus Lauf 13 (95,6° zwischen USD-Stage und
# cam.data) kostete drei Sim-Läufe, weil diese Prüfung fehlte.
