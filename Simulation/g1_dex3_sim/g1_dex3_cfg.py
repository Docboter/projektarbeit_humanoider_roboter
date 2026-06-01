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

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

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
    -0.59676,  1.01160,  0.05485,  0.0,     -0.01227,  0.0,     -0.01201,
    # right_dex3 [thumb0,thumb1,thumb2, index0,index1, middle0,middle1]
    -0.70991, -1.01463, -0.21031,  0.0,      0.01391,  0.0,      0.03354,
]
# HINWEIS: middle_0/index_0 beider Hände wurden von ihren Dataset-Werten (links +0.169/+0.163,
# rechts -0.171/-0.142) auf 0.0 gesetzt — die USD-Gelenklimits sind dort EINSEITIG und mit
# umgekehrtem Vorzeichen (links [-1.571,0], rechts [0,1.571]). Das deutet auf eine
# Vorzeichen-/Achsen-Konventions-Diskrepanz Dataset↔USD bei den Dex3-Fingern hin, die auch
# die Finger-AKTIONEN im Rollout betrifft (Greifen!) → offener Punkt, separat zu untersuchen.

# ---------------------------------------------------------------------------
# Articulation-Konfiguration
# ---------------------------------------------------------------------------

G1_DEX3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Pfad zum kombinierten G1+Dex3 USD-Asset.
        # Wird beim Env-Start per asset_path-Argument überschreibbar gemacht.
        usd_path="/data/assets/g1_dex3.usd",
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
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
        # Arme: positionsgeregelt (PD-Controller)
        # GR00T-Aktionen sind RELATIVE Deltas → werden im Control-Loop auf
        # aktuelle Position addiert und dann als Target gesetzt.
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=LEFT_ARM_JOINTS,
            effort_limit=300.0,
            velocity_limit=5.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_ARM_JOINTS,
            effort_limit=300.0,
            velocity_limit=5.0,
            stiffness=100.0,
            damping=10.0,
        ),
        # Hände: positionsgeregelt (ABSOLUTE Targets aus GR00T-Aktionen)
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=LEFT_DEX3_JOINTS,
            effort_limit=5.0,
            velocity_limit=3.0,
            stiffness=20.0,
            damping=2.0,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_DEX3_JOINTS,
            effort_limit=5.0,
            velocity_limit=3.0,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)

# ---------------------------------------------------------------------------
# Kamera-Parameter (aus dem realen Dataset abgeleitet)
# ---------------------------------------------------------------------------

def look_at_world_quat(eye, target, world_up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) für eine Kamera in Isaac-Lab-``convention="world"``.

    In dieser Konvention ist die **Blickachse +X** und **oben +Z**. Die Funktion
    richtet die Kamera so aus, dass sie von ``eye`` auf ``target`` blickt — damit lassen
    sich die Posen über anschauliche Punkte statt undurchsichtiger Quaternionen tunen.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(world_up, dtype=float)

    x = target - eye                      # Blickrichtung = Kamera-+X
    x /= np.linalg.norm(x)
    z = up - np.dot(up, x) * x            # Kamera-oben (+Z) = Welt-oben ⟂ Blickachse
    if np.linalg.norm(z) < 1e-6:          # Blick exakt vertikal → Ersatz-up
        z = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)                    # rechtshändig: x × y = z

    R = np.column_stack([x, y, z])        # Spalten = Kamera-Achsen im Weltframe
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * w)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * w)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * w)
    return (float(w), float(qx), float(qy), float(qz))


@configclass
class G1Dex3CameraCfg:
    """Kamera-Positionen und Intrinsics — möglichst nah an den Trainingsdaten."""

    # Auflösung der Kamerabilder (anpassen auf die tatsächliche Dataset-Auflösung)
    width: int = 640
    height: int = 480

    # Horizontaler FOV in Grad (typisch für RealSense D435 / ZED2)
    hfov_deg: float = 69.0

    # Kamera-Posen (pos in Meter, rot als (w, x, y, z) Quaternion, World-Frame)
    # WICHTIG: Aus den Dataset-Videos rekonstruieren! Dies sind Schätzwerte.
    cam_left_high: dict = None
    cam_right_high: dict = None
    # Wrist-Kameras werden relativ zum Wrist-Link definiert (lokal)
    cam_left_wrist_local: dict = None
    cam_right_wrist_local: dict = None
    # Szenen-Übersichtskamera — NUR fürs Video, NICHT Policy-Observation
    cam_scene: dict = None

    def __post_init__(self):
        # Externe Kameras: Kopf-Stereo-Paar, blickt von vorne-oben NACH UNTEN auf den Tisch.
        # Rekonstruiert aus den Dataset-Frames (Simulation/camera_reference/dataset_cam_*_high.png):
        # beide Hände kommen von unten ins Bild, Tisch füllt die unteren ~2/3.
        # Rotation per Look-at auf die Tischmitte — frühere (0.924,-0.383,0,0) war eine reine
        # Roll-Drehung um die +X-Blickachse (Bild verkippt, kein Pitch) und zeigte auf den Boden.
        #
        # Tisch-Oberfläche ~ (0.5, 0.0, 0.74); Roboter-Pelvis bei z=0.85 → Kopf ~ z=1.4, x≈0.
        # Werte sind Startschätzung — gegen die Referenz-Frames iterativ verfeinern.
        # Ziel auf den neuen, erreichbaren Würfelbereich (Tisch angehoben auf Oberseite 0.87,
        # Würfel bei x≈0.35, z≈0.90). Vorher (0.5,0,0.73) — zeigte auf den zu tiefen Alt-Tisch.
        high_target = (0.40, 0.0, 0.86)
        left_high_eye = (0.0, 0.06, 1.40)
        right_high_eye = (0.0, -0.06, 1.40)
        self.cam_left_high = {
            "pos": left_high_eye,
            "rot": look_at_world_quat(left_high_eye, high_target),
        }
        self.cam_right_high = {
            "pos": right_high_eye,
            "rot": look_at_world_quat(right_high_eye, high_target),
        }
        # Wrist-Kameras: am jeweiligen Wrist-Yaw-Link montiert (convention="world", Link-Frame).
        # Die Hand ragt entlang +X (Palm-Joint bei x=0.0415, Finger weiter bei +X). Look-at von
        # hinter/über dem Wrist-Origin (raus aus dem Palm-Mesh) auf die Fingerspitzen → die Kamera
        # zeigt garantiert auf die Hand (vorher: pos=(0.05,…) steckte IM Palm-Mesh → nur Grau;
        # rechte Cam zudem falsch herum (-X)). Roll/Feinframing nach Render-Vergleich justieren.
        # eye weiter zurück (-X) und höher (+Z), damit nicht nur die Hand formatfüllend ist,
        # sondern Tisch + Würfel hinter den Fingern sichtbar werden (wie in der Referenz).
        # Arme starten ASYMMETRISCH (Dataset-Pose) → linke Cam braucht mehr Pitch nach unten,
        # sonst zeigt sie über die Würfel hinweg (Render-Befund). Daher getrennte Targets.
        wrist_eye = (-0.08, 0.0, 0.13)
        left_wrist_target = (0.14, 0.0, -0.18)   # steiler runter → Tisch/Würfel ins Bild
        right_wrist_target = (0.16, 0.0, -0.05)
        self.cam_left_wrist_local = {"pos": wrist_eye, "rot": look_at_world_quat(wrist_eye, left_wrist_target)}
        self.cam_right_wrist_local = {"pos": wrist_eye, "rot": look_at_world_quat(wrist_eye, right_wrist_target)}

        # Szenen-Übersichtskamera (NUR fürs aufgenommene Video): zeigt die GANZE Szene —
        # Roboter (Pelvis z=0.85, Kopf ~1.4) + Tisch (x=0.5) — von schräg vorne-seitlich-oben.
        # Weltfest. eye in +X (vor dem Tisch), +Y (seitlich), +Z (oben); Blick zurück auf die Mitte.
        scene_eye = (1.8, 1.6, 1.7)
        scene_target = (0.30, 0.0, 0.80)
        self.cam_scene = {"pos": scene_eye, "rot": look_at_world_quat(scene_eye, scene_target)}


CAMERA_CFG = G1Dex3CameraCfg()
