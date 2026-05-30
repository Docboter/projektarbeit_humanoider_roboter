"""
Articulation-Konfiguration für G1 + Dex3-Hand in Isaac Lab.

Joint-Reihenfolge muss EXAKT mit g1_dex3_config.py aus dem GR00T-Fork
(examples/G1_DEX3/g1_dex3_config.py) übereinstimmen:

    left_arm  [0:7]  : ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow,
                       WristRoll, WristPitch, WristYaw
    right_arm [7:14] : (analog)
    left_dex3 [14:21]: Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1
    right_dex3[21:28]: Thumb0, Thumb1, Thumb2, Index0, Index1, Middle0, Middle1
                       ↑ Achtung: Reihenfolge rechts ≠ links!

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
    "left_dex3_thumb_joint0",
    "left_dex3_thumb_joint1",
    "left_dex3_thumb_joint2",
    "left_dex3_middle_joint0",
    "left_dex3_middle_joint1",
    "left_dex3_index_joint0",
    "left_dex3_index_joint1",
]

RIGHT_DEX3_JOINTS = [
    "right_dex3_thumb_joint0",
    "right_dex3_thumb_joint1",
    "right_dex3_thumb_joint2",
    "right_dex3_index_joint0",     # ← Reihenfolge rechts: Index vor Middle
    "right_dex3_index_joint1",
    "right_dex3_middle_joint0",
    "right_dex3_middle_joint1",
]

# Alle 28 Joints in der korrekten Reihenfolge (entspricht dem Action-Vektor)
ALL_JOINTS_ORDERED = (
    LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_DEX3_JOINTS + RIGHT_DEX3_JOINTS
)

# ---------------------------------------------------------------------------
# Articulation-Konfiguration
# ---------------------------------------------------------------------------

G1_DEX3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Pfad zum kombinierten G1+Dex3 USD-Asset.
        # Wird beim Env-Start per asset_path-Argument überschreibbar gemacht.
        usd_path="/workspace/assets/g1_dex3.usd",
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
        joint_pos={
            # Arm-Ruheposition (leicht angewinkelt, Hände über Tisch)
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.8,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.8,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # Finger gestreckt
            ".*dex3.*": 0.0,
        },
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
        "left_dex3": ImplicitActuatorCfg(
            joint_names_expr=LEFT_DEX3_JOINTS,
            effort_limit=5.0,
            velocity_limit=3.0,
            stiffness=20.0,
            damping=2.0,
        ),
        "right_dex3": ImplicitActuatorCfg(
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

    def __post_init__(self):
        # Externe Kameras (Kopf/Schulterhöhe), fest in der Welt
        self.cam_left_high = {
            "pos": (-0.5, 0.3, 1.4),
            "rot": (0.924, -0.383, 0.0, 0.0),  # ~45° nach unten geneigt
        }
        self.cam_right_high = {
            "pos": (-0.5, -0.3, 1.4),
            "rot": (0.924, -0.383, 0.0, 0.0),
        }
        # Wrist-Kameras: lokal am Wrist-Link (nach vorne schauend)
        self.cam_left_wrist_local = {
            "pos": (0.05, 0.0, 0.0),
            "rot": (0.707, 0.0, 0.707, 0.0),  # 90° zur Seite
        }
        self.cam_right_wrist_local = {
            "pos": (0.05, 0.0, 0.0),
            "rot": (0.707, 0.0, -0.707, 0.0),
        }


CAMERA_CFG = G1Dex3CameraCfg()
