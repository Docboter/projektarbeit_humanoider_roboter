# TL;DR: Articulation-Konfiguration für stock G1+Dex1-Greifer (Baseline, Embodiment UNITREE_G1).
"""
Articulation-Konfiguration für den **stock Unitree G1 mit Dex1-Parallelgreifer**
in Isaac Lab — für den Baseline-Closed-Loop-Test (un-finetuntes GR00T-N1.6-3B,
Embodiment ``UNITREE_G1``).

Dies ist die Greifer-Variante (NICHT die DEX3-3-Finger-Hand). Sie spiegelt
``g1_dex3_sim/g1_dex3_cfg.py``, ist aber vollständig eigenständig, damit der
DEX3-Pfad unverändert bleibt.

Embodiment-Layout (Quelle: app/Groot-1.6/gr00t/configs/data/embodiment_configs.py,
Schlüssel ``unitree_g1``):
    state  : left_leg, right_leg, waist, left_arm, right_arm, left_hand, right_hand
    action : left_arm(7, RELATIVE), right_arm(7, RELATIVE),
             left_hand(1, ABS binär), right_hand(1, ABS binär),
             waist(ABS), base_height_command(ABS), navigate_command(ABS)
    video  : ego_view (eine einzige Kamera)
    language: annotation.human.task_description

In der Sim werden nur **Arme + Greifer** von der Policy gesteuert; Beine/Waist
werden auf ihrer Default-Pose fixiert (Tabletop-Aufgabe, fixierter Torso). Die
Loco-Manip-Action-Dims (waist, base_height_command, navigate_command) werden im
Client verworfen — siehe client_g1.py.

USD-Asset: data/g1_gripper.usd (erzeugt aus
``g1_29dof_mode_15_with_dex1_1.urdf`` via convert_urdf_to_usd.py).
"""

from __future__ import annotations

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Joint-Namen (müssen mit dem USD-Asset / der URDF übereinstimmen)
# ---------------------------------------------------------------------------

# Arme: identische Namen wie beim DEX3-G1 (gleiches g1_description).
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

# Dex1-Parallelgreifer: je 2 prismatische Finger-Joints pro Hand (symmetrisch,
# Achsen ±y, Limit [-0.02, 0.0245]). Das UNITREE_G1-Embodiment behandelt die Hand
# als 1-DOF binäres Signal → beide Finger werden mit demselben Target gefahren.
LEFT_GRIPPER_JOINTS = ["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"]
RIGHT_GRIPPER_JOINTS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]

# Beine + Waist: nicht von der Policy gesteuert, auf Default-Pose fixiert.
LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

# Greifer-Grenzwerte (aus der URDF): offen = upper, geschlossen = lower.
GRIPPER_OPEN = 0.0245
GRIPPER_CLOSE = -0.02

# Policy-Action-Layout (nur die in der Sim genutzten Gruppen, in dieser Reihenfolge):
#   [0:7]   left_arm   (absolute Targets — Server konvertiert RELATIVE→ABS)
#   [7:14]  right_arm
#   [14]    left_hand  (1-DOF binär → beide linken Finger-Joints)
#   [15]    right_hand (1-DOF binär → beide rechten Finger-Joints)
POLICY_ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS  # 14 Arm-Joints in Action-Reihenfolge

# Arm-Startpose: dieselben G1-Arm-Winkel wie die DEX3-Dataset-Startpose (Frame 0).
# Gleiche Joint-Namen → gültige, zum Tisch greifende Ausgangsstellung.
ARM_INIT_STATE = {
    "left_shoulder_pitch_joint": -0.05662, "left_shoulder_roll_joint": 0.31445,
    "left_shoulder_yaw_joint": 0.12487, "left_elbow_joint": -0.28736,
    "left_wrist_roll_joint": 0.20897, "left_wrist_pitch_joint": 0.13095,
    "left_wrist_yaw_joint": -0.09767,
    "right_shoulder_pitch_joint": -0.40259, "right_shoulder_roll_joint": -0.34180,
    "right_shoulder_yaw_joint": -0.01584, "right_elbow_joint": 0.17989,
    "right_wrist_roll_joint": 0.00290, "right_wrist_pitch_joint": -0.09826,
    "right_wrist_yaw_joint": 0.29865,
}

# Vollständige Init-Pose: Arme wie oben, Greifer offen, Beine/Waist auf 0 (Stehpose).
INIT_JOINT_POS = dict(ARM_INIT_STATE)
for _j in LEFT_GRIPPER_JOINTS + RIGHT_GRIPPER_JOINTS:
    INIT_JOINT_POS[_j] = GRIPPER_OPEN

# ---------------------------------------------------------------------------
# Articulation-Konfiguration
# ---------------------------------------------------------------------------

G1_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Default greift nur, wenn die Env ohne --asset-path gestartet wird.
        # entrypoint_baseline.sh setzt ASSET_PATH explizit.
        usd_path="/data/assets/g1_gripper.usd",
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
            # Torso fixieren: Tabletop-Aufgabe ohne Balance-Controller. Beine/Waist
            # werden zusätzlich per PD auf Default gehalten (siehe lower_body-Actuator).
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos=INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Beine + Waist: steif auf Default-Pose halten (nicht policy-gesteuert).
        "lower_body": ImplicitActuatorCfg(
            joint_names_expr=LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS + WAIST_JOINTS,
            effort_limit=300.0,
            velocity_limit=5.0,
            stiffness=200.0,
            damping=20.0,
        ),
        # Arme: positionsgeregelt; GR00T liefert (über Server-Decode) absolute Targets.
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
        # Greifer: prismatische Finger, absolute Targets (binär offen/zu).
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=LEFT_GRIPPER_JOINTS,
            effort_limit=50.0,
            velocity_limit=0.2,
            stiffness=500.0,
            damping=20.0,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_GRIPPER_JOINTS,
            effort_limit=50.0,
            velocity_limit=0.2,
            stiffness=500.0,
            damping=20.0,
        ),
    },
)

# ---------------------------------------------------------------------------
# Kamera-Parameter
# ---------------------------------------------------------------------------

def look_at_world_quat(eye, target, world_up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) für eine Kamera in Isaac-Lab-``convention="world"``.

    Blickachse = +X, oben = +Z. Eigenständige Kopie aus g1_dex3_cfg.py (damit der
    DEX3-Pfad unberührt bleibt).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(world_up, dtype=float)

    x = target - eye
    x /= np.linalg.norm(x)
    z = up - np.dot(up, x) * x
    if np.linalg.norm(z) < 1e-6:
        z = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)

    R = np.column_stack([x, y, z])
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * w)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * w)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * w)
    return (float(w), float(qx), float(qy), float(qz))


@configclass
class G1GripperCameraCfg:
    """Kamera-Posen — ``UNITREE_G1`` erwartet GENAU eine Kamera ``ego_view``."""

    width: int = 640
    height: int = 480
    hfov_deg: float = 69.0

    # Egozentrische Kopfkamera (am Torso montiert), blickt nach vorne-unten auf den Tisch.
    ego_view_local: dict = None
    # Szenen-Übersichtskamera — NUR fürs Video, NICHT Teil der Policy-Observation.
    cam_scene: dict = None

    def __post_init__(self):
        # Ego-Kamera relativ zum Torso-Link (convention="world", Link-Frame).
        # Auf Kopfhöhe, blickt nach vorne-unten auf die Tischmitte.
        ego_eye = (0.05, 0.0, 0.45)
        ego_target = (0.45, 0.0, 0.05)
        self.ego_view_local = {"pos": ego_eye, "rot": look_at_world_quat(ego_eye, ego_target)}

        scene_eye = (1.8, 1.6, 1.7)
        scene_target = (0.30, 0.0, 0.80)
        self.cam_scene = {"pos": scene_eye, "rot": look_at_world_quat(scene_eye, scene_target)}


CAMERA_CFG = G1GripperCameraCfg()
