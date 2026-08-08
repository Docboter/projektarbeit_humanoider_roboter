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
        # Arme: positionsgeregelt (PD-Controller)
        # GR00T-Aktionen sind RELATIVE Deltas → werden im Control-Loop auf
        # aktuelle Position addiert und dann als Target gesetzt.
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
        # Hände: positionsgeregelt (ABSOLUTE Targets aus GR00T-Aktionen).
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

    # Horizontaler FOV in Grad. Overlay-Iteration: 69° zu eng (Sim zu nah), 90° zu weit
    # (Szene zu sparse, Hände winzig). 75° als Kompromiss.
    #
    # ACHTUNG, 2026-08-08: Dieser Wert war reine Dokumentation — er stand hier, wurde aber
    # nirgends gelesen. Die Kameras liefen mit der fest eingetragenen `focal_length=24.0`
    # aus `g1_dex3_blockstack_env.py`, also mit 47,2° statt 75°. Die Sim-Bilder waren damit
    # dauerhaft zu eng, und die damals als „zu eng" verworfenen 69° waren nie im Bild.
    # `focal_high`/`focal_wrist`/`focal_scene` unten rechnen die Werte jetzt tatsächlich um.
    hfov_deg: float = 75.0
    # Handgelenks- und Szenenkamera behalten vorerst ihr bisheriges Sichtfeld (focal 18.0 =
    # 60,4°) — für die ist keine Overlay-Iteration dokumentiert, ein Wechsel wäre geraten.
    hfov_wrist_deg: float = 60.4
    hfov_scene_deg: float = 60.4

    # Sensorbreite in mm. Isaac Lab leitet die vertikale Apertur aus dem Seitenverhältnis ab.
    horizontal_aperture_mm: float = 20.955

    # Kamera-Posen (pos in Meter, rot als (w, x, y, z) Quaternion, World-Frame)
    # WICHTIG: Aus den Dataset-Videos rekonstruieren! Dies sind Schätzwerte.
    cam_left_high: dict = None
    cam_right_high: dict = None
    # Wrist-Kameras werden relativ zum Wrist-Link definiert (lokal)
    cam_left_wrist_local: dict = None
    cam_right_wrist_local: dict = None
    # Szenen-Übersichtskamera — NUR fürs Video, NICHT Policy-Observation
    cam_scene: dict = None

    # Aus hfov_*_deg berechnet (siehe __post_init__) — die Env liest diese Werte.
    focal_high: float = None
    focal_wrist: float = None
    focal_scene: float = None

    def __post_init__(self):
        # Brennweiten aus dem gewünschten Sichtfeld, damit die hfov-Angaben oben wirken.
        def focal(hfov_deg: float) -> float:
            return self.horizontal_aperture_mm / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))

        self.focal_high = float(focal(self.hfov_deg))       # 75.0° -> 13.65 mm (vorher 24.0)
        self.focal_wrist = float(focal(self.hfov_wrist_deg))  # 60.4° -> 18.0 mm (unverändert)
        self.focal_scene = float(focal(self.hfov_scene_deg))  # 60.4° -> 18.0 mm (unverändert)
        # Externe Kameras: Kopf-Stereo-Paar, blickt von vorne-oben NACH UNTEN auf den Tisch.
        # Rekonstruiert aus den Dataset-Frames (Simulation/camera_reference/dataset_cam_*_high.png):
        # beide Hände kommen von unten ins Bild, Tisch füllt die unteren ~2/3.
        # Rotation per Look-at auf die Tischmitte — frühere (0.924,-0.383,0,0) war eine reine
        # Roll-Drehung um die +X-Blickachse (Bild verkippt, kein Pitch) und zeigte auf den Boden.
        #
        # Overlay-Iteration 6: Reale Referenz (dataset_cam_*_high.png) zeigt eindeutig einen
        # FLACHEN Vorwärts-Blick vom Kopf: beide Hände kommen von unten-links/rechts ins Bild,
        # Finger zeigen nach oben-vorne, Tisch füllt die Mitte. KEINE steile Top-Down-Sicht.
        # Iter 5 (target z=0.80, steil nach unten) ließ die Arme aus dem Bild fallen.
        # Fix: eye zentriert auf Kopfhöhe (z=1.45, x=0), Target weit nach VORNE (x=0.55) auf
        # Tischhöhe → flacher Pitch (~46°), Arme reichen von unten ins Bild, Tisch in der Mitte.
        #
        # Iteration 13 (2026-08-08), nachdem die Kameras überhaupt wieder rendern. Der
        # Overlay gegen Simulation/camera_reference/ zeigte drei Abweichungen, alle drei
        # gegen eine unabhängige Quelle geprüft statt geschätzt:
        #
        # a) MONTAGEPUNKT. Die Kameras standen auf z=1.45 und y=±0.08 — also NEBEN und ÜBER
        #    dem Kopf, weshalb der eigene Kopf ein Drittel des Bildes verdeckte (Lauf 16).
        #    Der echte G1 trägt seine Kopfkamera laut URDF im `d435_link`, pelvis-relativ
        #    (0.0537, 0.0175, 0.4739); das Pelvis sitzt env-lokal auf z=0.85, macht
        #    (0.0537, 0.0175, 1.3239).
        # b) STEREO-BASIS. ±0.08 wären 16 cm Basis. Aus den beiden Referenzbildern gemessen:
        #    Querversatz 40 px (SAD-Minimum über die Tischplatte), Würfelkantenlänge 40–46 px
        #    bei bekannten 5 cm → Motivabstand ~0.57 m → Basis ~4.7 cm. Der Wert hängt nicht
        #    am angenommenen FOV, weil Abstand und Winkel gemeinsam mitskalieren. Passt zu
        #    den 50 mm einer RealSense D435, also ±0.025.
        # c) PARALLEL STATT KONVERGENT. Beide Kameras auf EIN gemeinsames Ziel zu richten
        #    erzeugte ±8,3° Gierwinkel und damit die schräge Tischkante im Bild. Ein reales
        #    Stereopaar blickt parallel — deshalb bekommt jede Kamera ihr Ziel auf der
        #    eigenen y-Linie.
        #
        # Zielpunkt = Mitte des Würfel-Spawnbereichs (x≈0.34, z≈0.915). Das ergibt 55°
        # Neigung; damit liegt die Tischhinterkante bei ~5 % und die Vorderkante bei ~99 %
        # der Bildhöhe, der Tisch also vollständig im Bild wie in der Referenz, und die
        # Würfel stehen mittig. Beide Hände sind dabei 21,5° von der Blickachse entfernt
        # und damit deutlich innerhalb des 75°×59,9°-Sichtfelds.
        #
        # Iteration 14 (Lauf 17): Die Basis wird um y=0 zentriert statt um die y=0.0175 des
        # `d435_link`. Grund: in der Reset-Pose stehen die Handgelenke fast symmetrisch
        # (y=+0.158 / −0.144), und im Referenzbild liegen beide Hände symmetrisch um die
        # Bildmitte — die reale Kamera sitzt also auf der Mittellinie. Das `d435_link` ist
        # der Montageflansch eines Moduls, nicht der Mittelpunkt zwischen zwei Bildsensoren.
        # x und z bleiben beim URDF-Wert, die sind eindeutig.
        high_target_x, high_target_z = 0.34, 0.915
        left_high_eye  = (0.0537,  0.025, 1.3239)   # halbe Stereobasis links der Mittellinie
        right_high_eye = (0.0537, -0.025, 1.3239)   # halbe Stereobasis rechts der Mittellinie
        self.cam_left_high = {
            "pos": left_high_eye,
            "rot": look_at_world_quat(
                left_high_eye, (high_target_x, left_high_eye[1], high_target_z)),
        }
        self.cam_right_high = {
            "pos": right_high_eye,
            "rot": look_at_world_quat(
                right_high_eye, (high_target_x, right_high_eye[1], high_target_z)),
        }
        # Wrist-Kameras: am jeweiligen Wrist-Yaw-Link montiert (convention="world", Link-Frame).
        # Die Hand ragt entlang +X (Palm-Joint bei x=0.0415, Finger weiter bei +X). Look-at von
        # hinter/über dem Wrist-Origin (raus aus dem Palm-Mesh) auf die Fingerspitzen → die Kamera
        # zeigt garantiert auf die Hand (vorher: pos=(0.05,…) steckte IM Palm-Mesh → nur Grau;
        # rechte Cam zudem falsch herum (-X)). Roll/Feinframing nach Render-Vergleich justieren.
        # Overlay-Iteration 6: KORREKTUR des früheren Z-Flips. Die reale Referenz
        # (dataset_cam_left_wrist.png) zeigt die Kamera von OBEN-HINTEN nach UNTEN-VORNE über
        # die Finger auf den Tisch blickend — das dunkle Gehäuse oben im realen Bild ist der
        # Unterarm (= weißer Connector in der Sim), der nur das obere Drittel einnimmt.
        # Der vorherige Flip nach -Z (Kamera unter dem Wrist) war falsch und ließ den Connector
        # das ganze Bild füllen. Fix: eye wieder ÜBER den Wrist (+Z), leicht hinter den Knöcheln
        # (-X), Blick nach vorne-unten (+X, -Z) auf Fingerspitzen + Tisch.
        # Iter 7: Target leicht angehoben (-0.12 → -0.06), damit der Blick die Tischfläche
        # statt des Bodengitters dahinter trifft (Iter 6 pitchte minimal über die Tischkante).
        # Iter 10: Kamera ein Stück entlang der Handachse (+X) Richtung Finger geschoben
        # (eye -0.08 → 0.0, target 0.22 → 0.30). Iter 12: war etwas zu weit vorne, zurück auf
        # die Mitte zwischen Iter 9 und 10 (eye -0.04, target 0.26). Blickrichtung/Roll bleiben.
        wrist_eye = (-0.04, 0.0, 0.10)
        left_wrist_target  = (0.26, 0.0, -0.06)
        right_wrist_target = (0.26, 0.0, -0.06)
        # Iter 8 — ROLL-Korrektur: Pitch/Blickrichtung stimmten, aber die Hand stand im Sim
        # VERTIKAL, im Real liegt sie HORIZONTAL (Finger nach rechts statt nach unten) → ~90°
        # Roll-Versatz. Ursache: look_at nutzte default world_up=(0,0,1), was im rotierten
        # Wrist-Link-Frame den falschen Roll erzeugt. Fix: world_up auf die laterale Link-Achse
        # (+Y) legen, damit die Hand horizontal im Bild liegt.
        # Iter 9: linke Cam mit +Y war korrekt (Finger horizontal nach rechts, deckt sich mit
        # Real). Rechte Cam mit -Y war um 180° verdreht (Finger nach links statt oben-rechts) →
        # der rechte Wrist-Link nutzt DIESELBE Frame-Konvention wie links, kein gespiegeltes
        # Frame. Daher beide Cams +Y. Restlicher ~45°-Diagonal-Versatz rechts = reale Pose-
        # Differenz (asymmetrische Init-Pose), ggf. später per Z-Komponente im Up feinjustieren.
        # Iter 12: rechte Cam wieder auf (0,1,0) wie im letzten gerenderten Run — der in Iter 11
        # versuchte -Z-Tilt wurde nie validiert und auf Wunsch zurückgenommen. Beide Cams +Y.
        left_wrist_up  = (0.0, 1.0, 0.0)
        right_wrist_up = (0.0, 1.0, 0.0)
        self.cam_left_wrist_local = {
            "pos": wrist_eye,
            "rot": look_at_world_quat(wrist_eye, left_wrist_target, world_up=left_wrist_up),
        }
        self.cam_right_wrist_local = {
            "pos": wrist_eye,
            "rot": look_at_world_quat(wrist_eye, right_wrist_target, world_up=right_wrist_up),
        }

        # Szenen-Übersichtskamera (NUR fürs aufgenommene Video): zeigt die GANZE Szene —
        # Roboter (Pelvis z=0.85, Kopf ~1.4) + Tisch (x=0.5) — von schräg vorne-seitlich-oben.
        # Weltfest. eye in +X (vor dem Tisch), +Y (seitlich), +Z (oben); Blick zurück auf die Mitte.
        scene_eye = (1.8, 1.6, 1.7)
        scene_target = (0.30, 0.0, 0.80)
        self.cam_scene = {"pos": scene_eye, "rot": look_at_world_quat(scene_eye, scene_target)}


CAMERA_CFG = G1Dex3CameraCfg()
