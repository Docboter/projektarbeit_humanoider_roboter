"""
Block-Stacking-Umgebung für G1 + Dex3 in Isaac Lab.

Implementiert als DirectRLEnv (Isaac Lab 2.x):
    - Tisch + 3 Würfel mit randomisierten Startpositionen
    - 4 RGB-Kameras (cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist)
    - 28-dim Joint-State (left_arm + right_arm + left_dex3 + right_dex3)
    - Erfolgsmetrik: Würfel vertikal gestapelt
    - Episodenlänge: 5400 Steps @ 30 Hz = 180 Sekunden (siehe episode_length_s)

Abnahmekriterien (siehe ISAAC_LAB_SIM_PLAN.md):
    Phase A: leere Szene rendert ein RGB-Bild (EGL headless ok)
    Phase B: Roboter lädt, Arme per Skript fahrbar
    Phase C: obs-Dict hat korrektes Format/Shape
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Any

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from g1_dex3_cfg import (
    ALL_JOINTS_ORDERED,
    CAMERA_CFG,
    G1_DEX3_CFG,
    LEFT_ARM_JOINTS,
    LEFT_DEX3_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_DEX3_JOINTS,
)


# ---------------------------------------------------------------------------
# Render-Konfiguration — DLSS-Upscaling abschalten
# ---------------------------------------------------------------------------
def _make_sim_cfg(**kwargs) -> SimulationCfg:
    """SimulationCfg mit DLSS-freiem Rendering, soweit die Isaac-Lab-Version das kennt.

    ANLASS (2026-08-08, Isaac Sim 6.0 / Isaac Lab 3.0.0-beta2): Alle fünf Kameras
    lieferten praktisch leere Bilder — cam_left_high spannte über das ganze Bild nur die
    Helligkeitsstufen 244–249. Die Ursache liegt NICHT bei den Posen: der Dump
    (dump_camera_poses.py) belegt für jede Kamera Position und Blickrichtung
    deckungsgleich mit der Config, `quat_w_world/+X` bei 0.0°. Im Kit-Log steht dagegen:

        [Warning] [omni.rtx] DLSS increasing input dimensions:
            Render resolution of (320, 240) is below minimal input resolution of 300.

    DLSS rendert also intern auf halber Auflösung (640×480 → 320×240) und liegt damit
    unter seinem eigenen Minimum. DLAA bzw. „Off" rendern in Native-Auflösung, wodurch
    das Minimum entfällt. Die Kamera-Auflösung selbst bleibt bei 640×480, weil sie an den
    Datensatz gebunden ist — sie hochzudrehen wäre der falsche Hebel.

    Vorgeschichte: DLSS ist in diesem Projekt bereits als Renderproblem dokumentiert
    (docs/fehlerbehebung.md, docs/simulation/archiv/gpu-kompatibilitaet.md) — dort als
    `createDLSSContext error` auf GPUs ohne DLSS-RR-Unterstützung.

    Defensiv gebaut, weil Isaac Lab 3.0 Beta ist und die Feldnamen wandern können: es
    werden nur Felder gesetzt, die RenderCfg tatsächlich hat, und jeder Fehlschlag wird
    LAUT gemeldet statt still ignoriert (falsche Kit-Settings schluckt Kit sonst
    kommentarlos — genau die Falle, die hier Stunden gekostet hat).
    """
    import dataclasses

    render = None
    try:
        from isaaclab.sim import RenderCfg
    except ImportError:
        print("[Env] WARN: isaaclab.sim.RenderCfg nicht vorhanden — DLSS bleibt aktiv.",
              flush=True)
    else:
        available = {f.name for f in dataclasses.fields(RenderCfg)}
        # DLAA = DLSS-Kantenglättung OHNE Upscaling; "Off" als Rückfall.
        wanted = {"antialiasing_mode": "DLAA", "enable_dlssg": False, "dlss_mode": 2}
        use = {k: v for k, v in wanted.items() if k in available}
        missing = sorted(set(wanted) - set(use))
        try:
            render = RenderCfg(**use)
        except (TypeError, ValueError) as e:
            print(f"[Env] WARN: RenderCfg({use}) abgelehnt ({e}) — DLSS bleibt aktiv.",
                  flush=True)
            render = None
        else:
            print(f"[Env] RenderCfg gesetzt: {use}"
                  + (f"  (nicht unterstützt: {missing})" if missing else ""), flush=True)

    if render is None:
        return SimulationCfg(**kwargs)
    try:
        return SimulationCfg(render=render, **kwargs)
    except TypeError as e:
        print(f"[Env] WARN: SimulationCfg kennt kein 'render' ({e}) — DLSS bleibt aktiv.",
              flush=True)
        return SimulationCfg(**kwargs)


# ---------------------------------------------------------------------------
# Szenenkonfiguration
# ---------------------------------------------------------------------------

@configclass
class G1Dex3BlockstackSceneCfg(InteractiveSceneCfg):
    """Szene: Roboter am Tisch mit 3 Würfeln."""

    # Tisch (einfache Box als Placeholder — für echten Tisch USD ersetzen)
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            # Tisch 0.87 m hoch (Oberfläche z=0.87). Cubes sitzen auf der Oberfläche (Zentrum z=0.915,
            # Oberkante z=0.94 — durch leichte Erhöhung der block_z_surface erreicht, nicht des Tisches).
            # Tisch war testweise auf 0.89 angehoben, das blockierte aber die Roboterarme im
            # Closed-Loop: Modell versucht zu z≈0.915 zu greifen, Tisch (0.89) steckte die Hände fest.
            size=(0.8, 0.6, 0.87),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # Weiß wie im Dataset (war beige Platzhalter-Farbe).
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.435)),
    )

    # 3 Würfel (5 cm Kantenlänge, unterschiedliche Farben)
    block_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/block_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0, dynamic_friction=2.5, restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, -0.15, 0.915)),
    )

    block_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/block_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0, dynamic_friction=2.5, restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36, 0.0, 0.915)),
    )

    block_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/block_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0, dynamic_friction=2.5, restitution=0.0,
            ),
            # Gelb wie im Dataset (war blau (0.1,0.1,0.8) — die Realdaten nutzen rot/grün/gelb,
            # nicht blau). Angleichung an die Trainingsverteilung für den eingefrorenen Vision-Encoder.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.70, 0.10)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, 0.15, 0.915)),
    )

    # Schwarzes Stapel-Band (Landmarke wie im Dataset — dort wird auf einen kleinen
    # schwarzen Streifen gestapelt; in der Sim fehlte er). Dünnes, statisches Cuboid
    # flach auf der Tischoberkante (z=0.87). Position/Größe sind eine NÄHERUNG und
    # sollten an die echte Dataset-Lage angepasst werden.
    # Domain-Gap-Angleichung für den eingefrorenen Vision-Encoder (Platzier-/Stapel-Ziel).
    stack_band: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stack_band",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.04, 0.006),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.873)),
    )

    # Roboter
    robot: ArticulationCfg = G1_DEX3_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # Kameras (TiledCamera für effizientes batch-Rendering)
    cam_left_high: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_left_high",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_CFG.cam_left_high["pos"],
            rot=CAMERA_CFG.cam_left_high["rot"],
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=CAMERA_CFG.width,
        height=CAMERA_CFG.height,
    )

    cam_right_high: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_right_high",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_CFG.cam_right_high["pos"],
            rot=CAMERA_CFG.cam_right_high["rot"],
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=CAMERA_CFG.width,
        height=CAMERA_CFG.height,
    )

    cam_left_wrist: TiledCameraCfg = TiledCameraCfg(
        # Wrist-Kamera: Pfad relativ zum Wrist-Link des Roboters
        prim_path="{ENV_REGEX_NS}/robot/left_wrist_yaw_link/cam_left_wrist",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_CFG.cam_left_wrist_local["pos"],
            rot=CAMERA_CFG.cam_left_wrist_local["rot"],
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0),
        ),
        width=CAMERA_CFG.width,
        height=CAMERA_CFG.height,
    )

    cam_right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/right_wrist_yaw_link/cam_right_wrist",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_CFG.cam_right_wrist_local["pos"],
            rot=CAMERA_CFG.cam_right_wrist_local["rot"],
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0),
        ),
        width=CAMERA_CFG.width,
        height=CAMERA_CFG.height,
    )

    # Szenen-Übersichtskamera — NUR fürs Video (nicht Teil der Policy-Observation).
    # Weltfest, schräg vorne-seitlich-oben; weiterer FOV (focal 18 ≈ 60° HFOV), um Roboter
    # + Tisch komplett zu erfassen. Größere clipping_range, da weiter entfernt.
    cam_scene: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_scene",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_CFG.cam_scene["pos"],
            rot=CAMERA_CFG.cam_scene["rot"],
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 30.0),
        ),
        width=CAMERA_CFG.width,
        height=CAMERA_CFG.height,
    )

    num_envs = 1
    env_spacing = 2.0


# ---------------------------------------------------------------------------
# Env-Konfiguration
# ---------------------------------------------------------------------------

@configclass
class G1Dex3BlockstackEnvCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = _make_sim_cfg(
        dt=1.0 / 200.0,   # 200 Hz Physik
        render_interval=7, # Alle 7 Physics-Steps → ~28.6 Hz Policy-Obs
    )

    # Szene
    scene: G1Dex3BlockstackSceneCfg = G1Dex3BlockstackSceneCfg(num_envs=1, env_spacing=2.0)

    # Pflichtfelder von DirectRLEnvCfg
    decimation: int = 7          # Physics-Sub-Steps pro Policy-Step (= render_interval)
    action_space: int = 28       # Arm-Deltas + Hand-Absolute
    observation_space: int = 28  # nur joints; Bilder gehen separat ans Modell

    # Task-Parameter
    episode_length_s: float = 300.0   # 9000 Steps @ 30 Hz (5 min)
    policy_hz: float = 30.0
    execution_horizon: int = 8        # wie viele Chunk-Steps ausführen, dann re-plan

    # Block-Sampling-Bereich = erreichbarer Greifraum (aus Replay: Hände greifen bei x≈0.35,
    # y≈±0.19, z≈0.92). x/y eng um den Greifraum, z = neue Tischoberfläche (0.87) + halbe Würfelhöhe.
    block_x_range: tuple[float, float] = (0.30, 0.40)
    block_y_range: tuple[float, float] = (-0.20, 0.20)
    block_z_surface: float = 0.915    # Tischoberfläche 0.89 + halbe Würfel-Höhe (0.025)

    # Erfolgsparameter
    stack_xy_tol: float = 0.03  # max. horizontaler Versatz zwischen Würfel-Mittelpunkten
    stack_height_min: float = 0.08  # Mindesthöhe des Turms (2 × 0.05 m - Toleranz)
    stack_vel_max: float = 0.05  # max. Geschwindigkeit für "stabil gestapelt"

    # Task-Beschreibung (geht ans Language-Modell)
    task_description: str = "stack the blocks"

    # Domain Randomization: Beleuchtung + Materialfarben pro Episode randomisieren.
    # Ziel: Real→Sim-Gap schließen (gemessen: cam_left_wrist 0.427, Mittel 0.260).
    # Deaktivieren mit DR_ENABLED=0 Env-Var im Eval-Runner oder dr_enabled=False.
    dr_enabled: bool = True

    # ── Reward-Modus ──────────────────────────────────────────────────────────
    # "binary" (Default): spärlicher 0/1-Success-Reward — für Closed-Loop-Eval/Ablation.
    # "shaped":          dichter, vektorisierter Reward fürs RL-Training (πRL/FPO).
    # Der Eval-Pfad nutzt weiter "binary"; "shaped" wird nur vom RL-Trainer gesetzt.
    reward_mode: str = "binary"

    # Gewichte des Shaped-Reward (nur bei reward_mode="shaped"). Siehe
    # docs/weiterfuehrend/reinforcement-learning-plan.md §3.2.
    rew_reach: float = 1.0    # Annäherung Hand → nächster Würfel (kontinuierlich)
    rew_stack: float = 2.0    # Würfel horizontal zusammenführen + Turmhöhe aufbauen
    rew_success: float = 10.0  # Bonus für stabilen Stapel (nutzt _check_success)
    rew_smooth: float = 0.01  # Strafe auf Gelenkgeschwindigkeit (glättet Finger-Aktionen)


# ---------------------------------------------------------------------------
# Umgebung
# ---------------------------------------------------------------------------

class G1Dex3BlockstackEnv(DirectRLEnv):
    """
    Closed-Loop-Simulations-Umgebung für GR00T-N1.6-Evaluation.

    Diese Klasse ist eine Gym-kompatible Env, die die Isaac-Lab-DirectRLEnv-
    Schnittstelle implementiert. Der GR00T-Policy-Server wird NICHT hier
    gestartet — das erledigt run_g1_dex3_sim_eval.py.
    """

    cfg: G1Dex3BlockstackEnvCfg

    def __init__(self, cfg: G1Dex3BlockstackEnvCfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode=render_mode)

        # Joint-Index-Mapping aufbauen (nach erstem physics-Step)
        self._joint_ids: list[int] | None = None
        self._arm_joint_ids: list[int] | None = None
        self._hand_joint_ids: list[int] | None = None
        self._hand_body_ids: list[int] | None = None  # Handwurzel-Links (für Shaped-Reward)

        # Episode-Tracking
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Dex3-Finger-Gelenkgrenzen an die echte Dataset-Range weiten (URDF-Limits sind zu eng).
        self._widen_finger_joint_limits()

    # ------------------------------------------------------------------
    # Szene aufbauen
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # Boden und Licht manuell spawnen (GroundPlaneCfg/DomeLightCfg sind kein
        # gültiger InteractiveSceneCfg-Asset-Typ und müssen direkt aufgerufen werden)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.robot: Articulation = self.scene["robot"]
        self.table: RigidObject = self.scene["table"]
        self.blocks: list[RigidObject] = [
            self.scene["block_0"],
            self.scene["block_1"],
            self.scene["block_2"],
        ]
        self.cameras: dict[str, TiledCamera] = {
            "cam_left_high": self.scene["cam_left_high"],
            "cam_right_high": self.scene["cam_right_high"],
            "cam_left_wrist": self.scene["cam_left_wrist"],
            "cam_right_wrist": self.scene["cam_right_wrist"],
            "cam_scene": self.scene["cam_scene"],  # nur fürs Video
        }

        self.scene.filter_collisions(global_prim_paths=[])

    # ------------------------------------------------------------------
    # Finger-Gelenkgrenzen weiten
    # ------------------------------------------------------------------

    def _widen_finger_joint_limits(self):
        """Dex3-Finger-Positionsgrenzen zur Laufzeit an die echte Dataset-Range anheben.

        Die URDF/USD-Limits sind ~0,2–0,33 rad enger als der tatsächliche Bewegungsumfang
        (gemessen über 281.196 Frames von unitreerobotics/G1_Dex3_BlockStacking_Dataset).
        Ohne Weitung würde der Sim volle Greif-Kommandos der Policy an der Grenze abklemmen →
        Finger schließen nicht ganz. Werte = Union(USD-Limit, Dataset-Min/Max) + ~0,05 rad Marge.
        Kein Vorzeichen-Flip (Richtung stimmt) — nur Reichweite.
        """
        finger_limits = {
            # left hand (_1 joints: Vorzeichen korrekt, nur Reichweite erweitern)
            "left_hand_thumb_1_joint": (-0.66, 1.15),
            "left_hand_middle_1_joint": (-2.13, 0.05),
            "left_hand_index_1_joint": (-2.13, 0.05),
            # right hand (_1 joints analog)
            "right_hand_thumb_1_joint": (-1.11, 0.66),
            "right_hand_index_1_joint": (-0.05, 2.14),
            "right_hand_middle_1_joint": (-0.05, 2.14),
            # _0 joints (middle_0, index_0): Vorzeichen-Fix via _SIGN_FLIP_IDX →
            # nach Negation fallen Werte in die Original-USD-Limits, kein Weiten nötig.
        }
        names = list(finger_limits.keys())
        joint_ids, _ = self.robot.find_joints(names, preserve_order=True)
        limits = torch.tensor(
            [finger_limits[n] for n in names], device=self.device, dtype=torch.float32
        )
        limits = limits.unsqueeze(0).expand(self.num_envs, -1, -1)  # (num_envs, n_joints, 2)
        self.robot.write_joint_position_limit_to_sim(
            limits, joint_ids=joint_ids, warn_limit_violation=False
        )
        print(f"[Env] Dex3-Finger-Gelenkgrenzen an Dataset-Range geweitet "
              f"({len(joint_ids)} Gelenke).", flush=True)

    # ------------------------------------------------------------------
    # Visual Domain Randomization
    # ------------------------------------------------------------------

    def _setup_visual_dr(self) -> None:
        """Lazy init: traverses the USD stage once and caches PreviewSurface shader handles.

        Called on the first episode reset so that Isaac Sim's stage is fully populated.
        Finds shaders by traversing from each object's prim root — robust against Isaac Lab's
        internal USD path naming conventions.
        """
        import omni.usd
        from pxr import Usd, UsdShade

        stage = omni.usd.get_context().get_stage()
        self._dr_shaders: dict[str, "UsdShade.Shader | None"] = {}
        self._dr_rng = np.random.default_rng()

        targets = {
            "table":      "/World/envs/env_0/table",
            "block_0":    "/World/envs/env_0/block_0",
            "block_1":    "/World/envs/env_0/block_1",
            "block_2":    "/World/envs/env_0/block_2",
            "stack_band": "/World/envs/env_0/stack_band",
        }
        for name, root_path in targets.items():
            root = stage.GetPrimAtPath(root_path)
            shader = None
            if root.IsValid():
                for prim in Usd.PrimRange(root):
                    candidate = UsdShade.Shader(prim)
                    if candidate and candidate.GetIdAttr().Get() in (
                        "UsdPreviewSurface", "PreviewSurface"
                    ):
                        shader = candidate
                        break
            self._dr_shaders[name] = shader
            print(f"[DR] '{name}' shader: {'OK  ' + shader.GetPath().pathString if shader else 'NOT FOUND'}", flush=True)

        self._dr_light = stage.GetPrimAtPath("/World/Light")
        print(f"[DR] dome light: {'OK' if self._dr_light.IsValid() else 'NOT FOUND'}", flush=True)

    def _set_shader_color(self, shader, rgb: tuple | list) -> None:
        from pxr import Gf
        inp = shader.GetInput("diffuseColor")
        if inp:
            inp.Set(Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2])))

    def _randomize_visuals(self) -> None:
        """Per-episode visual DR: dome light intensity/color + object diffuse colors.

        Targets the cam_left_wrist domain gap (measured cosine distance 0.427) by
        varying the lighting and surface colors the wrist cameras see up close.
        """
        if not self.cfg.dr_enabled:
            return
        if not hasattr(self, "_dr_shaders"):
            return

        from pxr import Gf
        rng = self._dr_rng

        # 1. Dome light: intensity ±50 % + warm/cool white-balance shift
        if self._dr_light.IsValid():
            attr_i = self._dr_light.GetAttribute("inputs:intensity")
            attr_c = self._dr_light.GetAttribute("inputs:color")
            if attr_i:
                attr_i.Set(float(rng.uniform(1000.0, 3800.0)))
            if attr_c:
                r = float(rng.uniform(0.70, 1.00))
                g = float(rng.uniform(0.70, 0.95))
                b = float(rng.uniform(0.58, 0.95))
                attr_c.Set(Gf.Vec3f(r, g, b))

        # 2. Table surface: gray scale 0.68–0.96 with slight color tint
        shader = self._dr_shaders.get("table")
        if shader:
            base = float(rng.uniform(0.68, 0.96))
            tint = rng.uniform(-0.06, 0.06, size=3)
            color = np.clip([base + tint[0], base + tint[1], base + tint[2]], 0.0, 1.0)
            self._set_shader_color(shader, color)

        # 3. Blocks: keep hue recognizable, vary lightness ±10 %
        block_bases = {
            "block_0": (0.80, 0.10, 0.10),  # rot
            "block_1": (0.10, 0.60, 0.10),  # grün
            "block_2": (0.85, 0.70, 0.10),  # gelb
        }
        for name, base in block_bases.items():
            shader = self._dr_shaders.get(name)
            if shader:
                noise = rng.uniform(-0.10, 0.10, size=3)
                color = np.clip(np.array(base) + noise, 0.04, 1.0)
                self._set_shader_color(shader, color)

        # 4. Stack band: very dark, slight brightness variation
        shader = self._dr_shaders.get("stack_band")
        if shader:
            v = float(rng.uniform(0.01, 0.08))
            self._set_shader_color(shader, (v, v, v))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._episode_step[env_ids] = 0
        self._episode_success[env_ids] = False

        # Visual DR: lazy init beim ersten Reset, dann pro Episode
        if not hasattr(self, "_dr_shaders"):
            self._setup_visual_dr()
        self._randomize_visuals()

        # Roboter in Home-Pose zurücksetzen
        default_pos = self.robot.data.default_joint_pos[env_ids]
        default_vel = torch.zeros_like(default_pos)
        self.robot.set_joint_position_target(default_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)

        # Würfel neu sampeln (randomisierte Positionen auf dem Tisch)
        for block in self.blocks:
            block_pos = torch.zeros(len(env_ids), 3, device=self.device)
            block_pos[:, 0] = sample_uniform(
                self.cfg.block_x_range[0], self.cfg.block_x_range[1],
                (len(env_ids),), device=self.device
            )
            block_pos[:, 1] = sample_uniform(
                self.cfg.block_y_range[0], self.cfg.block_y_range[1],
                (len(env_ids),), device=self.device
            )
            block_pos[:, 2] = self.cfg.block_z_surface
            block_quat = torch.zeros(len(env_ids), 4, device=self.device)
            block_quat[:, 0] = 1.0  # Identity-Quaternion (w=1)
            block.write_root_pose_to_sim(
                torch.cat([block_pos, block_quat], dim=-1), env_ids=env_ids
            )
            block.write_root_velocity_to_sim(
                torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
            )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, Any]:
        """
        Gibt das Observation-Dict zurück:
            - "joint_pos": (num_envs, 28) float32 — alle Joints in Policy-Reihenfolge
            - "video.*": (num_envs, H, W, 3) uint8 — Kamerabilder
        """
        obs = {}

        # Joint-Positionen (28-dim) in der korrekten Reihenfolge
        if self._joint_ids is None:
            self._joint_ids = self._build_joint_id_mapping()
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids].clone()
        # USD→Dataset-Konvention: proximale Fingergelenke mit invertierter Achse negieren,
        # damit das Modell Beobachtungen in derselben Konvention sieht wie die Trainingsdaten.
        joint_pos[:, self._SIGN_FLIP_IDX] *= -1
        obs["joint_pos"] = joint_pos

        # Kamerabilder
        for cam_name, camera in self.cameras.items():
            rgb = camera.data.output["rgb"]  # (num_envs, H, W, 4) RGBA
            obs[f"video.{cam_name}"] = rgb[..., :3]  # RGB ohne Alpha-Kanal

        return obs

    def _build_joint_id_mapping(self) -> list[int]:
        """Ordnet Joint-Namen auf Isaac-Lab-interne Indices ab."""
        joint_name_to_idx = {
            name: idx for idx, name in enumerate(self.robot.data.joint_names)
        }
        ids = []
        for name in ALL_JOINTS_ORDERED:
            if name not in joint_name_to_idx:
                raise ValueError(
                    f"Joint '{name}' nicht im USD-Asset gefunden. "
                    f"Verfügbare Joints: {list(joint_name_to_idx.keys())}"
                )
            ids.append(joint_name_to_idx[name])
        return ids

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    # Policy-Indices der proximalen Fingergelenke mit invertierter Achsenkonvention.
    # Dataset+ = schließen; USD: links negativ = schließen, rechts positiv = schließen.
    # → Vorzeichen vor Übergabe ans Sim flippen (17=l_mid0, 19=l_idx0, 24=r_idx0, 26=r_mid0).
    _SIGN_FLIP_IDX: list[int] = [17, 19, 24, 26]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Setzt das absolute Positions-Target EINMALIG vor den Physics-Sub-Steps.

        WICHTIG: Die GR00T-Policy liefert über den Server bereits ABSOLUTE
        Gelenk-Targets für alle 28 Dimensionen. Der Server konvertiert relative
        Aktionen intern via processor.decode_action() zurück in absolute Werte
        (use_relative_action=true im Checkpoint, Referenz = beobachteter State).
        Deshalb dürfen die Aktionen hier NICHT noch einmal auf current_pos
        addiert werden — das würde die Verschiebung verdoppeln und die Arme
        wegdriften lassen.

        _apply_action() wird danach decimation-mal aufgerufen und hält dasselbe
        Target.
        """
        if self._joint_ids is None:
            self._joint_ids = self._build_joint_id_mapping()

        # Vorzeichen-Fix: proximale Finger-Joints (middle_0, index_0) haben invertierte
        # USD-Achse. Negation mappt Dataset-Konvention → USD-Konvention.
        actions = actions.clone()
        actions[:, self._SIGN_FLIP_IDX] *= -1

        # actions sind bereits absolute Gelenkpositionen (alle 28 Dims) in
        # Policy-Reihenfolge → in Isaac-interne Joint-Reihenfolge umschreiben
        full_target = self.robot.data.joint_pos.clone()
        for policy_idx, isaac_idx in enumerate(self._joint_ids):
            full_target[:, isaac_idx] = actions[:, policy_idx]
        self._full_target = full_target

    def _apply_action(self) -> None:
        """Wendet das in _pre_physics_step berechnete Target an (render_interval-mal)."""
        if not hasattr(self, "_full_target"):
            return
        self.robot.set_joint_position_target(self._full_target)

    # ------------------------------------------------------------------
    # Rewards & Termination
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        """Reward je nach cfg.reward_mode.

        "binary" (Default): spärlicher 0/1-Success-Reward — für Eval/Ablation.
        "shaped":          dichter RL-Reward (siehe _shaped_reward).
        """
        if self.cfg.reward_mode == "shaped":
            return self._shaped_reward()
        success = self._check_success()
        self._episode_success |= success
        return success.float()

    def _get_hand_positions(self) -> torch.Tensor:
        """(num_envs, 2, 3) Weltpositionen der beiden Handwurzel-Links (links, rechts)."""
        if self._hand_body_ids is None:
            names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
            ids, _ = self.robot.find_bodies(names, preserve_order=True)
            self._hand_body_ids = ids
        return self.robot.data.body_pos_w[:, self._hand_body_ids, :]

    def _shaped_reward(self) -> torch.Tensor:
        """Dichter, voll vektorisierter Reward fürs RL-Fine-tuning (alle num_envs).

        Komponenten (Gewichte in cfg, siehe reinforcement-learning-plan.md §3.2):
          - reach:   Hand nahe am nächsten Würfel  (exp-geformt, in (0,1])
          - stack:   Würfel horizontal zusammen + Turmhöhe aufbauen
          - success: Bonus für stabilen Stapel (binäres _check_success)
          - smooth:  Strafe auf Gelenkgeschwindigkeit (dämpft verrauschte Finger)

        Nutzt ausschließlich vorhandene Tensoren (Block-/Body-Posen, joint_vel) —
        keine zusätzlichen Sensoren nötig.
        """
        # Würfel-Weltpositionen (num_envs, 3, 3)
        block_pos = torch.stack([b.data.root_pos_w for b in self.blocks], dim=1)

        # 1. Reach: minimaler Abstand irgendeiner Hand zu irgendeinem Würfel
        hand_pos = self._get_hand_positions()              # (num_envs, 2, 3)
        dists = torch.cdist(hand_pos, block_pos)           # (num_envs, 2, 3)
        min_reach = dists.amin(dim=(1, 2))                 # (num_envs,)
        r_reach = torch.exp(-4.0 * min_reach)

        # 2. Stack: paarweise xy-Nähe der Würfel + aufgebaute Höhe
        xy = block_pos[..., :2]                            # (num_envs, 3, 2)
        pdist = torch.cdist(xy, xy)                        # (num_envs, 3, 3)
        iu = torch.triu_indices(3, 3, offset=1, device=self.device)
        mean_xy = pdist[:, iu[0], iu[1]].mean(dim=1)       # (num_envs,)
        r_together = torch.exp(-8.0 * mean_xy)
        height = block_pos[..., 2].amax(dim=1) - block_pos[..., 2].amin(dim=1)
        r_height = torch.clamp(height / self.cfg.stack_height_min, 0.0, 1.0)
        r_stack = 0.5 * (r_together + r_height)

        # 3. Success-Bonus (und Episode-Tracking aktuell halten)
        success = self._check_success()
        self._episode_success |= success
        r_success = success.float()

        # 4. Smoothness: quadratische Strafe auf Gelenkgeschwindigkeit
        r_smooth = -self.robot.data.joint_vel.pow(2).mean(dim=1)

        return (
            self.cfg.rew_reach * r_reach
            + self.cfg.rew_stack * r_stack
            + self.cfg.rew_success * r_success
            + self.cfg.rew_smooth * r_smooth
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._episode_step += 1
        terminated = self._check_success()
        time_out = self._episode_step >= int(self.cfg.episode_length_s * self.cfg.policy_hz)
        return terminated, time_out

    def _check_success(self) -> torch.Tensor:
        """
        Erfolg = Würfel vertikal ausgerichtet und stabil.
        Prüft paarweise horizontalen Abstand der Würfel-Schwerpunkte.
        """
        positions = torch.stack(
            [block.data.root_pos_w for block in self.blocks], dim=1
        )  # (num_envs, 3, 3)

        # Alle 3 Würfel müssen xy-nahe beieinander sein
        xy_ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(len(self.blocks)):
            for j in range(i + 1, len(self.blocks)):
                dist_xy = torch.norm(
                    positions[:, i, :2] - positions[:, j, :2], dim=-1
                )
                xy_ok &= dist_xy < self.cfg.stack_xy_tol

        # Turm-Höhe: höchster Würfel - niedrigster Würfel > Schwellwert
        z_max = positions[:, :, 2].max(dim=1).values
        z_min = positions[:, :, 2].min(dim=1).values
        height_ok = (z_max - z_min) > self.cfg.stack_height_min

        # Stabilität: alle Würfel haben geringe Geschwindigkeit
        vels = torch.stack(
            [block.data.root_lin_vel_w for block in self.blocks], dim=1
        )  # (num_envs, 3, 3)
        speed = vels.norm(dim=-1).max(dim=1).values
        stable = speed < self.cfg.stack_vel_max

        return xy_ok & height_ok & stable

    # ------------------------------------------------------------------
    # Hilfsmethoden für den Eval-Runner
    # ------------------------------------------------------------------

    def get_obs_for_policy(self) -> dict[str, np.ndarray]:
        """
        Gibt das Observation-Dict im GR00T-Eingabeformat zurück (numpy, CPU).
        Wird direkt vom Eval-Runner aufgerufen.
        """
        obs_tensor = self._get_observations()
        obs_np = {}
        for key, val in obs_tensor.items():
            arr = val[0].cpu().numpy()  # Env-Index 0 (single env)
            if key.startswith("video."):
                obs_np[key] = arr.astype(np.uint8)
            else:
                obs_np[key] = arr.astype(np.float32)
        # Umschlüsseln: "joint_pos" → "state.joint_pos"
        obs_np["state.joint_pos"] = obs_np.pop("joint_pos")
        return obs_np

    def get_obs_batched(self) -> dict[str, torch.Tensor]:
        """Batched Observation-Dict ALLER num_envs als GPU-Tensoren — für den RL-Rollout.

        Gegenstück zu get_obs_for_policy() (das nur Env 0 als numpy liefert). Behält die
        Tensoren auf dem Device (kein numpy/CPU-Roundtrip), damit der RL-Trainer die Policy
        batched über alle Envs auswerten kann. Schlüssel wie im GR00T-Format:
            "state.joint_pos": (num_envs, 28) float32
            "video.<cam>":     (num_envs, H, W, 3) uint8
        """
        obs = self._get_observations()
        out: dict[str, torch.Tensor] = {}
        for key, val in obs.items():
            if key.startswith("video."):
                out[key] = val.to(torch.uint8)
            elif key == "joint_pos":
                out["state.joint_pos"] = val.to(torch.float32)
            else:
                out[key] = val
        return out

    @property
    def episode_success(self) -> bool:
        """Gibt zurück, ob in der aktuellen Episode Erfolg verbucht wurde."""
        return bool(self._episode_success[0].item())
