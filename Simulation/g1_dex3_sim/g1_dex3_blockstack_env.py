"""
Block-Stacking-Umgebung für G1 + Dex3 in Isaac Lab.

Implementiert als DirectRLEnv (Isaac Lab 2.x):
    - Tisch + 3 Würfel mit randomisierten Startpositionen
    - 4 RGB-Kameras (cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist)
    - 28-dim Joint-State (left_arm + right_arm + left_dex3 + right_dex3)
    - Erfolgsmetrik: Würfel vertikal gestapelt
    - Episodenlänge: 600 Steps @ 30 Hz = 20 Sekunden

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
# Szenenkonfiguration
# ---------------------------------------------------------------------------

@configclass
class G1Dex3BlockstackSceneCfg(InteractiveSceneCfg):
    """Szene: Roboter am Tisch mit 3 Würfeln."""

    # Tisch (einfache Box als Placeholder — für echten Tisch USD ersetzen)
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.6, 0.74),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.45, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.37)),
    )

    # 3 Würfel (5 cm Kantenlänge, unterschiedliche Farben)
    block_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/block_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.1, 0.77)),
    )

    block_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/block_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.77)),
    )

    block_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/block_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.1, 0.77)),
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
            convention="ros",
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
            convention="ros",
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

    num_envs = 1
    env_spacing = 2.0


# ---------------------------------------------------------------------------
# Env-Konfiguration
# ---------------------------------------------------------------------------

@configclass
class G1Dex3BlockstackEnvCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = SimulationCfg(
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
    episode_length_s: float = 20.0    # 600 Steps @ 30 Hz
    policy_hz: float = 30.0
    execution_horizon: int = 8        # wie viele Chunk-Steps ausführen, dann re-plan

    # Block-Sampling-Bereich (Tischoberfläche)
    block_x_range: tuple[float, float] = (0.35, 0.65)
    block_y_range: tuple[float, float] = (-0.2, 0.2)
    block_z_surface: float = 0.77     # Tischoberfläche + halbe Würfel-Höhe

    # Erfolgsparameter
    stack_xy_tol: float = 0.03  # max. horizontaler Versatz zwischen Würfel-Mittelpunkten
    stack_height_min: float = 0.08  # Mindesthöhe des Turms (2 × 0.05 m - Toleranz)
    stack_vel_max: float = 0.05  # max. Geschwindigkeit für "stabil gestapelt"

    # Task-Beschreibung (geht ans Language-Modell)
    task_description: str = "stack the blocks"


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

        # Episode-Tracking
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        }

        self.scene.filter_collisions(global_prim_paths=[])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._episode_step[env_ids] = 0
        self._episode_success[env_ids] = False

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
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
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
        """Einfache binary-Success-Reward für Ablations; nicht für RL-Training."""
        success = self._check_success()
        self._episode_success |= success
        return success.float()

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

    @property
    def episode_success(self) -> bool:
        """Gibt zurück, ob in der aktuellen Episode Erfolg verbucht wurde."""
        return bool(self._episode_success[0].item())
