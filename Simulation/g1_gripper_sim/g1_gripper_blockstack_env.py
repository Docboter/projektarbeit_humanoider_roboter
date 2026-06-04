"""
Block-Stacking-Umgebung für den **stock Unitree G1 + Dex1-Parallelgreifer**
(Baseline-Test mit un-finetuntem GR00T-N1.6-3B, Embodiment ``UNITREE_G1``).

Spiegelt ``g1_dex3_sim/g1_dex3_blockstack_env.py`` und verwendet **dieselbe**
Tisch-/Würfel-/Erfolgs-Logik, damit die Success-Rate direkt mit dem DEX3-Lauf
vergleichbar ist. Eigenständig — der DEX3-Pfad bleibt unverändert.

Unterschiede zum DEX3-Env:
    - Roboter = G1 mit Dex1-Greifer (16 policy-gesteuerte Action-Dims: 14 Arm + 2 Hand)
    - genau EINE Policy-Kamera ``ego_view`` (UNITREE_G1-Embodiment)
    - Beine/Waist werden per PD auf Default gehalten (nicht policy-gesteuert)
    - keine DEX3-Vorzeichen-Flips, kein Hand-Recolor
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from g1_gripper_cfg import (
    CAMERA_CFG,
    G1_GRIPPER_CFG,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    POLICY_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
)

# ---------------------------------------------------------------------------
# Szenenkonfiguration (Tisch + 3 Würfel + Stapel-Band — identisch zum DEX3-Env)
# ---------------------------------------------------------------------------

@configclass
class G1GripperBlockstackSceneCfg(InteractiveSceneCfg):
    """Szene: Roboter am Tisch mit 3 Würfeln (gespiegelt vom DEX3-Env)."""

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.6, 0.89),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.445)),
    )

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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.70, 0.10)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, 0.15, 0.915)),
    )

    stack_band: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stack_band",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.04, 0.006),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.893)),
    )

    robot: ArticulationCfg = G1_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # Genau EINE Policy-Kamera: ego_view, am Torso-Link montiert (wie die reale D435).
    ego_view: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/torso_link/ego_view",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_CFG.ego_view_local["pos"],
            rot=CAMERA_CFG.ego_view_local["rot"],
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        width=CAMERA_CFG.width,
        height=CAMERA_CFG.height,
    )

    # Szenen-Übersichtskamera — NUR fürs Video.
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
class G1GripperBlockstackEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=7,
    )
    scene: G1GripperBlockstackSceneCfg = G1GripperBlockstackSceneCfg(num_envs=1, env_spacing=2.0)

    decimation: int = 7
    action_space: int = 16        # 14 Arm + 2 Hand (binär)
    observation_space: int = 16

    episode_length_s: float = 40.0
    policy_hz: float = 30.0
    execution_horizon: int = 8

    block_x_range: tuple[float, float] = (0.30, 0.40)
    block_y_range: tuple[float, float] = (-0.20, 0.20)
    block_z_surface: float = 0.915

    stack_xy_tol: float = 0.03
    stack_height_min: float = 0.08
    stack_vel_max: float = 0.05

    task_description: str = "stack the blocks"


# ---------------------------------------------------------------------------
# Umgebung
# ---------------------------------------------------------------------------

class G1GripperBlockstackEnv(DirectRLEnv):
    """Closed-Loop-Env für die GR00T-N1.6-Baseline (stock G1 + Dex1-Greifer)."""

    cfg: G1GripperBlockstackEnvCfg

    def __init__(self, cfg: G1GripperBlockstackEnvCfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode=render_mode)

        # Joint-Index-Mappings (lazy nach erstem Schritt aufgebaut)
        self._arm_ids: list[int] | None = None
        self._left_grip_ids: list[int] | None = None
        self._right_grip_ids: list[int] | None = None

        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    def _setup_scene(self):
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.robot: Articulation = self.scene["robot"]
        self.table: RigidObject = self.scene["table"]
        self.blocks: list[RigidObject] = [
            self.scene["block_0"], self.scene["block_1"], self.scene["block_2"],
        ]
        self.cameras: dict[str, TiledCamera] = {
            "ego_view": self.scene["ego_view"],
            "cam_scene": self.scene["cam_scene"],  # nur Video
        }
        self.scene.filter_collisions(global_prim_paths=[])

    # ------------------------------------------------------------------
    def _ensure_joint_ids(self):
        if self._arm_ids is not None:
            return
        name_to_idx = {n: i for i, n in enumerate(self.robot.data.joint_names)}

        def ids(names: list[str]) -> list[int]:
            for n in names:
                if n not in name_to_idx:
                    raise ValueError(
                        f"Joint '{n}' nicht im USD-Asset gefunden. "
                        f"Verfügbar: {list(name_to_idx.keys())}"
                    )
            return [name_to_idx[n] for n in names]

        self._arm_ids = ids(POLICY_ARM_JOINTS)             # 14 (left 7 + right 7)
        self._left_grip_ids = ids(LEFT_GRIPPER_JOINTS)     # 2
        self._right_grip_ids = ids(RIGHT_GRIPPER_JOINTS)   # 2

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._episode_step[env_ids] = 0
        self._episode_success[env_ids] = False

        default_pos = self.robot.data.default_joint_pos[env_ids]
        default_vel = torch.zeros_like(default_pos)
        self.robot.set_joint_position_target(default_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)

        for block in self.blocks:
            block_pos = torch.zeros(len(env_ids), 3, device=self.device)
            block_pos[:, 0] = sample_uniform(
                self.cfg.block_x_range[0], self.cfg.block_x_range[1],
                (len(env_ids),), device=self.device)
            block_pos[:, 1] = sample_uniform(
                self.cfg.block_y_range[0], self.cfg.block_y_range[1],
                (len(env_ids),), device=self.device)
            block_pos[:, 2] = self.cfg.block_z_surface
            block_quat = torch.zeros(len(env_ids), 4, device=self.device)
            block_quat[:, 0] = 1.0
            block.write_root_pose_to_sim(
                torch.cat([block_pos, block_quat], dim=-1), env_ids=env_ids)
            block.write_root_velocity_to_sim(
                torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict[str, Any]:
        self._ensure_joint_ids()
        obs: dict[str, Any] = {}
        obs["arm_pos"] = self.robot.data.joint_pos[:, self._arm_ids].clone()  # (N, 14)
        # Greifer-Öffnung je Hand als Skalar (Mittel der beiden Finger-Joints).
        lg = self.robot.data.joint_pos[:, self._left_grip_ids].mean(dim=1, keepdim=True)
        rg = self.robot.data.joint_pos[:, self._right_grip_ids].mean(dim=1, keepdim=True)
        obs["gripper"] = torch.cat([lg, rg], dim=1)  # (N, 2)
        for cam_name, camera in self.cameras.items():
            rgb = camera.data.output["rgb"]
            obs[f"video.{cam_name}"] = rgb[..., :3]
        return obs

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """actions: (N, 16) = [left_arm(7), right_arm(7), left_hand(1), right_hand(1)].

        Arme: absolute Targets (Server hat RELATIVE→ABS dekodiert).
        Hände: binäres Greifer-Signal → beide Finger-Joints, geклemmt auf URDF-Range.
        """
        self._ensure_joint_ids()
        actions = actions.clone()

        full_target = self.robot.data.joint_pos.clone()
        # Arme
        for k, isaac_idx in enumerate(self._arm_ids):
            full_target[:, isaac_idx] = actions[:, k]
        # Greifer: hand-Werte auf Greifer-Range klemmen, auf beide Finger spiegeln.
        left_cmd = actions[:, 14].clamp(GRIPPER_CLOSE, GRIPPER_OPEN)
        right_cmd = actions[:, 15].clamp(GRIPPER_CLOSE, GRIPPER_OPEN)
        for isaac_idx in self._left_grip_ids:
            full_target[:, isaac_idx] = left_cmd
        for isaac_idx in self._right_grip_ids:
            full_target[:, isaac_idx] = right_cmd
        self._full_target = full_target

    def _apply_action(self) -> None:
        if not hasattr(self, "_full_target"):
            return
        self.robot.set_joint_position_target(self._full_target)

    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        success = self._check_success()
        self._episode_success |= success
        return success.float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._episode_step += 1
        terminated = self._check_success()
        time_out = self._episode_step >= int(self.cfg.episode_length_s * self.cfg.policy_hz)
        return terminated, time_out

    def _check_success(self) -> torch.Tensor:
        positions = torch.stack(
            [block.data.root_pos_w for block in self.blocks], dim=1)  # (N, 3, 3)
        xy_ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(len(self.blocks)):
            for j in range(i + 1, len(self.blocks)):
                dist_xy = torch.norm(positions[:, i, :2] - positions[:, j, :2], dim=-1)
                xy_ok &= dist_xy < self.cfg.stack_xy_tol
        z_max = positions[:, :, 2].max(dim=1).values
        z_min = positions[:, :, 2].min(dim=1).values
        height_ok = (z_max - z_min) > self.cfg.stack_height_min
        vels = torch.stack([block.data.root_lin_vel_w for block in self.blocks], dim=1)
        speed = vels.norm(dim=-1).max(dim=1).values
        stable = speed < self.cfg.stack_vel_max
        return xy_ok & height_ok & stable

    # ------------------------------------------------------------------
    def get_obs_for_policy(self) -> dict[str, np.ndarray]:
        """Numpy-Obs (Env 0) für den Eval-Runner / client_g1.build_obs."""
        obs_tensor = self._get_observations()
        out: dict[str, np.ndarray] = {}
        for key, val in obs_tensor.items():
            arr = val[0].cpu().numpy()
            out[key] = arr.astype(np.uint8) if key.startswith("video.") else arr.astype(np.float32)
        return out

    @property
    def episode_success(self) -> bool:
        return bool(self._episode_success[0].item())
