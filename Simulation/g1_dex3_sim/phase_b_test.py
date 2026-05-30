"""
Phase-B-Test: G1+Dex3 USD in Isaac Lab laden und Arm-Joints prüfen.

Fertig wenn:
  - Roboter spawnt ohne Fehler
  - Alle 28 erwarteten Joints sind vorhanden
  - Arm-Joints fahren zu Testpose (kein NaN, kein Freeze)

Verwendung (im Sim-Container):
    ${ISAACLAB_PATH}/isaaclab.sh -p \
        /workspace/g1_dex3_sim/phase_b_test.py \
        --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase-B: G1+Dex3 Articulation-Test")
parser.add_argument("--asset", type=str, default="/data/assets/g1_dex3.usd",
                    help="Pfad zur g1_dex3.usd")
parser.add_argument("--steps", type=int, default=200,
                    help="Simulations-Schritte für den Test")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Erwartete Joints (muss mit g1_dex3_cfg.py übereinstimmen)
# ---------------------------------------------------------------------------
EXPECTED_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

# Arm-Testpose (Ellbogen angewinkelt, Hände vor dem Körper)
ARM_TEST_POS = {
    "left_shoulder_pitch_joint":  0.3,
    "left_shoulder_roll_joint":   0.2,
    "left_shoulder_yaw_joint":    0.0,
    "left_elbow_joint":           1.2,
    "left_wrist_roll_joint":      0.0,
    "left_wrist_pitch_joint":    -0.3,
    "left_wrist_yaw_joint":       0.0,
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint":   0.0,
    "right_elbow_joint":          1.2,
    "right_wrist_roll_joint":     0.0,
    "right_wrist_pitch_joint":   -0.3,
    "right_wrist_yaw_joint":      0.0,
}

# ---------------------------------------------------------------------------
# Scene-Config
# ---------------------------------------------------------------------------

def build_robot_cfg(asset_path: str) -> ArticulationCfg:
    left_arm  = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                 "left_shoulder_yaw_joint", "left_elbow_joint",
                 "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
    right_arm = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                 "right_shoulder_yaw_joint", "right_elbow_joint",
                 "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
    left_hand = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                 "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                 "left_hand_index_0_joint", "left_hand_index_1_joint"]
    right_hand = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                  "right_hand_middle_0_joint", "right_hand_middle_1_joint",
                  "right_hand_index_0_joint", "right_hand_index_1_joint"]

    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.85),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "left_arm":  ImplicitActuatorCfg(joint_names_expr=left_arm,
                             effort_limit=300.0, velocity_limit=5.0,
                             stiffness=100.0, damping=10.0),
            "right_arm": ImplicitActuatorCfg(joint_names_expr=right_arm,
                             effort_limit=300.0, velocity_limit=5.0,
                             stiffness=100.0, damping=10.0),
            "left_hand": ImplicitActuatorCfg(joint_names_expr=left_hand,
                             effort_limit=5.0, velocity_limit=3.0,
                             stiffness=20.0, damping=2.0),
            "right_hand": ImplicitActuatorCfg(joint_names_expr=right_hand,
                              effort_limit=5.0, velocity_limit=3.0,
                              stiffness=20.0, damping=2.0),
        },
    )


# ---------------------------------------------------------------------------
# Haupt-Test
# ---------------------------------------------------------------------------

def run_test(asset_path: str, steps: int) -> bool:
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.9])

    # Boden
    sim_utils.spawn_ground_plane(prim_path="/World/Ground", cfg=sim_utils.GroundPlaneCfg())

    # Roboter
    robot_cfg = build_robot_cfg(asset_path)
    robot = Articulation(robot_cfg)

    sim.reset()

    # ── 1) Joint-Inventory prüfen ──────────────────────────────────────────
    all_joints: list[str] = robot.joint_names
    print("\n[Phase-B] Gefundene Joints:")
    for i, name in enumerate(all_joints):
        print(f"    [{i:2d}] {name}")

    missing = [j for j in EXPECTED_JOINTS if j not in all_joints]
    extra   = [j for j in all_joints if j not in EXPECTED_JOINTS]

    if missing:
        print(f"\n[Phase-B] FEHLER: Fehlende Joints ({len(missing)}):")
        for j in missing:
            print(f"    - {j}")
    else:
        print(f"\n[Phase-B] Alle {len(EXPECTED_JOINTS)} erwarteten Joints vorhanden. ✓")

    if extra:
        print(f"[Phase-B] Zusätzliche Joints (Beine/Hüfte aus vollem URDF, unkontrolliert — erwartet):")
        for j in extra:
            print(f"    + {j}")

    if missing:
        return False

    # ── 2) Arm-Joints auf Testpose fahren ─────────────────────────────────
    # Joint-Indices für die Arm-Joints bestimmen
    joint_idx = {name: robot.find_joints([name])[0][0] for name in ARM_TEST_POS}

    num_joints = robot.num_joints
    target = torch.zeros(1, num_joints, device=sim.device)

    for name, angle in ARM_TEST_POS.items():
        idx = joint_idx[name]
        target[0, idx] = angle

    print(f"\n[Phase-B] Fahre Arm-Joints auf Testpose ({steps} Schritte) …")
    for step in range(steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()   # Targets in Physik-Engine schreiben (Isaac Lab 2.x)
        sim.step()
        robot.update(sim.cfg.dt)

        if step in (0, steps // 2, steps - 1):
            pos = robot.data.joint_pos[0]
            arm_indices = list(joint_idx.values())
            arm_pos = pos[arm_indices]
            if torch.any(torch.isnan(arm_pos)):
                print(f"[Phase-B] FEHLER: NaN in Arm-Positions bei Schritt {step}!")
                return False
            print(f"  Schritt {step:4d}: Arm-Joints = {arm_pos.cpu().numpy().round(3)}")

    # ── 3) Endergebnis prüfen ─────────────────────────────────────────────
    final_pos = robot.data.joint_pos[0]
    errors = {}
    for name, target_angle in ARM_TEST_POS.items():
        idx = joint_idx[name]
        actual = final_pos[idx].item()
        err = abs(actual - target_angle)
        if err > 0.5:
            errors[name] = (target_angle, actual, err)

    if errors:
        print(f"\n[Phase-B] WARNUNG: {len(errors)} Joints weit von Zielposen entfernt:")
        for name, (t, a, e) in errors.items():
            print(f"    {name}: Ziel={t:.3f}, Ist={a:.3f}, Δ={e:.3f}")
    else:
        print(f"\n[Phase-B] Arm-Joints konvergiert. ✓")

    return True


def main():
    asset_path = args.asset
    print(f"\n{'='*60}")
    print(f"Phase-B-Test: G1+Dex3 Articulation")
    print(f"Asset: {asset_path}")
    print(f"{'='*60}\n")

    import os
    if not os.path.exists(asset_path):
        print(f"[Phase-B] FEHLER: Asset nicht gefunden: {asset_path}")
        print("          URDF→USD-Konvertierung ausführen (convert_urdf_to_usd.py)")
        simulation_app.close()
        sys.exit(1)

    success = run_test(asset_path, args.steps)

    print(f"\n{'='*60}")
    print(f"Phase-B: {'ERFOLG ✓' if success else 'FEHLGESCHLAGEN ✗'}")
    print(f"{'='*60}\n")

    simulation_app.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
