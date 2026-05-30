"""
Konvertiert das G1+Hand URDF in eine Isaac Sim USD-Datei (Isaac Lab 2.x API).

Verwendung (im Sim-Container):
    ${ISAACLAB_PATH}/isaaclab.sh -p \
        /workspace/g1_dex3_sim/convert_urdf_to_usd.py \
        --headless \
        --urdf /data/assets/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf \
        --output /data/assets/g1_dex3.usd
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="URDF → USD Konvertierung für G1+Hand")
parser.add_argument("--urdf", type=str,
    default="/data/assets/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf",
    help="Pfad zur URDF-Datei")
parser.add_argument("--output", type=str,
    default="/data/assets/g1_dex3.usd",
    help="Ausgabepfad für die USD-Datei (muss .usd enden)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
from pathlib import Path

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

urdf_path = Path(args.urdf)
output_path = Path(args.output)

print(f"[convert] URDF:   {urdf_path}")
print(f"[convert] Output: {output_path}")

if not urdf_path.exists():
    print(f"[convert] FEHLER: URDF nicht gefunden: {urdf_path}")
    sys.exit(1)

output_path.parent.mkdir(parents=True, exist_ok=True)

cfg = UrdfConverterCfg(
    asset_path=str(urdf_path),
    usd_dir=str(output_path.parent),
    usd_file_name=output_path.name,
    fix_base=False,
    merge_fixed_joints=False,
    self_collision=False,
    make_instanceable=False,
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=100.0,
            damping=10.0,
        ),
    ),
)

converter = UrdfConverter(cfg)

usd_file = output_path.parent / output_path.name
if usd_file.exists():
    print(f"[convert] Erfolg: USD gespeichert unter {usd_file}")
    print(f"[convert] Dateigröße: {usd_file.stat().st_size / 1024:.0f} KB")
else:
    print(f"[convert] FEHLER: USD-Datei wurde nicht erzeugt.")
    sys.exit(1)

simulation_app.close()
