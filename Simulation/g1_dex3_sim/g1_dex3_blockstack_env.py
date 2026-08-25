# TL;DR: Isaac-Lab-Env (Roboter, Tisch, Würfel, 4 Policy-Kameras + Szenenkamera) für Eval und RL.
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
import os
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
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from g1_dex3_cfg import (
    ALL_JOINTS_ORDERED,
    CAMERA_CFG,
    DATASET_INIT_STATE,
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
    """SimulationCfg; Anti-Aliasing-Modus per RL_AA_MODE einstellbar (Default: Isaac-Defaults).

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

    # BEFUND 2026-08-08: antialiasing_mode="DLAA" wurde sauber gesetzt (alle drei Felder
    # akzeptiert, die DLSS-Auflösungswarnung verschwand) — die Bilder wurden dadurch aber
    # SCHLECHTER, nicht besser: Chroma 5.93 -> 0.96, Wertebereich auf 246-248 geschrumpft.
    # DLAA ist selbst ein temporales Verfahren; bei wenigen Settle-Steps verschlimmert es
    # die Konvergenz womöglich. Daher KEIN Default mehr, sondern ein Messhebel:
    #   RL_AA_MODE=DLAA|DLSS|FXAA|Off|TAA   (leer/ungesetzt = Isaac-Sim-Defaults)
    # So kostet ein Sweep über die Modi je einen `cams`-Lauf statt einer Code-Änderung.
    mode = os.environ.get("RL_AA_MODE", "").strip()
    if not mode:
        return SimulationCfg(**kwargs)

    render = None
    try:
        from isaaclab.sim import RenderCfg
    except ImportError:
        print("[Env] WARN: isaaclab.sim.RenderCfg nicht vorhanden — RL_AA_MODE wirkungslos.",
              flush=True)
    else:
        available = {f.name for f in dataclasses.fields(RenderCfg)}
        wanted = {"antialiasing_mode": mode, "enable_dlssg": False}
        use = {k: v for k, v in wanted.items() if k in available}
        missing = sorted(set(wanted) - set(use))
        try:
            render = RenderCfg(**use)
        except (TypeError, ValueError) as e:
            print(f"[Env] WARN: RenderCfg({use}) abgelehnt ({e}) — Defaults bleiben aktiv.",
                  flush=True)
            render = None
        else:
            print(f"[Env] RenderCfg gesetzt (RL_AA_MODE={mode}): {use}"
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
            # Tisch 0.87 m hoch (Oberfläche z=0.87). Ein 5-cm-Würfel ruht darauf mit dem
            # Mittelpunkt auf z=0.895, Oberkante 0.92.
            # Tisch war testweise auf 0.89 angehoben, das blockierte aber die Roboterarme im
            # Closed-Loop: Modell versucht zu z≈0.915 zu greifen, Tisch (0.89) steckte die Hände fest.
            # Danach stand block_z_surface auf 0.915, um die Oberkante auf 0.94 zu heben, ohne den
            # Tisch anzuheben. Das erreicht sein Ziel nicht: die Würfel fallen die zwei Zentimeter
            # und ruhen doch auf 0.895 (nachgemessen 2026-08-22 mit `server_rl_run.sh cams`).
            # Übrig blieb nur der Fall beim Reset — und eine um 2 cm falsche Ebene für jede
            # Rückprojektion Bild → Tisch. Seit 2026-08-22 spawnen die Würfel dort, wo sie ruhen.
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, -0.15, 0.895)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36, 0.0, 0.895)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, 0.15, 0.895)),
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
            focal_length=CAMERA_CFG.focal_high,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_CFG.horizontal_aperture_mm,
            # Nahebene 0.15 statt 0.1: schneidet die eigene Kopfschale weg, falls der
            # d435-Montagepunkt knapp innerhalb der Geometrie liegt. Aufgabenrelevantes
            # kommt der Kopfkamera nicht naeher als 0.4 m.
            clipping_range=(0.15, 20.0),
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
            focal_length=CAMERA_CFG.focal_high,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_CFG.horizontal_aperture_mm,
            # Nahebene 0.15 statt 0.1: schneidet die eigene Kopfschale weg, falls der
            # d435-Montagepunkt knapp innerhalb der Geometrie liegt. Aufgabenrelevantes
            # kommt der Kopfkamera nicht naeher als 0.4 m.
            clipping_range=(0.15, 20.0),
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
            focal_length=CAMERA_CFG.focal_wrist,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_CFG.horizontal_aperture_mm,
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
            focal_length=CAMERA_CFG.focal_wrist,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_CFG.horizontal_aperture_mm,
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
            focal_length=CAMERA_CFG.focal_scene,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_CFG.horizontal_aperture_mm,
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

def _parse_rgb(raw: str) -> tuple | None:
    """"0.75,0.73,0.70" -> (0.75, 0.73, 0.70). Leer/unparsbar -> None (Default beibehalten)."""
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if len(parts) != 3:
        return None
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        print(f"[Env] RGB-Wert '{raw}' nicht lesbar — ignoriert.", flush=True)
        return None


def _quat_to_matrix(q) -> np.ndarray:
    """(w, x, y, z) -> 3x3-Rotationsmatrix in Spaltenvektor-Konvention (v_welt = R @ v_lokal)."""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """3x3-Rotationsmatrix (Spaltenvektor-Konvention) -> (w, x, y, z).

    Shepperd-Variante: es wird immer die betragsgrößte Komponente zuerst bestimmt, damit
    keine Division durch eine fast-Null-Spur entsteht.
    """
    m = np.asarray(r, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q = np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s,
                      (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _plain_camera_cfg(tc: TiledCameraCfg) -> CameraCfg:
    """TiledCameraCfg -> CameraCfg mit identischer Pose, Optik und Auflösung.

    Für den Halbierungstest bei leeren Kamerabildern: rendert die gewöhnliche `Camera`
    dieselbe Pose korrekt, liegt es an `TiledCamera` in Isaac Lab 3.0-beta; bleibt auch
    sie leer, liegt es an Szene oder Render-Setup. Kostet Durchsatz (ein Render-Product
    je Kamera und Env statt eines gekachelten), ist also nur ein Messwerkzeug.
    """
    return CameraCfg(
        prim_path=tc.prim_path,
        offset=CameraCfg.OffsetCfg(
            pos=tc.offset.pos, rot=tc.offset.rot, convention=tc.offset.convention,
        ),
        data_types=list(tc.data_types),
        spawn=tc.spawn,
        width=tc.width,
        height=tc.height,
    )


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
    # 1 = bisher: nach jedem Policy-Step rendern. >1 = automatische Zwischen-Renderings
    # unterdruecken; der Eval-Runner rendert dann explizit unmittelbar vor get_action().
    camera_render_every_n: int = 1

    # Block-Sampling-Bereich = erreichbarer Greifraum (aus Replay: Hände greifen bei x≈0.35,
    # y≈±0.19, z≈0.92). x/y eng um den Greifraum, z = neue Tischoberfläche (0.87) + halbe Würfelhöhe.
    block_x_range: tuple[float, float] = (0.30, 0.40)
    block_y_range: tuple[float, float] = (-0.20, 0.20)
    block_z_surface: float = 0.895    # Tischoberfläche 0.87 + halbe Würfel-Höhe (0.025)
    # Gierwinkel der Würfel beim Reset, in GRAD. Bis zum 2026-08-25 standen sie fest
    # achsparallel; im Realdatensatz liegen sie schräg zur Tischkante (gemessen am
    # 2026-08-25 über 40 Episoden, Median 10°, p90 36° — Zahl noch maskenlimitiert,
    # siehe extract_block_layout.py). Der Default bleibt 0/0, damit sich an laufenden
    # Vergleichen nichts still ändert; (0, 90) macht die Streuung an.
    # Bei einem 4-zähligen Würfel deckt 0…90° bereits alle Lagen ab.
    block_yaw_range_deg: tuple[float, float] = (0.0, 0.0)

    # Erfolgsparameter
    stack_xy_tol: float = 0.03  # max. horizontaler Versatz zwischen Würfel-Mittelpunkten
    stack_height_min: float = 0.08  # Mindesthöhe des Turms (2 × 0.05 m - Toleranz)
    stack_vel_max: float = 0.05  # max. Geschwindigkeit für "stabil gestapelt"
    # Dataset-Replays müssen auch nach einem erfolgreichen Zwischenzustand bis zum Ende
    # der Originalepisode weiterlaufen. Der Default bleibt für Eval/RL unverändert.
    terminate_on_success: bool = True

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

    def __post_init__(self):
        post = getattr(super(), "__post_init__", None)
        if callable(post):
            post()

        try:
            render_every = int(os.environ.get("CAMERA_RENDER_EVERY_N", "1"))
        except ValueError as exc:
            raise ValueError("CAMERA_RENDER_EVERY_N muss eine positive Ganzzahl sein") from exc
        if render_every < 1:
            raise ValueError("CAMERA_RENDER_EVERY_N muss >= 1 sein")
        self.camera_render_every_n = render_every
        if render_every > 1:
            # DirectRLEnv wuerde sonst innerhalb jedes Action-Chunks weiterhin rendern,
            # obwohl diese Frames nie an die Policy gehen. Der Runner ruft unmittelbar
            # vor jeder Policy-Observation force_policy_camera_render() auf.
            self.sim.render_interval = 1_000_000_000
            print(
                f"[cam] synchrones Policy-Rendering aktiv: alle {render_every} Schritte; "
                "automatische Zwischen-Renderings aus.",
                flush=True,
            )

        # ── SCENE_CAM=0: Übersichtskamera weglassen ────────────────────────────────
        # Gemessen im Sim-Eval (2026-08-13, LIVESTREAM=2): 94 % der Wanduhr stecken in
        # Sim+Render, nur 5 % in der Policy-Inferenz. Von den fünf Kameras geht genau
        # eine — cam_scene — ausschließlich ins MP4; die Policy sieht sie nie. Läuft der
        # Lauf live (WebRTC-Viewport) oder ohne Video, wird sie also jeden Step umsonst
        # gerendert. Abschalten kostet daher NICHTS an Aussagekraft: die Modell-Eingabe
        # bleibt Byte für Byte dieselbe, Erfolgsraten bleiben vergleichbar.
        if os.environ.get("SCENE_CAM", "1").strip().lower() in ("0", "false", "no"):
            if "cam_scene" in vars(self.scene):
                delattr(self.scene, "cam_scene")
                print("[cam] SCENE_CAM=0 → cam_scene wird nicht angelegt (eine von fünf "
                      "Kameras weniger je Step). Kein MP4, keine Übersicht in Spur B.",
                      flush=True)

        # ── CAM_RES_SCALE: Kameraauflösung skalieren ───────────────────────────────
        # Der stärkste Renderhebel (Pixelzahl geht quadratisch ein), aber der einzige, der
        # die MODELL-EINGABE verändert: die 640×480 sind gegen die Dataset-Referenzframes
        # kalibriert. Ein Lauf mit Skalierung ist zum Zuschauen gedacht — seine Erfolgsrate
        # ist NICHT mit den bisherigen Läufen vergleichbar. Deshalb Default 1.0 und eine
        # laute Warnung, statt still schneller zu werden.
        try:
            scale = float(os.environ.get("CAM_RES_SCALE", "1").strip() or 1.0)
        except ValueError:
            scale = 1.0
        if scale > 0 and abs(scale - 1.0) > 1e-6:
            for name in ("cam_left_high", "cam_right_high", "cam_left_wrist",
                         "cam_right_wrist", "cam_scene"):
                cam = vars(self.scene).get(name)
                if cam is None:
                    continue
                # Auf Vielfache von 8 runden — der Tiled-Renderer legt die Bilder in einem
                # gemeinsamen Puffer ab, krumme Kantenlängen sind dort unnötiges Risiko.
                cam.width = max(64, int(round(cam.width * scale / 8)) * 8)
                cam.height = max(64, int(round(cam.height * scale / 8)) * 8)
            first = vars(self.scene).get("cam_left_high")
            print(f"[cam] !! CAM_RES_SCALE={scale} → Kameras auf "
                  f"{getattr(first, 'width', '?')}×{getattr(first, 'height', '?')}. "
                  f"Das ist NICHT die kalibrierte Auflösung: nur zum Zuschauen, "
                  f"Erfolgsraten dieses Laufs nicht mit anderen vergleichen.", flush=True)

        # ── RL_CAMERA_CLASS=camera: Halbierungstest für leere Kamerabilder ──────────
        # Befund 2026-08-08 (runs/20260808/11): cam_left_high, cam_right_high und
        # cam_right_wrist liefern ein Bild mit GENAU EINEM Grauwert, während
        # cam_left_wrist die Hand und cam_scene den Boden zeigen — bei identischer
        # Konfiguration und mit abgeschalteter DR. Drei Sensoren schreiben ihren Puffer
        # also gar nicht. Diese Umschaltung rendert dieselben Posen mit der
        # gewöhnlichen `Camera`: zeigt die den Tisch, liegt es an `TiledCamera` in
        # Isaac Lab 3.0-beta; bleibt sie leer, liegt es an Szene oder Render-Setup.
        if os.environ.get("RL_CAMERA_CLASS", "tiled").strip().lower() != "camera":
            return
        done = []
        for name in ("cam_left_high", "cam_right_high", "cam_left_wrist",
                     "cam_right_wrist", "cam_scene"):
            # vars() statt getattr: nach einem SCENE_CAM=0-delattr könnte getattr sonst
            # einen Klassen-Default zurückgeben und die Kamera hier wieder eintragen.
            tc = vars(self.scene).get(name)
            if tc is None:
                print(f"[cam] '{name}' nicht in der Szene — übersprungen.", flush=True)
                continue
            setattr(self.scene, name, _plain_camera_cfg(tc))
            done.append(name)
        print(f"[cam] RL_CAMERA_CLASS=camera → {len(done)} Sensoren auf Camera "
              f"umgestellt: {', '.join(done) or '(keine!)'}", flush=True)


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
        self._reach_body_ids: list[int] | None = None  # Kontaktflächen (für die Reach-Diagnose)
        self._reach_frame: str = "?"
        self._reach_offsets: torch.Tensor | None = None  # lokaler Versatz Gelenk→Fingerkuppe

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
        # Der Isaac-Default-Boden ist fast schwarz mit weißem Raster und füllt in den
        # Kopfkameras ~45 %, in den Wrist-Kameras ~25 % des Bildes — im Referenzbild ist
        # dort heller Laborboden und weiße Wand. RL_GROUND_COLOR hellt ihn auf; ungesetzt
        # bleibt der Isaac-Default. Gemessen wird die Wirkung mit
        #   RL_GROUND_COLOR=... ./Simulation/server_rl_run.sh cams  &&  … gap
        # Achtung (runs/20260808/22): der Wert setzt KEINE neutrale Albedo, sondern wirkt als
        # Tint auf Isaacs Rastertextur. Der nahezu neutrale Wert 0.35,0.35,0.36 rendert
        # sichtbar blau, die Chroma steigt vom Default 2.7 auf 19.7 (real: 9.1). Aufhellen
        # erhöht hier also zwangsläufig die Sättigung; ein wirklich neutraler Boden bräuchte
        # ein ersetztes Material statt einer Tönung.
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_color = _parse_rgb(os.environ.get("RL_GROUND_COLOR", ""))
        if ground_color is not None:
            ground_cfg.color = ground_color
            print(f"[Env] Bodenfarbe -> {ground_color} (RL_GROUND_COLOR)", flush=True)
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        # Dome-Intensität als Messhebel (RL_DOME_INTENSITY), Default unverändert 2000.
        # Die Juni-Referenzbilder mit korrektem Kontrast (min/median/max 32/229/239)
        # entstanden unter Isaac Sim 4.x, die gleichmäßig hellen (245/248/249) unter 6.0 —
        # bei identischer Config. Verdacht: 6.0 bewertet die Intensitätseinheit anders und
        # überstrahlt die Szene. Diagnose in einem Lauf:
        #   RL_DOME_SWEEP=500,120,30 ./Simulation/server_rl_run.sh cams
        dome_intensity = float(os.environ.get("RL_DOME_INTENSITY", 2000.0))
        light_cfg = sim_utils.DomeLightCfg(intensity=dome_intensity, color=(0.75, 0.75, 0.75))
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
        # Nur fürs Video — bei SCENE_CAM=0 gar nicht erst angelegt. Alles, was sie
        # konsumiert, kennt den Fall bereits: der Eval-Runner weicht auf cam_left_high
        # aus, die Live-Ansicht ebenso, und _force_camera_prim_orientations() läuft über
        # genau dieses Dict.
        try:
            self.cameras["cam_scene"] = self.scene["cam_scene"]
        except KeyError:
            print("[cam] cam_scene nicht in der Szene — Übersicht/Video entfallen "
                  "(SCENE_CAM=0).", flush=True)

        self.scene.filter_collisions(global_prim_paths=[])

        self._force_camera_prim_orientations()

    # ------------------------------------------------------------------
    # Kamera-Ausrichtung direkt in die Stage schreiben
    # ------------------------------------------------------------------

    # Spalten = die lokalen USD-Kameraachsen, ausgedrückt in Isaac-Lab-``convention="world"``
    # (dort ist +X Blickrichtung, +Y links, +Z oben). USD blickt entlang -Z, oben ist +Y,
    # rechts ist +X:  X_usd = -Y_welt,  Y_usd = +Z_welt,  Z_usd = -X_welt.
    _WORLD_TO_USD = np.array([[0.0, 0.0, -1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]])

    def _force_camera_prim_orientations(self):
        """Die Orientierung jedes Kamera-Prims selbst schreiben, statt sie Isaac Lab zu überlassen.

        Befund 2026-08-08 (``runs/20260808/13``): der USD-Abgleich in ``dump_camera_poses.py``
        zeigt, dass die Kamera-Prims in der Stage **anders** ausgerichtet sind als
        ``cam.data`` meldet — 95.6° bei den High-Cams, 102.8° bei den Wrist-Cams, 136.8° bei
        ``cam_scene``. Die Position stimmt exakt, die Optik am Prim (focal/aperture) auch.

        Warum ``cam.data`` das nicht sieht: Isaac Lab rechnet die Offset-Rotation beim Spawn
        von ``convention="world"`` nach OpenGL um und beim Lesen wieder zurück. Ist die
        Umrechnung fehlerhaft, hebt der Rückweg sie auf — ``quat_w_world`` gibt brav die
        Config zurück, während der Renderer die verdrehte Pose benutzt. Genau deshalb waren
        alle bisherigen „IM BILD"-Aussagen wertlos.

        Dass die Stage-Pose die gerenderte ist, bestätigen die Bilder Zug um Zug: die
        High-Cams blicken nach oben (nur Dome), ``cam_right_wrist`` in den leeren Raum
        (genau EIN Grauwert), ``cam_left_wrist`` in den Roboter (die einzige Kamera mit
        Inhalt) und ``cam_scene`` knapp über den Horizont (der schmale Bodenkeil unten).

        **Bestätigt in ``runs/20260808/14``:** nach dem Schreiben zeigt `cam_left_high`
        Tisch, Würfel und die eigene Hand (min/median/max 0/229/241, chroma 6.9, dunkel
        26.9 % — die Juni-Referenz unter Isaac Sim 4.x lag bei 32/229/239, chroma 6.0,
        11–22 %), `cam_scene` die vollständige Szene mit Roboter, Tisch und Würfeln.
        Umgekehrt meldet ``cam.data`` seitdem eine *falsche* Blickrichtung — dieselbe
        kaputte Umrechnung, nur rückwärts. Die Frustum-Diagnostik im Kamera-Dump ist damit
        unbrauchbar, das Bild ist die Referenz.

        Abschalten mit ``RL_CAM_USD_FIX=0``.
        """
        if os.environ.get("RL_CAM_USD_FIX", "1").strip().lower() in ("0", "false", "no"):
            print("[cam] RL_CAM_USD_FIX=0 → Prim-Orientierung bleibt wie von Isaac Lab "
                  "geschrieben.", flush=True)
            return
        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
        except Exception as e:  # noqa: BLE001
            print(f"[cam] !! USD-Stage nicht erreichbar ({e}) — Orientierung unverändert.",
                  flush=True)
            return
        if stage is None:
            print("[cam] !! USD-Stage ist None — Orientierung unverändert.", flush=True)
            return

        fixed, failed = 0, 0
        for name, cam in self.cameras.items():
            # Die Offset-Rotation vom Sensor selbst holen, nicht über CAMERA_CFG: die
            # Wrist-Kameras heißen dort `cam_*_wrist_local` und wurden in Lauf 14 deshalb
            # stillschweigend übersprungen. `cam.cfg.offset` ist genau die Quelle, aus der
            # auch Isaac Lab beim Spawn liest — parent-relativ, für alle fünf gleich.
            offset = getattr(getattr(cam, "cfg", None), "offset", None)
            if offset is None or getattr(offset, "rot", None) is None:
                print(f"[cam] !! '{name}' hat keinen offset.rot — übersprungen.", flush=True)
                continue
            convention = str(getattr(offset, "convention", "world")).lower()
            if convention != "world":
                print(f"[cam] !! '{name}' nutzt convention='{convention}' — dieser Fix rechnet "
                      f"nur von 'world' um, übersprungen.", flush=True)
                continue
            quat = np.asarray(offset.rot, dtype=float)            # (w, x, y, z), Welt-Konvention
            rot_usd = _quat_to_matrix(quat) @ self._WORLD_TO_USD  # Spaltenvektor-Konvention
            q_usd = _matrix_to_quat(rot_usd)                      # (w, x, y, z)

            prims = [p for p in stage.Traverse()
                     if p.GetName() == name and p.IsA(UsdGeom.Camera)]
            if not prims:
                print(f"[cam] !! Kein Camera-Prim namens '{name}' in der Stage.", flush=True)
                continue
            for prim in prims:
                try:
                    xf = UsdGeom.Xformable(prim)
                    # Die lokale Translation erhalten und den Op-Stack neu und eindeutig
                    # aufbauen — sonst konkurriert ein evtl. vorhandener rotateXYZ-Op.
                    local = xf.GetLocalTransformation(Usd.TimeCode.Default())
                    t = local.ExtractTranslation()
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(t[0], t[1], t[2]))
                    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                        Gf.Quatd(float(q_usd[0]),
                                 Gf.Vec3d(float(q_usd[1]), float(q_usd[2]), float(q_usd[3]))))
                    fixed += 1
                except Exception as e:  # noqa: BLE001
                    print(f"[cam] !! '{prim.GetPath()}' nicht beschreibbar ({e})", flush=True)
                    failed += 1
        print(f"[cam] RL_CAM_USD_FIX: {fixed} Kamera-Prims neu ausgerichtet"
              + (f", {failed} fehlgeschlagen" if failed else "")
              + ". Kontrolle: der USD-Abgleich im Kamera-Dump muss jetzt 0.0° melden.",
              flush=True)

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

        Maßgeblich ist `observation.state`, also was das echte Gelenk erreicht HAT. Die
        kommandierten `action`-Werte gehen stellenweise deutlich weiter (left_index_0 bis
        -1,762 gegen -1,089 erreicht); die hat auch die reale Hand nicht ausgefahren, ein
        Klemmen dort bildet die Hardware ab statt sie zu verfälschen.
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
            # _0 joints (index_0, middle_0). Die USD-Grenze endet hier auf beiden Seiten
            # exakt an der Null (links [-1.571, 0], rechts [0, 1.571]), der echte Datensatz
            # fährt aber ~0,2 rad in die Gegenrichtung darüber hinaus, und rechts index_0
            # zusätzlich über 1,571. Beides wurde bis 2026-08-24 auf 0 bzw. 1,571 geklemmt.
            "left_hand_middle_0_joint": (-1.571, 0.25),   # Datensatz [-1.394, 0.195]
            "left_hand_index_0_joint": (-1.571, 0.32),    # Datensatz [-1.089, 0.267]
            "right_hand_index_0_joint": (-0.25, 1.70),    # Datensatz [-0.199, 1.646]
            "right_hand_middle_0_joint": (-0.25, 1.571),  # Datensatz [-0.178, 1.454]
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

        # Die Startpose der _0-Gelenke liegt außerhalb der ORIGINAL-USD-Grenzen (links
        # +0,169/+0,163, rechts -0,171/-0,142). Isaac prüft `init_state` beim Spawn dagegen und
        # bricht mit ValueError ab, lange bevor diese Weitung greift — deshalb spawnt der
        # Roboter mit gekappten Werten (SPAWN_JOINT_POS). Jetzt, wo die Grenzen weit genug sind,
        # kommen die echten Datensatzwerte zurück; `_reset_idx` liest genau dieses
        # `default_joint_pos`, der gekappte Zustand lebt also nur bis zum ersten Reset.
        dataset_pos = dict(zip(ALL_JOINTS_ORDERED, DATASET_INIT_STATE))
        restored = [n for n in names if n in dataset_pos]
        for name in restored:
            self.robot.data.default_joint_pos[:, joint_ids[names.index(name)]] = float(
                dataset_pos[name])

        print(f"[Env] Dex3-Finger-Gelenkgrenzen an Dataset-Range geweitet "
              f"({len(joint_ids)} Gelenke, {len(restored)} Startwerte zurückgesetzt).",
              flush=True)

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

    # Hinweis zur Handfarbe (die reale DEX3 ist schwarz, das URDF-Asset weiß): das gehört
    # NICHT hierher. Ein Laufzeit-Recolor über die Material-Bindung erreicht die Hand-Meshes
    # nicht — sie liegen in USD-Prototypen (die /visuals-Prims sind instanceable), und eine
    # Bindung am Instance-Root komponiert nicht hinein. Gemessen in runs/20260808/21:
    # 0 getroffene Materialien. Erledigt wird das offline auf dem Asset, seit Juni:
    #   Simulation/g1_dex3_sim/recolor_hands_black.py  (de-instanziert + bindet je Mesh)
    # Die Launcher setzen ASSET_PATH auf das erzeugte g1_dex3_blackhands.usd.

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

        # env_origins: write_root_pose_to_sim() erwartet WELT-Koordinaten, block_*_range aus
        # der Config ist env-LOKAL. Roboter, Tisch und Stapel-Band brauchen hier nichts —
        # deren Posen kommen aus dem Spawn unter dem Env-Xform und sind bereits korrekt
        # (nachgemessen am 2026-08-08: table welt=(1.5,-1,0.435), robot_root=(1,-1,0.85)
        # bei env_origin=(1,-1,0)). Nur die selbst gesampelten Würfel müssen den Versatz
        # bekommen, weil sie als Weltkoordinaten geschrieben werden.
        env_offset = self.scene.env_origins[env_ids]

        # Roboter in Home-Pose zurücksetzen
        default_pos = self.robot.data.default_joint_pos[env_ids]
        default_vel = torch.zeros_like(default_pos)
        self.robot.set_joint_position_target(default_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)

        # Würfel neu sampeln (randomisierte Positionen auf dem Tisch, env-lokal → + Versatz).
        #
        # Jeder Würfel bekommt ein EIGENES y-Band. Vorher zogen alle drei unabhängig aus
        # demselben Rechteck (x 0.30–0.40, y -0.20–0.20): zwei 5-cm-Würfel überlappen dort
        # mit ~18 % je Paar, bei drei Paaren also in ~44 % aller Resets. PhysX löst die
        # Durchdringung auf, indem es die Würfel auseinanderschießt — in runs/20260808/08
        # lagen sie danach bei z=0.025, also auf dem Boden, 1,5 m vom Tisch entfernt.
        # Mit disjunkten Bändern plus Rand ist Überlappung ausgeschlossen; die x-Achse
        # bleibt voll randomisiert und die y-Gesamtspanne unverändert.
        n_blocks = max(len(self.blocks), 1)
        y_lo, y_hi = self.cfg.block_y_range
        slot = (y_hi - y_lo) / n_blocks
        margin = min(0.03, 0.4 * slot)   # >= 2*margin Abstand zum Nachbarband
        for i, block in enumerate(self.blocks):
            block_pos = torch.zeros(len(env_ids), 3, device=self.device)
            block_pos[:, 0] = sample_uniform(
                self.cfg.block_x_range[0], self.cfg.block_x_range[1],
                (len(env_ids),), device=self.device
            )
            block_pos[:, 1] = sample_uniform(
                y_lo + i * slot + margin, y_lo + (i + 1) * slot - margin,
                (len(env_ids),), device=self.device
            )
            block_pos[:, 2] = self.cfg.block_z_surface
            block_pos += env_offset
            lo, hi = self.cfg.block_yaw_range_deg
            half = torch.deg2rad(sample_uniform(
                float(lo), float(hi), (len(env_ids),), device=self.device)) / 2.0
            block_quat = torch.zeros(len(env_ids), 4, device=self.device)
            block_quat[:, 0] = torch.cos(half)   # Drehung um z: (cos φ/2, 0, 0, sin φ/2)
            block_quat[:, 3] = torch.sin(half)
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
        # USD→Dataset-Konvention: Gelenke mit invertierter Achse negieren, damit das Modell
        # Beobachtungen in derselben Konvention sieht wie die Trainingsdaten. Seit 2026-08-24
        # ist die Liste leer — der Datensatz ist bereits seitenweise in USD-Konvention.
        if self._SIGN_FLIP_IDX:
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

    # Policy-Indices, deren Vorzeichen gespiegelt werden muss: KEINE.
    #
    # Die Annahme war "Datensatz: positiv = schließen, USD: links negativ = schließen, rechts
    # positiv = schließen", also müsse eine Seite gespiegelt werden. Sie ist falsch. Der
    # Datensatz ist bereits SEITENWEISE in der USD-Konvention aufgezeichnet. Die
    # _1-Beugegelenke zeigen es: sie wurden nie gespiegelt und passen trotzdem beide exakt in
    # ihre Grenzen (meta/stats.json, ganzer Datensatz):
    #     left_hand_index_1   [-2.083, -0.008]   Grenze [-2.13,  0.05]
    #     right_hand_index_1  [ 0.010,  2.085]   Grenze [-0.05,  2.14]
    # Die _0-Gelenke folgen derselben Konvention — links negativ, rechts positiv:
    #     left_hand_index_0   [-1.089,  0.267]   Grenze [-1.571, 0.0]
    #     right_hand_index_0  [-0.199,  1.646]   Grenze [ 0.0,   1.571]
    #
    # Eine Spiegelung dreht diese Werte aus ihrer Grenze heraus, wo `set_joint_position_target`
    # sie auf 0 klemmt — das Gelenk bewegt sich dann überhaupt nicht mehr. Genau das geschah
    # bis 2026-08-24 mit BEIDEN Händen ([17, 19, 24, 26]) und danach noch mit der linken
    # ([17, 19]). Gemessen mit `server_rl_run.sh tipcheck`: in Episode 0 klemmten auf den
    # linken _0-Gelenken 550 bzw. 587 von 1173 Frames.
    #
    # Die Liste wirkt in `_get_observations` UND `_pre_physics_step`, also in jedem Lauf: Eval,
    # Grasp, RL, Replay, Co-Training. Die Policy sah verdrehte Beobachtungen und ihre Aktionen
    # wurden aus der Grenze gedreht — das hebt sich nicht auf, das Gelenk stand einfach.
    # Der schmale Überstand über die Nulllinie hinaus (beide Hände fahren ~0,2 rad in die
    # Gegenrichtung) ist echt und wird jetzt von `_widen_finger_joint_limits` abgedeckt,
    # nicht mehr weggespiegelt.
    _SIGN_FLIP_IDX: list[int] = []

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

        # Dataset-Konvention → USD-Konvention. Seit 2026-08-24 leer, siehe _SIGN_FLIP_IDX.
        actions = actions.clone()
        if self._SIGN_FLIP_IDX:
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

    # Messkörper für die Reach-Diagnose, in absteigender Aussagekraft. Die Fingerspitzen
    # sind die Flächen, die den Würfel tatsächlich berühren — nur bei ihnen heißt ein
    # kleiner Abstand auch "am Würfel". Die Fallbacks greifen, falls die USD-Konvertierung
    # andere Namen behalten hat; welcher Satz benutzt wurde, landet in results.json, damit
    # die Zahl nie ohne ihren Bezugsrahmen gelesen wird.
    # Der Frame eines distalen Fingerglieds sitzt IM Gelenk, nicht an der Kuppe: die
    # Kollisionsmesh reicht von dort noch 5,2 cm weiter (STL-Bounding-Box aus
    # dexterous_hand_description/dex3_1, +x bei Zeige-/Mittelfinger, ∓y beim Daumen).
    # Ohne diesen Versatz misst man das distale Gelenk und liegt um mehr als eine ganze
    # Würfelkante daneben — und schlimmer: das Beugen dieses Gelenks DREHT den Frame nur,
    # verschiebt seinen Ursprung also nicht. Der eigentliche Griff wäre unsichtbar.
    _TIP_LEN: float = 0.052
    _REACH_BODY_SETS: tuple[tuple[str, list[tuple[str, tuple[float, float, float]]]], ...] = (
        ("fingertip", [
            ("left_hand_index_1_link", (_TIP_LEN, 0.0, 0.0)),
            ("left_hand_middle_1_link", (_TIP_LEN, 0.0, 0.0)),
            ("left_hand_thumb_2_link", (0.0, -_TIP_LEN, 0.0)),
            ("right_hand_index_1_link", (_TIP_LEN, 0.0, 0.0)),
            ("right_hand_middle_1_link", (_TIP_LEN, 0.0, 0.0)),
            ("right_hand_thumb_2_link", (0.0, _TIP_LEN, 0.0)),
        ]),
        ("palm", [("left_hand_palm_link", (0.0, 0.0, 0.0)),
                  ("right_hand_palm_link", (0.0, 0.0, 0.0))]),
        ("wrist", [("left_wrist_yaw_link", (0.0, 0.0, 0.0)),
                   ("right_wrist_yaw_link", (0.0, 0.0, 0.0))]),
    )

    def get_contact_points_w(self) -> torch.Tensor:
        """Weltpositionen der Kontaktflächen (Fingerkuppen), (num_envs, n, 3).

        Gemeinsame Quelle für Eval, Replay und jede Greif-Diagnose, damit alle drei
        denselben Bezugspunkt messen. Der lokale Versatz wird mit der Körper-Orientierung
        mitgedreht — sonst zeigte er beim gebeugten Finger in die falsche Richtung.
        """
        if self._reach_body_ids is None:
            for tag, entries in self._REACH_BODY_SETS:
                names = [n for n, _ in entries]
                try:
                    ids, _ = self.robot.find_bodies(names, preserve_order=True)
                except ValueError:
                    continue
                if len(ids) == len(names):
                    self._reach_body_ids = ids
                    self._reach_frame = tag
                    self._reach_offsets = torch.tensor(
                        [o for _, o in entries], device=self.device, dtype=torch.float32
                    )
                    break
            if self._reach_body_ids is None:
                raise RuntimeError(
                    "Reach-Diagnose: keiner der bekannten Messkörper im Asset gefunden "
                    f"(geprüft: {[t for t, _ in self._REACH_BODY_SETS]}). "
                    "Ohne Bezugskörper wäre die Zahl nicht interpretierbar."
                )
            print(f"[Reach-Diagnose] Messkörper: {self._reach_frame} "
                  f"(Versatz bis {float(self._reach_offsets.abs().max()) * 100:.1f} cm)",
                  flush=True)

        pos = self.robot.data.body_pos_w[:, self._reach_body_ids, :]
        quat = self.robot.data.body_quat_w[:, self._reach_body_ids, :]
        return pos + quat_apply(quat, self._reach_offsets.expand_as(pos))

    def get_reach_diagnostics(self) -> tuple[float, np.ndarray, str]:
        """Kleinster Kontakt-Würfel-Abstand [m], Würfel-Weltpositionen (3, 3), Messkörper.

        Diagnose für die Eval: die binäre Erfolgsrate sagt bei 0/20 nicht, WORAN es lag.
        Der Abstand trennt genau die zwei Fälle, die man sonst nur am Video auseinanderhält
        — "Hand kommt nie an einen Würfel" (Geometrie/Reichweite, z. B. der offene
        Tischabstand) gegen "Hand steht am Würfel, greift aber nicht" (Wahrnehmung/Politik).
        Nur die zweite Lesart rechtfertigt TUNE_VISUAL=1.

        Gemessen wird ab den Fingerkuppen, NICHT ab der Handwurzel wie im Shaped-Reward.
        Zwischen left_wrist_yaw_link und der Kuppe liegen laut URDF rund 22 cm
        (0.0415 Handwurzel→Handfläche + 0.0777 →Fingergrund + 0.0458 →distales Gelenk
        + 0.052 →Kuppe). Ein Handwurzel-Abstand von 15 cm ist deshalb mehrdeutig: er kann
        "Würfel liegt in der Greiföffnung" ebenso bedeuten wie "Würfel 15 cm daneben".
        Genau diese Unterscheidung soll die Diagnose treffen, also darf sie nicht am
        falschen Ende der Hand messen — auch nicht am distalen Gelenk, das noch 5,2 cm
        vor der Kuppe sitzt (Lauf 25/28 haben genau daran zu viel gemessen).

        Nutzt dieselben Tensoren wie _shaped_reward — kein zusätzlicher Sensor, kein
        Render-Pass. Muss VOR env.step() gelesen werden: DirectRLEnv setzt bei done im
        selben Step zurück, danach stünde hier schon das Layout der nächsten Episode.
        """
        contact_pos = self.get_contact_points_w()
        block_pos = torch.stack([b.data.root_pos_w for b in self.blocks], dim=1)
        dists = torch.cdist(contact_pos, block_pos)        # (num_envs, n_contact, 3)
        return float(dists[0].amin().item()), block_pos[0].cpu().numpy(), self._reach_frame

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
        terminated = self._check_success() if self.cfg.terminate_on_success else torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
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

    def force_policy_camera_render(self) -> dict[str, Any]:
        """Erzeugt genau vor einer Policy-Anfrage einen frischen Kamera-Frame.

        Bei CAMERA_RENDER_EVERY_N=1 bleibt der historische DirectRLEnv-Pfad aktiv und
        diese Methode ist ein No-op. Im synchronen Modus ist sie die einzige Stelle,
        die den RTX-Renderer zwischen zwei Action-Chunks explizit aufruft.
        """
        if not hasattr(self, "_policy_render_generation"):
            self._policy_render_generation = 0
        if self.cfg.camera_render_every_n > 1:
            self.sim.render()
            # ``SimulationContext.render()`` erzeugt den RTX-Frame, aktualisiert aber
            # bei Isaac-Lab 3.0 nicht zwingend die Python-seitigen TiledCamera-Puffer.
            # Normalerweise erledigt ``scene.update()`` das innerhalb von ``env.step()``;
            # der Schnellpfad hat diese automatische Aktualisierung absichtlich
            # unterdrückt. Ohne dieses Update kann die Policy trotz neuem Render einen
            # alten RGB-Tensor erhalten.
            #
            # ``0.0`` ist hier absichtlich: Die Simulationszeit ist bereits durch den
            # vorangehenden Action-Chunk fortgeschritten. Es soll nur der aktuelle
            # Render-Produkt-Puffer eingelesen, keine zusätzliche Sensorzeit erfunden
            # werden. Bei update_period=0 (Default der Kameras) ist jedes Update fällig.
            for camera in self.cameras.values():
                camera.update(0.0)
            self._policy_render_generation += 1

        sensor_tokens = {}
        for name, camera in self.cameras.items():
            token = None
            # Isaac-Lab-Versionen exponieren den letzten Sensor-Zeitpunkt unter
            # unterschiedlichen (teilweise privaten) Namen. Rein diagnostisch lesen.
            for owner in (camera, getattr(camera, "data", None)):
                if owner is None:
                    continue
                # Nur echte Zeitstempel vergleichen. Ein generisches ``frame``-Attribut
                # kann bei manchen Camera-Implementierungen statische Kalibrierungsdaten
                # enthalten und wuerde dann faelschlich einen veralteten Frame melden.
                for attr in ("_timestamp_last_update", "timestamp"):
                    if hasattr(owner, attr):
                        value = getattr(owner, attr)
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().reshape(-1).tolist()
                        elif isinstance(value, np.ndarray):
                            value = value.reshape(-1).tolist()
                        token = value
                        break
                if token is not None:
                    break
            sensor_tokens[name] = token
        return {
            "generation": int(self._policy_render_generation),
            "sensor_tokens": sensor_tokens,
        }

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
