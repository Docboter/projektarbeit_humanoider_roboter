"""
Closed-Loop-Eval-Runner für GR00T N1.6 auf G1 + Dex3 in Isaac Lab.

Startet den Isaac-Lab-Sim-Client, verbindet sich mit dem GR00T-Policy-Server
(der in einem separaten Container läuft) und führt N Episoden durch.

Verwendung (innerhalb des Sim-Containers):
    ${ISAACLAB_PATH}/isaaclab.sh -p g1_dex3_sim/run_g1_dex3_sim_eval.py \\
        --headless \\
        --server tcp://localhost:5555 \\
        --num-episodes 20 \\
        --execution-horizon 8 \\
        --video-dir /data/sim_videos \\
        --results-file /data/sim_results.json

Architektur:
    GR00T-Policy-Server (Container 1) ←── ZMQ :5555 ──→ dieser Runner (Container 2)

Phasen-Abnahme (ISAAC_LAB_SIM_PLAN.md):
    Phase D: get_action(dummy_obs) liefert (16, 28) ✓
    Phase E: Roboter bewegt sich plausibel auf Policy-Aktionen ✓
    Phase F: Success-Rate + Rollout-Video ✓
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Isaac Lab App-Setup MUSS vor allen anderen Imports stehen
from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI-Argumente
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="GR00T G1-Dex3 Closed-Loop Sim Eval")
parser.add_argument(
    "--server", type=str, default="tcp://localhost:5555",
    help="ZMQ-Adresse des GR00T-Policy-Servers"
)
parser.add_argument(
    "--num-episodes", type=int, default=20,
    help="Anzahl der Eval-Episoden"
)
parser.add_argument(
    "--execution-horizon", type=int, default=8,
    help="Wie viele Steps eines 16-Step-Chunks ausgeführt werden, bevor neu geplant wird"
)
parser.add_argument(
    "--task-description", type=str, default="stack the blocks",
    help="Language-Prompt für die Policy"
)
parser.add_argument(
    "--video-dir", type=str, default="/data/sim_videos",
    help="Verzeichnis für Rollout-Videos (leer = keine Videos)"
)
parser.add_argument(
    "--results-file", type=str, default="/data/sim_results.json",
    help="JSON-Datei für Eval-Ergebnisse"
)
parser.add_argument(
    "--ping-retries", type=int, default=10,
    help="Wie oft auf den Server gewartet wird (je 5 s)"
)
parser.add_argument(
    "--asset-path", type=str, default="",
    help="Pfad zum G1+Dex3 USD-Asset (überschreibt den Default in g1_dex3_cfg.py; "
         "leer = cfg-Default verwenden)"
)
parser.add_argument(
    "--episode-length-s", type=float, default=0.0,
    help="Zeitbudget je Episode in Sekunden (0 = cfg-Default, aktuell 300 s = 9000 Steps). "
         "Bestimmt die Laufzeit unmittelbar: Steps = Wert * policy_hz, und jeder Step "
         "rendert 4 Kameras. Anker für eine sinnvolle Wahl ist die menschliche "
         "Teleop-Demo: replay_episode0.npz hat 1173 Steps = 39 s bei 30 Hz."
)

# Isaac-Lab-eigene Argumente hinzufügen und parsen
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Isaac-Lab-Anwendung starten (MUSS vor jedem omni.*/isaacsim.*-Import passieren)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (nach AppLauncher-Start)
# ---------------------------------------------------------------------------

import numpy as np
import torch

from client import PolicyClient, build_obs
from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg
from g1_dex3_cfg import CAMERA_CFG, G1_DEX3_CFG

# ---------------------------------------------------------------------------
# Video-Aufzeichnung
# ---------------------------------------------------------------------------

def save_episode_video(frames: list[np.ndarray], episode: int, video_dir: str) -> None:
    """Speichert eine Liste von RGB-Frames als MP4."""
    if not frames:
        return
    try:
        import imageio
        os.makedirs(video_dir, exist_ok=True)
        path = os.path.join(video_dir, f"episode_{episode:04d}.mp4")
        imageio.mimwrite(path, frames, fps=30, quality=8)
        print(f"[Video] Gespeichert: {path}")
    except Exception as e:
        print(f"[Video] Fehler beim Speichern: {e}")


# ---------------------------------------------------------------------------
# Einzelne Episode
# ---------------------------------------------------------------------------

def run_episode(
    env: G1Dex3BlockstackEnv,
    client: PolicyClient,
    execution_horizon: int,
    task_description: str,
    record_video: bool,
    episode: int,
    video_dir: str | None,
) -> dict:
    """
    Führt eine einzelne Eval-Episode durch.

    Returns:
        dict mit Feldern: success, num_steps, duration_s, success_step,
        min_reach_m, min_reach_step, block_shift_max_m
    """
    obs_dict, _ = env.reset()
    client.reset()

    frames: list[np.ndarray] = []
    step = 0
    success = False
    success_step = -1
    t_start = time.perf_counter()

    # Reichweiten-Diagnose (s. env.get_reach_diagnostics): hält fest, ob die Hand einem
    # Würfel überhaupt nahekam und ob sich die Würfel bewegt haben. Ohne das ist ein 0/20
    # nicht interpretierbar — "nie in Reichweite" und "in Reichweite, aber nicht gegriffen"
    # führen zu völlig verschiedenen nächsten Schritten.
    reach_start, block_pos_start = env.get_reach_diagnostics()
    min_reach = reach_start
    min_reach_step = 0
    block_pos_last = block_pos_start

    max_steps = int(env.cfg.episode_length_s * env.cfg.policy_hz)

    while step < max_steps:
        # Observation fürs Modell aufbauen
        obs_np = env.get_obs_for_policy()
        policy_obs = build_obs(
            cam_left_high=obs_np["video.cam_left_high"],
            cam_right_high=obs_np["video.cam_right_high"],
            cam_left_wrist=obs_np["video.cam_left_wrist"],
            cam_right_wrist=obs_np["video.cam_right_wrist"],
            joint_pos=obs_np["state.joint_pos"],
            task_description=task_description,
        )

        # Debug (einmalig, Episode 1 / Step 0): erste Obs ALLER 4 Kameras dumpen, um die
        # Kamera-Posen gegen Simulation/camera_reference/ zu verifizieren (v. a. Wrist-Cams).
        if episode == 1 and step == 0 and video_dir:
            import imageio
            os.makedirs(video_dir, exist_ok=True)
            for _ck in ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"):
                _arr = np.asarray(obs_np[f"video.{_ck}"])
                while _arr.ndim > 3:
                    _arr = _arr[0]
                imageio.imwrite(os.path.join(video_dir, f"_debug_obs_{_ck}.png"),
                                _arr.astype(np.uint8))
            print("[Debug] Erste Obs aller 4 Kameras gespeichert: _debug_obs_*.png", flush=True)

        # Action-Chunk vom GR00T-Server holen (16, 28)
        try:
            chunk = client.get_action(policy_obs)  # np.ndarray (16, 28)
        except Exception as e:
            print(f"[Episode] Fehler bei get_action (Step {step}): {e}")
            break

        # Chunk schrittweise ausführen
        for t in range(min(execution_horizon, len(chunk))):
            action_t = torch.tensor(
                chunk[t], dtype=torch.float32, device=env.device
            ).unsqueeze(0)  # (1, 28)

            # VOR dem Step lesen — nach einem done hat DirectRLEnv bereits zurückgesetzt
            # und die Werte gehörten zur nächsten Episode.
            reach_now, block_pos_last = env.get_reach_diagnostics()
            if reach_now < min_reach:
                min_reach = reach_now
                min_reach_step = step

            obs_step, _, terminated, time_out, info = env.step(action_t)

            # Video-Frame nach dem Step erfassen — aus der Szenen-Übersichtskamera (ganze Szene
            # von schräg oben), NICHT der Policy-Kamera. Fallback auf cam_left_high, falls cam_scene
            # (noch) nicht vorhanden ist.
            if record_video:
                cam_key = "video.cam_scene" if "video.cam_scene" in obs_step else "video.cam_left_high"
                frame = obs_step[cam_key][0].cpu().numpy().astype(np.uint8)
                frames.append(frame)

            step += 1

            # Fortschritt sichtbar machen — run_episode printet sonst bis Episodenende nichts,
            # was bei langsamem 4-Kamera-Rendering wie ein Hang aussieht. flush=True für Live-Output.
            if step % 25 == 0:
                print(
                    f"    … Step {step}/{max_steps} ({time.perf_counter() - t_start:.0f}s)",
                    flush=True,
                )

            # Erfolg über das zurückgegebene `terminated` erkennen (robust gegen
            # DirectRLEnv-Auto-Reset, der env.episode_success im selben Step löschen kann).
            # terminated == _check_success() → Würfel erfolgreich gestapelt.
            if terminated.any() and not success:
                success = True
                success_step = step
                print(f"[Episode] Erfolg bei Step {step}!")

            if terminated.any() or time_out.any():
                break

        if terminated.any() or time_out.any():
            break

    duration = time.perf_counter() - t_start

    # Video der Episode speichern (Frames wurden während des Rollouts gesammelt).
    if record_video and video_dir:
        save_episode_video(frames, episode, video_dir)

    # Größte Verschiebung eines Würfels gegenüber dem Reset-Layout. Bleibt sie im
    # Millimeterbereich, wurde in der ganzen Episode kein Würfel angefasst — dann ist
    # die Erfolgsrate ohnehin nur die Bestätigung dieser Beobachtung.
    block_shift = np.linalg.norm(block_pos_last - block_pos_start, axis=1)

    return {
        "success": success,
        "num_steps": step,
        "duration_s": round(duration, 2),
        "success_step": success_step,
        "min_reach_m": round(min_reach, 4),
        "min_reach_step": min_reach_step,
        "block_shift_max_m": round(float(block_shift.max()), 4),
        "block_shift_m": [round(float(v), 4) for v in block_shift],
    }


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main():
    # Asset-Pfad ggf. in der Cfg überschreiben (leerer Default = cfg-Wert behalten)
    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    cfg.execution_horizon = args.execution_horizon
    cfg.task_description = args.task_description
    cfg.dr_enabled = os.getenv("DR_ENABLED", "1") != "0"
    # Wirkt an beiden Stellen gleichzeitig, weil beide dieselbe cfg lesen: run_episode
    # bildet daraus max_steps, und _get_dones() den time_out. Ein Wert hier kann also
    # nicht auseinanderlaufen mit dem, was die Umgebung selbst als Episodenende sieht.
    if args.episode_length_s > 0:
        cfg.episode_length_s = args.episode_length_s

    print("=" * 60)
    print("GR00T N1.6 — G1+Dex3 Closed-Loop Sim Eval")
    print("=" * 60)
    print(f"  Server:           {args.server}")
    print(f"  Episoden:         {args.num_episodes}")
    print(f"  Exec-Horizon:     {args.execution_horizon}")
    print(f"  Task:             {args.task_description}")
    print(f"  Video-Dir:        {args.video_dir or '(kein Video)'}")
    print(f"  Asset:            {cfg.scene.robot.spawn.usd_path}")
    print(f"  Domain Rand.:     {'AN' if cfg.dr_enabled else 'AUS (DR_ENABLED=0)'}")
    print(
        f"  Episodenlaenge:   {cfg.episode_length_s:.0f}s "
        f"= {int(cfg.episode_length_s * cfg.policy_hz)} Steps"
        f"{'' if args.episode_length_s > 0 else '  (cfg-Default)'}"
    )
    print()

    # Phase A-Check: Isaac Lab startet, Sim läuft
    print("[Phase A] Isaac-Lab-Sim wird initialisiert …")

    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode="rgb_array" if args.video_dir else None)

    print("[Phase A] Sim läuft.")

    # Phase D: ZMQ-Verbindung aufbauen
    print(f"[Phase D] Verbinde mit GR00T-Server: {args.server}")
    client = PolicyClient(server_url=args.server, timeout_ms=20_000)

    if not client.ping(retries=args.ping_retries, delay=5.0):
        print("[FEHLER] Server nicht erreichbar. Abbruch.")
        env.close()
        sys.exit(1)

    print("[Phase D] Server erreichbar — starte Eval-Schleife.")
    print()

    # Dummy-Test (Phase D Abnahme): get_action mit synthetischer Obs
    dummy_rgb = np.zeros((CAMERA_CFG.height, CAMERA_CFG.width, 3), dtype=np.uint8)
    dummy_joint = np.zeros(28, dtype=np.float32)
    dummy_obs = build_obs(dummy_rgb, dummy_rgb, dummy_rgb, dummy_rgb, dummy_joint)
    dummy_chunk = client.get_action(dummy_obs)
    assert dummy_chunk.shape == (16, 28), f"Unerwartete Chunk-Shape: {dummy_chunk.shape}"
    print(f"[Phase D] Dummy-get_action OK: shape={dummy_chunk.shape}")
    print()

    # Eval-Schleife
    results = []
    record = bool(args.video_dir)

    for ep in range(args.num_episodes):
        print(f"--- Episode {ep + 1}/{args.num_episodes} ---")
        ep_result = run_episode(
            env=env,
            client=client,
            execution_horizon=args.execution_horizon,
            task_description=args.task_description,
            record_video=record,
            episode=ep + 1,
            video_dir=args.video_dir,
        )
        ep_result["episode"] = ep + 1
        results.append(ep_result)

        status = "ERFOLG" if ep_result["success"] else "misslungen"
        print(
            f"  Episode {ep + 1}: {status} | "
            f"{ep_result['num_steps']} Steps | "
            f"{ep_result['duration_s']:.1f}s | "
            f"min. Hand-Würfel-Abstand {ep_result['min_reach_m'] * 100:.1f} cm "
            f"(Step {ep_result['min_reach_step']}) | "
            f"Würfel verschoben max. {ep_result['block_shift_max_m'] * 100:.1f} cm"
        )

    # Auswertung
    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / len(results) if results else 0.0
    print()
    print("=" * 60)
    print(f"Ergebnis: {n_success}/{len(results)} Erfolge — Success-Rate: {success_rate:.1%}")
    # Bei 0 Erfolgen ist das die eigentliche Information: kam die Hand nie an einen Würfel
    # (Geometrie), oder stand sie daneben ohne zu greifen (Wahrnehmung/Politik)?
    if results:
        reaches = sorted(r["min_reach_m"] for r in results)
        shifts = [r["block_shift_max_m"] for r in results]
        n_touched = sum(1 for s in shifts if s > 0.01)
        print(
            f"  Min. Hand-Würfel-Abstand: bester {reaches[0] * 100:.1f} cm, "
            f"Median {reaches[len(reaches) // 2] * 100:.1f} cm"
        )
        print(f"  Episoden mit bewegtem Würfel (>1 cm): {n_touched}/{len(results)}")
    print("=" * 60)

    # Ergebnisse speichern
    summary = {
        "num_episodes": len(results),
        "num_success": n_success,
        "success_rate": success_rate,
        "server": args.server,
        "task_description": args.task_description,
        "execution_horizon": args.execution_horizon,
        "episodes": results,
    }
    summary["episode_length_s"] = cfg.episode_length_s
    summary["max_steps_per_episode"] = int(cfg.episode_length_s * cfg.policy_hz)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Ergebnisse gespeichert: {args.results_file}")
    # Erfolgsmarker wie in dump_camera_poses.py / measure_domain_gap.py: isaaclab.sh gibt
    # den Exit-Code des Pythons NICHT durch (ein Traceback endet trotzdem mit 0), ein
    # Abbruch mitten in der Eval-Schleife bliebe sonst unbemerkt.
    print("[eval] fertig.", flush=True)

    # Aufräumen
    client.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
