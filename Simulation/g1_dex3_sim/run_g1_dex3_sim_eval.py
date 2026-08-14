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
    "--checkpoint-path", type=str, default=os.environ.get("CHECKPOINT_PATH", ""),
    help="Checkpoint-Verzeichnis, das der Policy-Server geladen hat. Wird NICHT geladen, "
         "sondern nur für die Ergebnis-Datei ausgelesen (s. checkpoint_fingerprint). "
         "Leer = die Herkunft des Laufs bleibt unbelegt."
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
# Checkpoint-Herkunft
# ---------------------------------------------------------------------------

def checkpoint_fingerprint(path: str) -> dict:
    """Belegt, WELCHER Checkpoint diesen Lauf erzeugt hat.

    Der Pfad allein genügt dafür nicht: liegen mehrere Checkpoints unter
    /data/checkpoints und wurde CHECKPOINT_PATH beim Start vergessen, greift der
    Default — der Lauf misst dann still ein anderes Modell als gemeint (genau das
    verhindert `ensure_checkpoint` gerade NICHT, es überspringt bei existierendem
    Verzeichnis nur den Download). Deshalb wandern zusätzlich `run_id` aus
    wandb_config.json, `global_step` aus trainer_state.json und die Gewichtsgrößen
    mit: drei Merkmale, die zwei Trainingsläufe zuverlässig auseinanderhalten.

    Fehlschläge sind hier nie fatal — eine fehlende Datei darf keine 20 Episoden
    kosten, sie kostet nur den jeweiligen Eintrag.
    """
    info: dict = {"path": path}
    if not path:
        info["note"] = "nicht übergeben (--checkpoint-path / CHECKPOINT_PATH leer)"
        return info
    p = Path(path)
    if not p.is_dir():
        info["note"] = "Verzeichnis nicht lesbar"
        return info
    try:
        info["run_id"] = json.loads((p / "wandb_config.json").read_text()).get("run_id")
    except Exception as e:
        info["run_id_error"] = str(e)
    try:
        info["global_step"] = json.loads(
            (p / "trainer_state.json").read_text()
        ).get("global_step")
    except Exception as e:
        info["global_step_error"] = str(e)
    try:
        info["weight_bytes"] = {f.name: f.stat().st_size for f in sorted(p.glob("*.safetensors"))}
    except Exception as e:
        info["weight_bytes_error"] = str(e)
    return info


# ---------------------------------------------------------------------------
# Gestufte Meilensteine
# ---------------------------------------------------------------------------
# Der binäre Stapel-Erfolg steht bei 0/20 und bewegt sich nicht. Eine Maßnahme
# (TUNE_VISUAL, anderer Checkpoint, Kontaktparameter) lässt sich dagegen nicht
# bewerten: 0/20 vorher, 0/20 nachher, kein Erkenntnisgewinn. Diese Leiter macht
# den Fortschritt VOR dem Stapeln sichtbar. Jede Schwelle mit Begründung, damit
# sie nicht später zurechtgebogen wird.

# Fingerkuppe→Würfelmitte. 4 cm ist derselbe Wert, mit dem der Replay-Greiftest
# "Würfel noch in der Hand" prüft (run_g1_dex3_replay.py), also derselbe Maßstab.
REACH_THRESHOLD_M = 0.04
# Würfel-Verschiebung nach der Karenzzeit. 1 cm liegt klar über dem Solver-Rauschen
# von ~2 mm, das nach dem Einschwingen übrig bleibt.
TOUCH_THRESHOLD_M = 0.01
# Anhebung. Schwelle aus der vorregistrierten Regel in rl-anleitung.md ("hebt (a)
# den Würfel, max_cube_lift_cm > ~2"). Zum Vergleich: Lauf 29 erreichte 7,9 cm.
LIFT_THRESHOLD_M = 0.02
# Fingerspanne des Kommandos. Referenz ist die menschliche Demonstration aus
# replay_episode0.npz: 2,094 rad über die Episode (Lauf 29). 30 % davon trennt
# "hat einen Griff versucht" von dem 0,39-rad-Zucken aus Lauf 30 (19 %).
DEMO_FINGER_SPAN_RAD = 2.094
GRASP_ATTEMPT_FRACTION = 0.30


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
        dict mit Feldern: success, num_steps, duration_s, success_step, reach_frame,
        reach_start_m, min_reach_m, min_reach_step, block_shift_max_m, block_shift_m,
        finger_span_max_rad, max_cube_lift_cm, milestones
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
    reach_start, block_pos_start, reach_frame = env.get_reach_diagnostics()
    min_reach = reach_start
    min_reach_step = 0
    block_pos_last = block_pos_start

    # Karenzzeit für die Würfel-Verschiebung: nach dem Reset setzt der Kontakt-Solver die
    # Würfel noch ein Stück zurecht — in Lauf 24 rund 2 cm, und zwar bei ALLEN dreien,
    # obwohl keine Hand in ihrer Nähe war. Ohne diese Karenz meldet die Diagnose in jeder
    # Episode "Würfel bewegt" und misst damit den Solver statt den Roboter.
    settle_steps = 30   # 1 s bei 30 Hz

    # Fingerspur: versucht die Politik überhaupt zu greifen? Die Dims 14:28 des
    # Action-Vektors sind die 14 Dex3-Fingergelenke (s. ALL_JOINTS_ORDERED in
    # g1_dex3_cfg.py). Bleibt ihre Spannweite über die Episode nahe 0, hält die Politik
    # die Finger starr — dann scheitert der Griff nicht an der Kontaktphysik, sondern
    # daran, dass nie einer versucht wurde. Das trennt die beiden Erklärungen, die ein
    # "Fingerspitzen am Würfel, aber kein Stapel" sonst offenlässt.
    hand_cmd_min = np.full(14, np.inf)
    hand_cmd_max = np.full(14, -np.inf)

    # Anhebung: die fehlende Zwischenstufe zwischen "Würfel berührt" und "gestapelt".
    # Ohne sie liefert jede Maßnahme 0/20 zurück, ohne zu sagen, ob sie in die richtige
    # Richtung ging — gegen eine Metrik, die immer 0 ist, lässt sich nichts optimieren.
    # Bezug ist das EINGESCHWUNGENE Layout (nach settle_steps), nicht der Reset: der
    # Kontakt-Solver setzt die Würfel direkt nach dem Reset noch zurecht, das zählte
    # sonst als Anhebung (derselbe Fehler, den die Karenzzeit bei block_shift behebt).
    max_cube_lift = 0.0

    max_steps = int(env.cfg.episode_length_s * env.cfg.policy_hz)

    # Zeitbudget je Anteil. Ohne diese Aufteilung ist "die Sim ist langsam" nicht
    # handhabbar: die beiden Anteile führen zu GEGENSÄTZLICHEN Maßnahmen —
    #   Inferenz dominiert  -> EXECUTION_HORIZON hoch (weniger Aufrufe je Sim-Sekunde),
    #                          bzw. Policy-Server auf eine eigene GPU;
    #   Rendering dominiert -> weniger/kleinere Kameras, Livestream aus.
    # Die Messung kostet nur perf_counter-Aufrufe und läuft deshalb immer mit.
    t_obs = t_infer = t_env = 0.0
    # Inferenz zusätzlich ABSOLUT je Aufruf, nicht nur als Anteil. Grund: der Anteil ist eine
    # reine Sim-Größe (er misst vor allem, wie teuer das Kamera-Rendering ist, das es auf
    # echter Hardware gar nicht gibt). Für den Weg auf einen realen Roboter zählt die
    # Latenz je Aufruf gegen die Zeit, die ein Chunk abdeckt (16 Schritte / 30 Hz = 533 ms) —
    # und dort ist der SCHLECHTESTE Aufruf die relevante Zahl, nicht der Mittelwert: ein
    # einzelner Ausreißer über dem Budget ist eine Lücke in der Aktionsfolge.
    n_infer = 0
    t_infer_max = 0.0

    while step < max_steps:
        # Observation fürs Modell aufbauen
        _t0 = time.perf_counter()
        obs_np = env.get_obs_for_policy()
        policy_obs = build_obs(
            cam_left_high=obs_np["video.cam_left_high"],
            cam_right_high=obs_np["video.cam_right_high"],
            cam_left_wrist=obs_np["video.cam_left_wrist"],
            cam_right_wrist=obs_np["video.cam_right_wrist"],
            joint_pos=obs_np["state.joint_pos"],
            task_description=task_description,
        )
        t_obs += time.perf_counter() - _t0

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
        _t0 = time.perf_counter()
        try:
            chunk = client.get_action(policy_obs)  # np.ndarray (16, 28)
        except Exception as e:
            print(f"[Episode] Fehler bei get_action (Step {step}): {e}")
            break
        _dt_infer = time.perf_counter() - _t0
        t_infer += _dt_infer
        n_infer += 1
        t_infer_max = max(t_infer_max, _dt_infer)

        # Chunk schrittweise ausführen
        for t in range(min(execution_horizon, len(chunk))):
            action_t = torch.tensor(
                chunk[t], dtype=torch.float32, device=env.device
            ).unsqueeze(0)  # (1, 28)

            # VOR dem Step lesen — nach einem done hat DirectRLEnv bereits zurückgesetzt
            # und die Werte gehörten zur nächsten Episode.
            reach_now, block_pos_last, _ = env.get_reach_diagnostics()
            if reach_now < min_reach:
                min_reach = reach_now
                min_reach_step = step
            if step < settle_steps:
                block_pos_start = block_pos_last
            else:
                # z-Differenz gegen das eingeschwungene Layout, größter Wert über alle Würfel
                lift = float((block_pos_last[:, 2] - block_pos_start[:, 2]).max())
                max_cube_lift = max(max_cube_lift, lift)

            hand_cmd = np.asarray(chunk[t][14:28], dtype=float)
            np.minimum(hand_cmd_min, hand_cmd, out=hand_cmd_min)
            np.maximum(hand_cmd_max, hand_cmd, out=hand_cmd_max)

            _t0 = time.perf_counter()
            obs_step, _, terminated, time_out, info = env.step(action_t)
            t_env += time.perf_counter() - _t0

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
                _el = time.perf_counter() - t_start
                # Echtzeit-Faktor: Sim-Sekunden (step / policy_hz) gegen Wanduhr. 1.0 = Echtzeit,
                # 10 = zehnmal langsamer. Dazu die Aufteilung, WOHIN die Zeit geht.
                _sim_s = step / env.cfg.policy_hz
                _rt = _el / max(_sim_s, 1e-6)
                _ms = 1000 * t_infer / max(n_infer, 1)
                print(
                    f"    … Step {step}/{max_steps} ({_el:.0f}s, "
                    f"{step / max(_el, 1e-6):.1f} Steps/s, {_rt:.1f}x Echtzeit) | "
                    f"Obs {t_obs / _el:.0%} · Inferenz {t_infer / _el:.0%} · "
                    f"Sim+Render {t_env / _el:.0%} | Inferenz {_ms:.0f} ms/Aufruf "
                    f"(max {1000 * t_infer_max:.0f}), Budget "
                    f"{1000 * execution_horizon / env.cfg.policy_hz:.0f} ms",
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

    # Größte Verschiebung eines Würfels gegenüber dem eingeschwungenen Layout. Bleibt sie
    # im Millimeterbereich, wurde in der ganzen Episode kein Würfel angefasst — dann ist
    # die Erfolgsrate ohnehin nur die Bestätigung dieser Beobachtung.
    block_shift = np.linalg.norm(block_pos_last - block_pos_start, axis=1)

    # Bei einem Abbruch vor dem ersten Chunk stehen hier noch die inf-Startwerte.
    finger_span = (
        hand_cmd_max - hand_cmd_min if np.isfinite(hand_cmd_min).all() else np.zeros(14)
    )

    return {
        "success": success,
        "num_steps": step,
        "duration_s": round(duration, 2),
        # Durchsatz + Zeitaufteilung, damit ein "die Sim war zäh" später belegbar ist
        # statt erinnert. steps_per_s gegen policy_hz (30) gelesen = Echtzeit-Faktor.
        "steps_per_s": round(step / duration, 2) if duration > 0 else 0.0,
        "time_share": {
            "obs": round(t_obs / duration, 3) if duration > 0 else 0.0,
            "inference": round(t_infer / duration, 3) if duration > 0 else 0.0,
            "sim_render": round(t_env / duration, 3) if duration > 0 else 0.0,
        },
        # Die einzige hier gemessene Zahl, die auch für echte Hardware gilt: dort entfällt
        # das Rendering ersatzlos, die Policy-Latenz bleibt. Vergleichsmaßstab ist
        # budget_ms = wie lange die ausgeführten Schritte eines Chunks dauern.
        "inference_ms": {
            "mean": round(1000 * t_infer / n_infer, 1) if n_infer else 0.0,
            "max": round(1000 * t_infer_max, 1),
            "calls": n_infer,
            "budget_ms": round(1000 * execution_horizon / env.cfg.policy_hz, 1),
        },
        "success_step": success_step,
        "reach_frame": reach_frame,
        # Abstand zum Zeitpunkt 0 als Bezugsgröße: erst die Differenz zu min_reach_m zeigt,
        # ob die Politik den Abstand überhaupt verkleinert hat oder ob der beste Wert der
        # Episode einfach die Ausgangspose war.
        "reach_start_m": round(reach_start, 4),
        "min_reach_m": round(min_reach, 4),
        "min_reach_step": min_reach_step,
        "block_shift_max_m": round(float(block_shift.max()), 4),
        "block_shift_m": [round(float(v), 4) for v in block_shift],
        "finger_span_max_rad": round(float(finger_span.max()), 4),
        "max_cube_lift_cm": round(max_cube_lift * 100, 2),
        # Gestufte Meilensteine: jede Stufe für sich auswertbar, damit ein 0/20 beim
        # Stapeln nicht mehr die einzige Information der Episode ist.
        "milestones": {
            "reached": bool(min_reach <= REACH_THRESHOLD_M),
            "grasp_attempted": bool(
                finger_span.max() >= GRASP_ATTEMPT_FRACTION * DEMO_FINGER_SPAN_RAD
            ),
            "touched": bool(block_shift.max() > TOUCH_THRESHOLD_M),
            "lifted": bool(max_cube_lift > LIFT_THRESHOLD_M),
            "stacked": bool(success),
        },
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
    # Zuerst, nicht zuletzt: bei einem Abbruch mitten im Lauf steht die Herkunft dann
    # trotzdem im Log, auch wenn es nie zur Ergebnis-Datei kommt.
    ckpt = checkpoint_fingerprint(args.checkpoint_path)
    print(f"  Checkpoint:       {ckpt.get('path') or '(unbekannt)'}")
    print(
        f"                    run_id={ckpt.get('run_id', '?')}  "
        f"global_step={ckpt.get('global_step', '?')}"
    )
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
            f"{ep_result['reach_frame']}-Würfel-Abstand "
            f"{ep_result['reach_start_m'] * 100:.1f} → {ep_result['min_reach_m'] * 100:.1f} cm "
            f"(Step {ep_result['min_reach_step']}) | "
            f"Würfel verschoben max. {ep_result['block_shift_max_m'] * 100:.1f} cm | "
            f"Fingerspanne {ep_result['finger_span_max_rad']:.2f} rad | "
            f"Anhebung max. {ep_result['max_cube_lift_cm']:.1f} cm"
        )
        reached_stages = [k for k, v in ep_result["milestones"].items() if v]
        print(f"      Meilensteine: {' → '.join(reached_stages) if reached_stages else '(keiner)'}")

    # Auswertung
    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / len(results) if results else 0.0
    print()
    print("=" * 60)
    print(f"Ergebnis: {n_success}/{len(results)} Erfolge — Success-Rate: {success_rate:.1%}")
    # Bei 0 Erfolgen ist das die eigentliche Information: kam die Hand nie an einen Würfel
    # (Geometrie), oder stand sie daneben ohne zu greifen (Wahrnehmung/Politik)?
    if results:
        frame = results[0]["reach_frame"]
        reaches = sorted(r["min_reach_m"] for r in results)
        shifts = [r["block_shift_max_m"] for r in results]
        n_touched = sum(1 for s in shifts if s > 0.01)
        # Wie weit hat die Politik den Abstand gegenüber der Ausgangspose verkleinert?
        # Werte um 0 heißen: die beste Annäherung der Episode war der Reset selbst, der
        # Roboter hat sich also nie auf einen Würfel zubewegt.
        gains = sorted(r["reach_start_m"] - r["min_reach_m"] for r in results)
        print(
            f"  Min. {frame}-Würfel-Abstand: bester {reaches[0] * 100:.1f} cm, "
            f"Median {reaches[len(reaches) // 2] * 100:.1f} cm"
        )
        print(
            f"  Annäherung ggü. Startpose: bester {gains[-1] * 100:.1f} cm, "
            f"Median {gains[len(gains) // 2] * 100:.1f} cm"
        )
        print(f"  Episoden mit bewegtem Würfel (>1 cm nach Einschwingen): "
              f"{n_touched}/{len(results)}")
        spans = sorted(r["finger_span_max_rad"] for r in results)
        median_span = spans[len(spans) // 2]
        print(
            f"  Fingerspanne (Kommando): Median {median_span:.2f} rad "
            f"= {median_span / DEMO_FINGER_SPAN_RAD:.0%} der Demonstration "
            f"({DEMO_FINGER_SPAN_RAD:.2f} rad)"
        )
        lifts = sorted(r["max_cube_lift_cm"] for r in results)
        print(
            f"  Würfel-Anhebung: beste {lifts[-1]:.1f} cm, "
            f"Median {lifts[len(lifts) // 2]:.1f} cm"
        )
        print()
        # Die Leiter als Ganzes: an welcher Stufe bricht es ab? Das ist die Zahl, gegen
        # die sich eine Maßnahme bewerten lässt — nicht die Erfolgsrate, die 0 bleibt.
        #
        # Die Stufen sind bewusst NICHT als monotone Kette implementiert (jede prüft ihre
        # eigene Bedingung), weil die Abweichungen davon informativ sind: 'touched' ohne
        # 'reached' heißt, dass etwas anderes als die sechs Fingerkuppen den Würfel
        # angestoßen hat — Handfläche oder Fingerglieder. Genau das zeigen die Zahlen aus
        # Lauf 30 (Würfel 3,8 cm verschoben bei 6,3 cm Kuppenabstand). Eine erzwungene
        # Monotonie würde diesen Fall unsichtbar machen.
        print("  Meilenstein-Leiter (Episoden, die die Stufe erreichen):")
        for stage in ("reached", "grasp_attempted", "touched", "lifted", "stacked"):
            n = sum(1 for r in results if r["milestones"][stage])
            bar = "#" * round(20 * n / len(results))
            print(f"    {stage:<16} {n:>3}/{len(results)}  {bar}")
    print("=" * 60)

    # Ergebnisse speichern
    summary = {
        "num_episodes": len(results),
        "num_success": n_success,
        "success_rate": success_rate,
        # Ohne dieses Feld ist ein Ergebnis nicht zuordenbar: die Datei liegt immer unter
        # demselben Pfad und wird von jedem Lauf überschrieben.
        "checkpoint": ckpt,
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
