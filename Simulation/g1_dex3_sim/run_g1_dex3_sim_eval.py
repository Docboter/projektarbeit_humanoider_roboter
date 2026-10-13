# TL;DR: Closed-Loop-Eval: verbindet Sim mit GR00T-Policy-Server per ZMQ; `server_rl_run.sh eval`.
"""
Closed-Loop-Eval-Runner für GR00T (N1.6 oder N1.7, s. GROOT_VERSION) auf G1 + Dex3 in Isaac Lab.

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
from pathlib import Path
import sys
import time

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
    "--task-description", type=str, default="",
    help="Language-Prompt für die Policy. Leer = aus meta/tasks.* des echten Datensatzes "
         "(TASK_DATASET_PATH), sonst 'stack three block' (s. task_text.py)"
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
parser.add_argument(
    "--layout", type=str, default=os.environ.get("EVAL_LAYOUT", ""),
    help="layout.json aus extract_block_layout.py (am besten mit --held-out erzeugt): "
         "Würfel starten an echten Startlagen samt Gierwinkel, reihum je Episode. "
         "Leer = Zufallsbänder aus der Env-Cfg (Startlagen, die so nie im Training "
         "vorkamen — Review §3.2). Default aus EVAL_LAYOUT."
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

from client import PolicyClient, build_obs
from eval_metrics import (
    HAND_CONTACT_PROXY_M,
    LIFT_CLEAR_THRESHOLD_M,
    clopper_pearson,
    format_rate_ci,
    lifted_clear_step,
)
from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg
from g1_dex3_cfg import CAMERA_CFG
import numpy as np
from scene_geometry import CUBE_EDGE, TABLE_HEIGHT
from split_math import layout_train_count
from task_text import DEFAULT_DATASET_PATH, resolve_task_description
import torch


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
# Würfel-Startlagen
# ---------------------------------------------------------------------------

# Grenze der zurückgehaltenen Episoden: bis 2026-10-13 fest 240 (= int(301 * 0.8)), was
# nur für den echten Datensatz mit Ratio 0,8 stimmt (Review 2026-10-13, S6). Jetzt steht
# n_train im Kopf der layout.json (extract_block_layout.py, Formel split_math.py =
# Training/scripts/lib_split.sh); ältere Dateien rechnet split_math.layout_train_count
# aus meta/info.json + TRAIN_SPLIT_RATIO nach — mit Warnung.

def load_eval_layout(path: str) -> tuple[list[dict], dict]:
    """layout.json → (Einträge für cfg.block_layout, Zusammenfassung fürs Ergebnis-JSON).

    Übernommen werden nur Episoden, in denen alle drei Würfel gefunden wurden — eine
    halbe Lage ist keine echte Startlage. Ein Layout aus Trainings-Episoden wird
    akzeptiert, aber laut markiert: dann ist Memorisierung nicht ausgeschlossen.
    """
    raw = json.loads(Path(path).read_text())
    episodes = raw.get("episodes", {})
    entries, skipped = [], []
    for key in sorted(episodes, key=int):
        rec = episodes[key]
        cubes = rec.get("cubes") or []
        if len(cubes) < 3 or not all(cubes[:3]):
            skipped.append(int(key))
            continue
        entries.append({"key": key, "cubes": cubes[:3], "yaws": rec.get("cubes_yaw_deg")})
    if not entries:
        raise SystemExit(f"--layout {path}: keine Episode mit allen drei Würfeln.")
    n_train, n_train_source = layout_train_count(
        raw, dataset_fallback=os.environ.get("TASK_DATASET_PATH") or DEFAULT_DATASET_PATH)
    if n_train_source != "layout.json":
        print(f"[layout] {n_train_source}", flush=True)
    # None = Grenze unbekannt: dann lieber KEINE Aussage als eine falsche.
    train_eps = (None if n_train is None
                 else [int(e["key"]) for e in entries if int(e["key"]) < n_train])
    info = {
        "path": path,
        "split": raw.get("split", "unbekannt"),
        "n_train": n_train,
        "n_train_source": n_train_source,
        "episodes": [int(e["key"]) for e in entries],
        "skipped_incomplete": skipped,
        "from_training_episodes": train_eps,
    }
    return entries, info


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
# `lifted_clear` (Review 2026-10-13, M3): die Schwerpunkt-Metrik oben zählt einen auf die
# Ecke gekippten Würfel schon fast mit (+1,8 cm). Die kipp-robuste Variante misst die
# Würfel-UNTERSEITE gegen die Tischoberkante (> LIFT_CLEAR_THRESHOLD_M = 2 cm) und
# verlangt im selben Schritt Handkontakt. Kontaktsensoren hat die Env nicht; Proxy ist
# der Fingerkuppen-Abstand zur Würfelmitte <= HAND_CONTACT_PROXY_M (4 cm), s.
# eval_metrics.py. `lifted` bleibt für die Vergleichbarkeit mit älteren Läufen stehen.
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
    # Im synchronen Modus ist der automatische Renderer praktisch deaktiviert. Ein
    # initialer Frame stellt sicher, dass DirectRLEnv.reset() beim Aufbau seiner (hier
    # ungenutzten) Observation nicht auf einen noch uninitialisierten Kamera-Puffer trifft.
    if env.cfg.camera_render_every_n > 1:
        env.force_policy_camera_render()
    obs_dict, _ = env.reset()
    # Fehler beim Server-Reset sind derselbe Fall wie bei get_action: die Episode ist
    # ungültig, nicht misslungen (s. error_reason unten).
    error_reason = None
    try:
        client.reset()
    except Exception as e:
        error_reason = f"reset: {type(e).__name__}: {e}"
        print(f"[Episode] Fehler bei reset: {e}", flush=True)
    layout_key = (getattr(env, "last_layout_keys", None) or [None])[0]

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
    # Kipp-robuste Anhebung: größte Unterseiten-Höhe über dem Tisch eines Würfels MIT
    # Handkontakt (Proxy) und ob dabei die Schwelle fiel. Tischoberkante in Weltkoordinaten
    # = Env-Ursprung + TABLE_HEIGHT (eine Quelle: scene_geometry.py).
    lifted_clear = False
    max_clear_lift = 0.0
    table_top_z = float(env.scene.env_origins[0, 2]) + TABLE_HEIGHT

    # Taille: hat keine Aktuatorgruppe, nur die Wurzel ist fixiert (g1_dex3_cfg.py). Größte
    # Abweichung der waist_*-Gelenke von der Startpose belegt, ob der Torso absackt und die
    # Kopfkamera damit eine andere Perspektive bekommt (Review §3.8).
    waist_ids = [i for i, n in enumerate(env.robot.joint_names) if "waist" in n]
    waist_start = env.robot.data.joint_pos[0, waist_ids].clone() if waist_ids else None
    waist_drift = 0.0

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
    last_render_generation = None
    last_sensor_tokens = None
    reported_static_sensor_tokens = False

    while step < max_steps and error_reason is None:
        if env.cfg.camera_render_every_n > 1:
            if env.cfg.camera_render_every_n != execution_horizon:
                raise RuntimeError(
                    "Vergleichbarer synchroner Modus verlangt "
                    "CAMERA_RENDER_EVERY_N == EXECUTION_HORIZON; "
                    f"erhalten {env.cfg.camera_render_every_n} != {execution_horizon}"
                )
            render_state = env.force_policy_camera_render()
            generation = render_state["generation"]
            if last_render_generation is not None and generation <= last_render_generation:
                raise RuntimeError("Kein neuer Policy-Render vor get_action()")
            tokens = render_state["sensor_tokens"]
            comparable_tokens = {
                key: value for key, value in tokens.items() if value is not None
            }
            if last_sensor_tokens is not None and comparable_tokens == last_sensor_tokens:
                # TiledCamera exponiert den Zeitstempel je nach Isaac-Lab-Version
                # nicht einheitlich. Er ist nur eine Diagnose und kein belastbarer
                # Beweis für einen alten RGB-Puffer. force_policy_camera_render()
                # aktualisiert daher den Kamera-Puffer explizit; hier protokollieren
                # wir eine statische private Zeitstempel-Implementierung einmalig,
                # statt eine ansonsten gültige Eval abzubrechen.
                if not reported_static_sensor_tokens:
                    print(
                        "[cam] Warnung: Isaac-Lab-Sensor-Zeitstempel nach explizitem "
                        "Kamera-Update unveraendert; Eval laeuft mit dem aktualisierten "
                        "RGB-Puffer weiter.",
                        flush=True,
                    )
                    reported_static_sensor_tokens = True
            last_render_generation = generation
            last_sensor_tokens = comparable_tokens or None

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
            # Bis 2026-10-13 nur print + break: die Episode zählte als normales 0-Ergebnis,
            # ein Server-Timeout sah aus wie eine misslungene Policy (Review S1). Jetzt ist
            # sie UNGÜLTIG (error=True) und geht in keine Quote ein.
            error_reason = f"get_action (Step {step}): {type(e).__name__}: {e}"
            print(f"[Episode] Fehler bei get_action (Step {step}): {e}", flush=True)
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
                # lifted_clear: Pose + Handabstand je Würfel, im selben Schritt.
                cube_quat = np.stack(
                    [b.data.root_quat_w[0].cpu().numpy() for b in env.blocks])
                hand_dist = torch.cdist(
                    env.get_contact_points_w()[0:1],
                    torch.stack([b.data.root_pos_w[0] for b in env.blocks]).unsqueeze(0),
                )[0].amin(dim=0).cpu().numpy()                     # (n_cubes,)
                ok_clear, clear = lifted_clear_step(
                    block_pos_last, cube_quat, hand_dist, table_top_z, CUBE_EDGE / 2)
                lifted_clear = lifted_clear or ok_clear
                max_clear_lift = max(max_clear_lift, clear)

            hand_cmd = np.asarray(chunk[t][14:28], dtype=float)
            np.minimum(hand_cmd_min, hand_cmd, out=hand_cmd_min)
            np.maximum(hand_cmd_max, hand_cmd, out=hand_cmd_max)

            _t0 = time.perf_counter()
            obs_step, _, terminated, time_out, info = env.step(action_t)
            t_env += time.perf_counter() - _t0
            if waist_start is not None and not (terminated.any() or time_out.any()):
                waist_drift = max(waist_drift, float(
                    (env.robot.data.joint_pos[0, waist_ids] - waist_start).abs().max()))

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
        # error=True: Episode wegen eines Server-/Kommunikationsfehlers abgebrochen. Sie
        # ist kein Messergebnis und zählt in keine Quote (Review 2026-10-13, S1).
        "error": error_reason is not None,
        "error_reason": error_reason,
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
        # Welche echte Startlage (Episodenindex aus layout.json) diese Episode hatte;
        # None = Zufallsbänder.
        "layout_episode": layout_key,
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
        # Unterseite über Tisch, nur Würfel mit Handkontakt-Proxy (s. lifted_clear).
        "max_clear_lift_cm": round(max_clear_lift * 100, 2),
        "waist_drift_max_rad": round(waist_drift, 4),
        # Gestufte Meilensteine: jede Stufe für sich auswertbar, damit ein 0/20 beim
        # Stapeln nicht mehr die einzige Information der Episode ist.
        "milestones": {
            "reached": bool(min_reach <= REACH_THRESHOLD_M),
            "grasp_attempted": bool(
                finger_span.max() >= GRASP_ATTEMPT_FRACTION * DEMO_FINGER_SPAN_RAD
            ),
            "touched": bool(block_shift.max() > TOUCH_THRESHOLD_M),
            "lifted": bool(max_cube_lift > LIFT_THRESHOLD_M),
            "lifted_clear": bool(lifted_clear),
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
    # Der Text muss genau der aus dem Training sein. Bis 2026-10-13 stand hier fest
    # "stack the blocks", der Datensatz kennt nur "stack three block" (Review §3.1).
    args.task_description, task_source = resolve_task_description(args.task_description)
    cfg.task_description = args.task_description
    cfg.dr_enabled = os.getenv("DR_ENABLED", "1") != "0"
    # DR_SEED macht die Ziehung reproduzierbar (je Episode ein eigener Strom). Default 0:
    # ohne festen Seed sind A/B-Läufe, die sich in genau einem Faktor unterscheiden
    # sollen, nicht vergleichbar (Review §3.7). Negativ oder leer = wie früher pro Prozess
    # auswürfeln (gleiche Konvention wie RENDER_DR_SEED). Derselbe Seed fixiert auch den
    # Torch-Zufall, aus dem die Zufallsbänder die Würfellagen ziehen.
    _dr_seed = os.getenv("DR_SEED", "0").strip()
    cfg.dr_seed = int(_dr_seed) if _dr_seed and int(_dr_seed) >= 0 else None
    if cfg.dr_seed is not None:
        torch.manual_seed(cfg.dr_seed)
        np.random.seed(cfg.dr_seed)
    if cfg.camera_render_every_n > 1 and cfg.camera_render_every_n != args.execution_horizon:
        raise ValueError(
            "CAMERA_RENDER_EVERY_N muss im vergleichbaren schnellen Modus exakt "
            f"EXECUTION_HORIZON entsprechen ({cfg.camera_render_every_n} != "
            f"{args.execution_horizon})"
        )
    # Wirkt an beiden Stellen gleichzeitig, weil beide dieselbe cfg lesen: run_episode
    # bildet daraus max_steps, und _get_dones() den time_out. Ein Wert hier kann also
    # nicht auseinanderlaufen mit dem, was die Umgebung selbst als Episodenende sieht.
    if args.episode_length_s > 0:
        cfg.episode_length_s = args.episode_length_s
    layout_info: dict = {"path": "", "note": "Zufallsbänder (kein --layout / EVAL_LAYOUT)"}
    if args.layout:
        cfg.block_layout, layout_info = load_eval_layout(args.layout)

    print("=" * 60)
    # Version aus dem Entrypoint (lib_groot_version.sh exportiert GROOT_VERSION nach der
    # Auflösung aus dem Checkpoint); bis 2026-10-13 stand hier fest "N1.6", auch bei N1.7.
    _gv = os.getenv("GROOT_VERSION", "").strip()
    _gv = f"N{_gv}" if _gv in ("1.6", "1.7") else "(Version nicht übergeben)"
    print(f"GR00T {_gv} — G1+Dex3 Closed-Loop Sim Eval")
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
    print(f"  Kamera-Render:    alle {cfg.camera_render_every_n} Policy-Schritt(e)")
    print(f"  Task:             {args.task_description!r}  [{task_source}]")
    print(f"  Video-Dir:        {args.video_dir or '(kein Video)'}")
    print(f"  Asset:            {cfg.scene.robot.spawn.usd_path}")
    _dr_state = "AN" if cfg.dr_enabled else "AUS (DR_ENABLED=0)"
    if cfg.dr_enabled and cfg.dr_seed is not None:
        _dr_state += f", Seed {cfg.dr_seed} je Episode"
    print(f"  Domain Rand.:     {_dr_state}")
    if args.layout:
        print(f"  Würfel-Start:     {len(cfg.block_layout)} echte Lagen aus {args.layout} "
              f"(Split: {layout_info['split']})")
        if layout_info["from_training_episodes"] is None:
            print(f"  [WARNUNG] Split-Grenze unbekannt ({layout_info['n_train_source']}) — "
                  "ob die Lagen aus Trainings-Episoden stammen, ist nicht prüfbar.")
        elif layout_info["from_training_episodes"]:
            print(f"  [WARNUNG] {len(layout_info['from_training_episodes'])} Lagen stammen aus "
                  f"TRAININGS-Episoden (< {layout_info['n_train']}) — Memorisierung nicht "
                  "ausgeschlossen. Für die Messung: layout mit LAYOUT_HELD_OUT=1 erzeugen.")
    else:
        print("  Würfel-Start:     Zufallsbänder — NICHT die Startlagen der Realdaten "
              "(Review §3.2). Echte Lagen: EVAL_LAYOUT=<layout.json>")
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
    dummy_obs = build_obs(
        dummy_rgb, dummy_rgb, dummy_rgb, dummy_rgb, dummy_joint,
        task_description=args.task_description,
    )
    # Der erste Aufruf ist der teuerste (torch.compile/TensorRT-Warmup, CUDA-Graphen) und
    # kann den 20-s-Timeout reißen. Der Client baut seinen Socket danach neu auf, ein
    # zweiter Versuch trifft den dann warmen Server (Review 2026-10-13, S1).
    dummy_chunk = None
    for _attempt in range(1, 4):
        try:
            dummy_chunk = client.get_action(dummy_obs)
            break
        except Exception as e:
            print(f"[Phase D] Dummy-get_action Versuch {_attempt}/3 fehlgeschlagen: "
                  f"{type(e).__name__}: {e}", flush=True)
    if dummy_chunk is None:
        print("[FEHLER] Policy-Server liefert keine Aktion. Abbruch.", flush=True)
        client.close()
        env.close()
        sys.exit(1)
    assert dummy_chunk.shape == (16, 28), f"Unerwartete Chunk-Shape: {dummy_chunk.shape}"
    print(f"[Phase D] Dummy-get_action OK: shape={dummy_chunk.shape}")
    print()

    # Eval-Schleife
    results = []
    record = bool(args.video_dir)
    # Zwei Fehler in Folge = der Server ist weg oder hängt. Weiterlaufen lieferte nur
    # weitere ungültige Episoden, die man sonst für ein plausibles 0/N hielte.
    max_consecutive_errors = 2
    consecutive_errors = 0
    aborted_reason = None

    for ep in range(args.num_episodes):
        print(f"--- Episode {ep + 1}/{args.num_episodes} ---")
        if cfg.block_layout:
            env.layout_index = ep   # Lage ep (reihum), s. _place_blocks_from_layout
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

        if ep_result["error"]:
            consecutive_errors += 1
            print(f"  Episode {ep + 1}: ABGEBROCHEN (ungültig) — {ep_result['error_reason']}",
                  flush=True)
            if consecutive_errors >= max_consecutive_errors:
                aborted_reason = (f"{consecutive_errors} Episoden in Folge abgebrochen, "
                                  f"zuletzt: {ep_result['error_reason']}")
                print(f"[FEHLER] {aborted_reason} — Eval wird beendet.", flush=True)
                break
            continue
        consecutive_errors = 0

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
            f"Anhebung max. {ep_result['max_cube_lift_cm']:.1f} cm "
            f"(Unterseite mit Hand {ep_result['max_clear_lift_cm']:.1f} cm)"
        )
        reached_stages = [k for k, v in ep_result["milestones"].items() if v]
        print(f"      Meilensteine: {' → '.join(reached_stages) if reached_stages else '(keiner)'}")

    # Auswertung — Quoten NUR über gültige Episoden. Eine abgebrochene Episode ist kein
    # Misserfolg der Policy, sondern keine Messung.
    valid = [r for r in results if not r["error"]]
    n_aborted = len(results) - len(valid)
    n_success = sum(1 for r in valid if r["success"])
    success_rate = n_success / len(valid) if valid else 0.0
    print()
    print("=" * 60)
    print(f"Episoden: {len(valid)} gültig / {n_aborted} abgebrochen"
          f"{' (Lauf vorzeitig beendet)' if aborted_reason else ''}")
    print(f"Ergebnis: {n_success}/{len(valid)} Erfolge — Success-Rate: {success_rate:.1%}")
    # Bei 0 Erfolgen ist das die eigentliche Information: kam die Hand nie an einen Würfel
    # (Geometrie), oder stand sie daneben ohne zu greifen (Wahrnehmung/Politik)?
    milestone_ci: dict = {}
    if valid:
        frame = valid[0]["reach_frame"]
        reaches = sorted(r["min_reach_m"] for r in valid)
        shifts = [r["block_shift_max_m"] for r in valid]
        n_touched = sum(1 for s in shifts if s > 0.01)
        # Wie weit hat die Politik den Abstand gegenüber der Ausgangspose verkleinert?
        # Werte um 0 heißen: die beste Annäherung der Episode war der Reset selbst, der
        # Roboter hat sich also nie auf einen Würfel zubewegt.
        gains = sorted(r["reach_start_m"] - r["min_reach_m"] for r in valid)
        print(
            f"  Min. {frame}-Würfel-Abstand: bester {reaches[0] * 100:.1f} cm, "
            f"Median {reaches[len(reaches) // 2] * 100:.1f} cm"
        )
        print(
            f"  Annäherung ggü. Startpose: bester {gains[-1] * 100:.1f} cm, "
            f"Median {gains[len(gains) // 2] * 100:.1f} cm"
        )
        print(f"  Episoden mit bewegtem Würfel (>1 cm nach Einschwingen): "
              f"{n_touched}/{len(valid)}")
        spans = sorted(r["finger_span_max_rad"] for r in valid)
        median_span = spans[len(spans) // 2]
        print(
            f"  Fingerspanne (Kommando): Median {median_span:.2f} rad "
            f"= {median_span / DEMO_FINGER_SPAN_RAD:.0%} der Demonstration "
            f"({DEMO_FINGER_SPAN_RAD:.2f} rad)"
        )
        lifts = sorted(r["max_cube_lift_cm"] for r in valid)
        print(
            f"  Würfel-Anhebung (Schwerpunkt): beste {lifts[-1]:.1f} cm, "
            f"Median {lifts[len(lifts) // 2]:.1f} cm"
        )
        clears = sorted(r["max_clear_lift_cm"] for r in valid)
        print(
            f"  Würfel-Unterseite über Tisch (mit Hand, Proxy <= "
            f"{HAND_CONTACT_PROXY_M * 100:.0f} cm): beste {clears[-1]:.1f} cm, "
            f"Median {clears[len(clears) // 2]:.1f} cm"
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
        print("  Meilenstein-Leiter (gültige Episoden, die die Stufe erreichen):")
        for stage in ("reached", "grasp_attempted", "touched", "lifted", "lifted_clear",
                      "stacked"):
            n = sum(1 for r in valid if r["milestones"][stage])
            bar = "#" * round(20 * n / len(valid))
            print(f"    {stage:<16} {n:>3}/{len(valid)}  {bar}")
        # Clopper-Pearson 95 % (Review 2026-10-13, M4): bei n = 10 bis 20 ist das
        # Intervall so breit, dass zwei Läufe mit 0/10 und 1/10 sich NICHT unterscheiden.
        print("  95-%-Konfidenzintervalle (Clopper-Pearson, exakt):")
        for stage in ("lifted", "lifted_clear", "stacked"):
            n = sum(1 for r in valid if r["milestones"][stage])
            print(f"    {stage:<16} {format_rate_ci(n, len(valid))}")
            milestone_ci[stage] = {"k": n, "n": len(valid)}
    print("=" * 60)

    # KI-Grenzen fürs JSON (gleiche Funktion wie die Ausgabe oben).
    for stage, rec in milestone_ci.items():
        lo, hi = clopper_pearson(rec["k"], rec["n"])
        rec.update({"rate": rec["k"] / rec["n"], "ci95": [round(lo, 4), round(hi, 4)]})

    # Ergebnisse speichern
    summary = {
        # num_episodes = durchgeführte Episoden (gültig + abgebrochen); Quoten beziehen
        # sich auf num_valid. Bis 2026-10-13 zählte jede Episode als gültig (Review S1).
        "num_episodes": len(results),
        "num_episodes_requested": args.num_episodes,
        "num_valid": len(valid),
        "num_aborted": n_aborted,
        "aborted": aborted_reason is not None,
        "aborted_reason": aborted_reason,
        "num_success": n_success,
        "success_rate": success_rate,
        "milestone_ci95": milestone_ci,
        "lifted_clear_definition": {
            "bottom_above_table_m": LIFT_CLEAR_THRESHOLD_M,
            "hand_contact": "Proxy: Fingerkuppen-Abstand zur Würfelmitte <= "
                            f"{HAND_CONTACT_PROXY_M} m (keine Kontaktsensoren in der Env)",
        },
        "groot_version": os.getenv("GROOT_VERSION", ""),
        # Ohne dieses Feld ist ein Ergebnis nicht zuordenbar (s. auch die Archivkopie
        # mit Zeitstempel + Checkpoint-Name unten).
        "checkpoint": ckpt,
        "server": args.server,
        "task_description": args.task_description,
        "task_description_source": task_source,
        "execution_horizon": args.execution_horizon,
        "policy_hz": cfg.policy_hz,
        "camera_render_every_n": cfg.camera_render_every_n,
        "inference_backend": os.getenv("GROOT_INFERENCE_BACKEND", "eager"),
        "scene_camera_enabled": os.getenv("SCENE_CAM", "1") not in ("0", "false", "no"),
        # Messaufbau, damit zwei Läufe vergleichbar sind (Review §3.7 / §8 Eval-Manifest)
        "cube_layout": layout_info,
        "domain_randomization": {"enabled": cfg.dr_enabled, "seed": cfg.dr_seed},
        "physics_dt": cfg.sim.dt,
        "decimation": cfg.decimation,
        "episodes": results,
    }
    summary["episode_length_s"] = cfg.episode_length_s
    summary["max_steps_per_episode"] = int(cfg.episode_length_s * cfg.policy_hz)
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Je Lauf eine eigene Datei (Zeitstempel + Checkpoint-Name), damit ein Folgelauf das
    # Ergebnis nicht überschreibt; --results-file bleibt als "neuester Lauf" (Kopie)
    # bestehen, alle Leser (server_rl_run.sh, optimize, KISSKI) lesen weiter dort.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_name = Path(args.checkpoint_path).name if args.checkpoint_path else "unbekannt"
    ckpt_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in ckpt_name) or "ckpt"
    archive = results_path.with_name(
        f"{results_path.stem}_{stamp}_{ckpt_name}{results_path.suffix or '.json'}")
    payload = json.dumps(summary, indent=2)
    archive.write_text(payload)
    results_path.write_text(payload)
    print(f"Ergebnisse gespeichert: {results_path} (Archiv: {archive.name})")

    # Aufräumen
    client.close()
    if aborted_reason is not None:
        # KEIN Erfolgsmarker: die Entrypoints werten ihn aus (isaaclab.sh reicht den
        # Exit-Code nicht zuverlässig durch). Exit != 0 trotzdem, für direkte Aufrufe.
        print(f"[eval] ABGEBROCHEN: {aborted_reason}", flush=True)
        env.close()
        simulation_app.close()
        sys.exit(2)
    # Erfolgsmarker wie in dump_camera_poses.py / measure_domain_gap.py: isaaclab.sh gibt
    # den Exit-Code des Pythons NICHT durch (ein Traceback endet trotzdem mit 0), ein
    # Abbruch mitten in der Eval-Schleife bliebe sonst unbemerkt.
    print("[eval] fertig.", flush=True)
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
