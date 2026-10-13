# TL;DR: Zeigt die Szene ohne Modell/Checkpoint (Home-Pose); `server_rl_run.sh view`.
"""
Szene ansehen — die Isaac-Lab-Umgebung OHNE Modell, OHNE Checkpoint, OHNE HF-Gewichte.

"Leer" heisst hier: leer an Policy, nicht an Szene. Aufgebaut werden Roboter, Tisch,
Wuerfel, Licht und die Kameras — also exakt dieselbe `G1Dex3BlockstackEnv` wie in Eval,
Replay und RL-Lauf. Was fehlt, ist alles, was Gewichte braucht: kein GR00T-Server, kein
Gr00tPolicy, kein Checkpoint-Download, kein HF_TOKEN. Der Roboter haelt still seine
Home-Pose, damit man die Szene in Ruhe ansehen kann.

Wozu das gut ist:
  * Die LIVE-Variante (WebRTC-Viewport) ausprobieren, bevor ein echter Lauf davon abhaengt
    — jeder andere Weg dorthin (`check`, `eval`, `grasp`) verlangt erst ~10 GB Checkpoint.
  * Nachsehen, wie die Szene ueberhaupt aussieht: Tischhoehe, Wuerfelpositionen,
    Kamerawinkel, Bodenfarbe, Beleuchtung.
  * Nachweis, dass Isaac Sim + RT-Core-Rendering auf dieser GPU laufen, ohne dass ein
    Modell die Diagnose verwaessert.

Was NICHT gemessen wird: gar nichts. Es gibt keine Erfolgsrate, keine results.json und
keinen Vergleich mit anderen Laeufen. Das Skript ist ein Fenster, kein Messinstrument —
deshalb sind hier auch Render-Sparhebel erlaubt, die in einem Messlauf verboten waeren
(CAM_RES_SCALE veraendert die Modell-Eingabe; hier gibt es keine).

Sichtbar wird der Lauf ueber einen der beiden Live-Wege (docs/simulation/live-ansicht.md):
  --livestream 2   Spur A: kompletter Isaac-Sim-Viewport per WebRTC, freie Kamera.
  --live-view      Spur B: die gerenderten Kamerabilder als MJPEG im Browser.
Ohne beides laeuft die Sim headless — dann sieht niemand etwas, und das Skript sagt es.

Verwendung (im Sim-Container, RT-Core-GPU):
    unset VIRTUAL_ENV
    ${ISAACLAB_PATH}/_isaac_sim/python.sh /workspace/g1_dex3_sim/view_sim.py \\
        --livestream 2 --enable_cameras \\
        --asset-path /data/assets/g1_dex3.usd
Bequemer:  ./Simulation/server_rl_run.sh view
"""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher


# ---------------------------------------------------------------------------
# Env-Defaults fuer die CLI
# ---------------------------------------------------------------------------
# Gleiche Begruendung wie in rl_finetune.py: server_rl_run.sh mountet dieses Verzeichnis
# live in den Container, /scripts dagegen nicht. Liest das Skript die Schalter selbst aus
# Env-Vars, genuegt ein `-e LIVE_VIEW=1` am docker exec — kein Image-Rebuild.
def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


parser = argparse.ArgumentParser(description="G1+Dex3-Szene ansehen (kein Modell, kein Checkpoint)")
parser.add_argument("--asset-path", type=str, default="",
                    help="Pfad zum G1+Dex3-USD (leer = cfg-Default aus g1_dex3_cfg.py)")
parser.add_argument("--num-envs", type=int, default=_env_int("VIEW_NUM_ENVS", 1),
                    help="Parallele Envs. 1 zeigt eine aufgeraeumte Szene, >1 das Klon-"
                         "Gitter. Env: VIEW_NUM_ENVS.")
# Immer endlich: bricht die SSH-Sitzung weg, liefe ein unbegrenzter Lauf sonst unbemerkt
# weiter und blockierte die GPU. Ein "0 = bis Strg-C" gibt es bewusst NICHT mehr — der Lauf
# haengt an einem `docker exec` ohne TTY, Strg-C toetet nur den lokalen Client und erreicht
# den Prozess im Container nie. Wer laenger zusehen will, setzt eine groessere Zahl.
DEFAULT_DURATION_S = 3600.0
parser.add_argument("--duration-s", type=float,
                    default=_env_float("VIEW_DURATION_S", DEFAULT_DURATION_S),
                    help="Laufzeit in Sekunden (> 0), danach sauberes Ende. Laenger "
                         "zusehen: groesseren Wert setzen, z. B. 14400 (4 h). "
                         "Env: VIEW_DURATION_S.")
parser.add_argument("--episode-length-s", type=float, default=_env_float("EPISODE_LENGTH_S", 0.0),
                    help="Sekunden bis zum Auto-Reset (Wuerfel werden dabei neu gewuerfelt). "
                         "0 = cfg-Default (300 s). Env: EPISODE_LENGTH_S.")
parser.add_argument("--log-every", type=int, default=100,
                    help="Alle n Steps eine Tempo-Zeile ausgeben (0 = nie)")
# ── Spur B: MJPEG-Bilder im Browser ─────────────────────────────────────────
parser.add_argument("--live-view", action="store_true", default=_env_flag("LIVE_VIEW"),
                    help="Kamerabilder als MJPEG im Browser. Env: LIVE_VIEW.")
parser.add_argument("--live-view-port", type=int, default=_env_int("LIVE_VIEW_PORT", 8900),
                    help="HTTP-Port der Live-Ansicht. Env: LIVE_VIEW_PORT.")
parser.add_argument("--live-view-every-n", type=int, default=_env_int("LIVE_VIEW_EVERY_N", 1),
                    help="Nur jedes n-te Frame publizieren. Env: LIVE_VIEW_EVERY_N.")
parser.add_argument("--live-view-cams",
                    default=os.environ.get("LIVE_VIEW_CAMS", "cam_left_high,cam_left_wrist"),
                    help="Kommagetrennte Kameras. Env: LIVE_VIEW_CAMS.")
# ── Spur A: WebRTC-Viewport ─────────────────────────────────────────────────
# --livestream selbst kommt von AppLauncher.add_app_launcher_args (die Shell-Seite baut
# es in lib_livestream.sh zusammen). Hier nur der Nachzieh-Takt fuer den Viewport.
parser.add_argument("--livestream-update-every-n", type=int,
                    default=_env_int("LIVESTREAM_UPDATE_EVERY_N", 1),
                    help="Nur mit --livestream: alle n Steps simulation_app.update() rufen, "
                         "damit der Viewport nachzieht. 0 = nie. "
                         "Env: LIVESTREAM_UPDATE_EVERY_N.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports nach AppLauncher-Start
# ---------------------------------------------------------------------------

from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402
from live_view import LiveView  # noqa: E402


def main() -> None:
    # AppLauncher setzt --livestream auf -1, wenn nichts angegeben wurde.
    ls_mode = int(getattr(args, "livestream", -1) or 0)
    ls_active = ls_mode > 0

    cfg = G1Dex3BlockstackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    if args.episode_length_s > 0:
        cfg.episode_length_s = args.episode_length_s

    # 0/negativ hiess frueher "unbegrenzt, bis Strg-C". Das war unbrauchbar: der Lauf
    # startet per `docker exec` ohne TTY, Strg-C beendet nur den Client auf dem Host. Im
    # Container liefe die Sim weiter, der Erfolgsmarker unten kaeme nie, und
    # server_rl_run.sh meldete den Lauf faelschlich als gescheitert.
    duration_s = args.duration_s
    if duration_s <= 0:
        print(f"[view] --duration-s {duration_s:g} gibt es nicht mehr (Strg-C erreicht den "
              f"Container nicht) — nutze {DEFAULT_DURATION_S:.0f} s. Laenger zusehen: "
              f"VIEW_DURATION_S=14400.", flush=True)
        duration_s = DEFAULT_DURATION_S

    laufzeit = f"{duration_s:.0f} s"
    spur_a = f"an, Modus {ls_mode}" if ls_active else "aus"
    spur_b = f"an, Port {args.live_view_port}" if args.live_view else "aus"

    print("=" * 66)
    print("G1+Dex3 — SZENE ANSEHEN (kein Modell, kein Checkpoint, keine HF-Gewichte)")
    print("=" * 66)
    print(f"  Asset:            {cfg.scene.robot.spawn.usd_path}")
    print(f"  Envs:             {args.num_envs}")
    print(f"  Auto-Reset:       alle {cfg.episode_length_s:.0f} s (Wuerfel neu gewuerfelt)")
    print(f"  Laufzeit:         {laufzeit}")
    print(f"  WebRTC (Spur A):  {spur_a}")
    print(f"  Browser (Spur B): {spur_b}")
    print()

    if not ls_active and not args.live_view:
        print("[view] !! Weder --livestream noch --live-view: der Lauf ist headless und "
              "zeigt NIEMANDEM etwas. Gemeint war vermutlich LIVESTREAM=2 (Viewport) "
              "oder LIVE_VIEW=1 (Browser). Siehe docs/simulation/live-ansicht.md",
              flush=True)

    # render_mode bleibt None: es wird kein MP4 geschrieben, also braucht es auch keine
    # rgb_array-Kopie je Step. Die Kameras rendern trotzdem — sie haengen als Sensoren in
    # der Szene und sind die Bildquelle fuer Spur B.
    env = G1Dex3BlockstackEnv(cfg)

    view_cams = [c.strip() for c in args.live_view_cams.split(",") if c.strip()]
    live = LiveView(
        enabled=args.live_view,
        port=args.live_view_port,
        every_n=args.live_view_every_n,
        cams=view_cams or ["cam_left_high"],
        title=f"G1 DEX3 — Szene ansehen ({args.num_envs} Envs, kein Modell)",
    )

    # Halte-Ziel: die Aktionen sind ABSOLUTE Gelenk-Targets in Policy-Reihenfolge, und
    # _get_observations() liefert genau diese Groesse zurueck (die Vorzeichen-Flips der
    # proximalen Fingergelenke heben sich auf dem Rueckweg durch _pre_physics_step wieder
    # auf). Die Beobachtung nach dem Reset als Aktion zu wiederholen heisst deshalb exakt:
    # "bleib in der Home-Pose stehen".
    obs, _ = env.reset()
    hold = obs["joint_pos"].clone()

    print("[view] Szene steht. Der Roboter haelt seine Home-Pose.", flush=True)
    if ls_active:
        print("[view] Der Viewport erscheint im WebRTC-Client, sobald Isaac Sim die Szene "
              "geladen hat (~1 min).", flush=True)

    ls_update_every = args.livestream_update_every_n if ls_active else 0
    t0 = time.perf_counter()
    step = 0
    try:
        while simulation_app.is_running():
            elapsed = time.perf_counter() - t0
            if elapsed >= duration_s:
                print(f"[view] Laufzeit {duration_s:.0f} s erreicht — beende.", flush=True)
                break

            obs, _, terminated, time_out, _ = env.step(hold)
            step += 1

            # Nach einem Auto-Reset steht der Roboter wieder in der Home-Pose; das
            # Halte-Ziel neu greifen, damit es nicht auf einer Pose von vorher festhaengt.
            if bool((terminated | time_out).any()):
                hold = obs["joint_pos"].clone()

            live.publish_obs(obs, step=step, laufzeit_s=round(elapsed, 1))
            if ls_update_every and step % ls_update_every == 0:
                simulation_app.update()

            if args.log_every > 0 and step % args.log_every == 0:
                sps = step / max(elapsed, 1e-6)
                print(f"[view] Step {step} ({elapsed:.0f}s, {sps:.1f} Steps/s, "
                      f"{cfg.policy_hz / max(sps, 1e-6):.1f}x Echtzeit)", flush=True)
    except KeyboardInterrupt:
        print("\n[view] Strg-C — beende.", flush=True)

    dauer = time.perf_counter() - t0
    live.close()
    # Erfolgsmarker VOR env.close(): isaaclab.sh verschluckt den Exit-Code, server_rl_run.sh
    # erkennt den sauberen Durchlauf nur an dieser Zeile (wie '[replay] fertig.'). Sie darf
    # deshalb nicht daran haengen, dass auch noch das Herunterfahren glattgeht.
    print(f"[view] fertig. {step} Steps in {dauer:.0f} s.", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
