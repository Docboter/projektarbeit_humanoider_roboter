#!/usr/bin/env python3
"""Diagnose: KONFIGURIERTE gegen TATSAECHLICH gerenderte Kamera-Pose.

Anlass (2026-08-08): Die Live-Ansicht zeigte fuer `cam_scene` fast nur den hellen
Dome-Hintergrund plus eine Ecke Bodengitter — kein Tisch, kein Roboter. Nachrechnen der
konfigurierten Pose (Auge (1.8, 1.6, 1.7), Ziel (0.30, 0, 0.80), HFOV 60.4 Grad) ergab
aber, dass Tisch, Wuerfel und Roboter vollstaendig im Bild liegen muessten. Die Zahlen in
g1_dex3_cfg.py sind also nicht das Problem — irgendetwas zwischen Konfiguration und
Render weicht ab. Dieses Skript misst genau diese Luecke, statt an Koordinaten zu raten.

Ausgegeben wird je Kamera:
  * konfigurierter Offset (cfg.scene.<cam>.offset)
  * tatsaechliche Weltpose nach dem Reset (camera.data.pos_w / quat_w_world)
  * die Pose RELATIV zum Env-Ursprung — der entscheidende Vergleich, denn die Kameras
    haengen unter {ENV_REGEX_NS} und werden je Env geklont
  * der Winkel zwischen konfigurierter und tatsaechlicher Blickrichtung

Ausserdem ein PNG je Kamera. Bewusst mit Pillow statt imageio: Pillow ist im
Isaac-Sim-Kit-Python nachweislich vorhanden (die Live-Ansicht nutzt es), imageio nicht.

--num-envs 4 ist Default, weil der Befund am Vier-Env-RL-Lauf auftrat und ein
Ein-Env-Lauf die Klon-Frage per Konstruktion nicht beantworten kann.

Verwendung (im Sim-Container, RT-Core-GPU):
    unset VIRTUAL_ENV
    ${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/dump_camera_poses.py \\
        --headless --enable_cameras \\
        --asset-path /data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd
Bequemer:  ./Simulation/server_rl_run.sh cams
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kamera-Posen: konfiguriert vs. gerendert")
parser.add_argument("--asset-path", type=str, default=None, help="G1+Dex3 USD-Asset")
parser.add_argument("--out-dir", type=str, default="/data/cam_dump", help="PNG-Ausgabe")
parser.add_argument("--num-envs", type=int, default=4, help="Envs (Klon-Verhalten pruefen)")
parser.add_argument("--settle-steps", type=int, default=8, help="Steps vor der Messung")
parser.add_argument(
    "--no-dr", action="store_true",
    default=os.environ.get("DR_ENABLED", "1").strip() in ("0", "false", "no", "off"),
    help="Visuelle Domain Randomization abschalten (DR_ENABLED=0). Sie ist der einzige "
         "Code, der die Shader von Tisch/Würfeln/Stapel-Band anfasst — genau die "
         "Objekte, die im Render fehlen, während der Roboter sichtbar ist.",
)
parser.add_argument(
    "--dome-sweep", type=str, default=os.environ.get("RL_DOME_SWEEP", ""),
    help="Kommaliste von Dome-Light-Intensitäten, die zusätzlich gerendert werden "
         "(z. B. '500,120,30'). Leer = aus. Prüft den Verdacht, dass Isaac Sim 6.0 die "
         "Intensitäts-Einheit anders interpretiert als 4.x und die Szene überstrahlt.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- erst nach AppLauncher importierbar ---------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402
from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402


def quat_rotate(q, v):
    """Vektor v mit Quaternion q=(w, x, y, z) drehen."""
    w, x, y, z = [float(c) for c in q]
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def first_attr(obj, names):
    """Erstes vorhandenes Attribut zurueckgeben (Isaac Lab 3.0 ist Beta — Namen wandern)."""
    for n in names:
        if hasattr(obj, n):
            return n, getattr(obj, n)
    return None, None


def scene_points(env, origins):
    """Weltpositionen der Szenenobjekte zur LAUFZEIT — {Name: (3,)}.

    Der bislang ungeprüfte Teil: Alle Frustum-Rechnungen gingen von den Werten aus der
    Config aus (Tisch bei (0.5, 0, 0.435), Würfel bei z=0.915). Stehen die Objekte real
    woanders — falscher USD-Maßstab, weggekippt, anderes Asset —, ist jede daraus
    abgeleitete Aussage wertlos. Deshalb hier die echten Root-Posen aus der Physik.
    """
    pts = {}

    def add(name, obj):
        d = getattr(obj, "data", None)
        p = getattr(d, "root_pos_w", None) if d is not None else None
        if p is not None:
            pts[name] = p.detach().cpu().numpy()[0]

    add("table", getattr(env, "table", None))
    for i, b in enumerate(getattr(env, "blocks", []) or []):
        add(f"block_{i}", b)
    robot = getattr(env, "robot", None)
    add("robot_root", robot)
    # Ein paar Links, die in den Kameras sichtbar sein MÜSSEN, wenn alles stimmt.
    d = getattr(robot, "data", None)
    names = list(getattr(d, "body_names", []) or []) if d is not None else []
    bp = getattr(d, "body_pos_w", None) if d is not None else None
    if bp is not None and names:
        bp = bp.detach().cpu().numpy()[0]
        for want in ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"):
            if want in names:
                pts[f"link:{want}"] = bp[names.index(want)]
    return pts


def camera_tangents(cam, width, height):
    """Halbwinkel-Tangenten aus der Intrinsik, die Isaac Lab SELBST meldet. -> (tan_h, tan_v).

    Wichtiger als es aussieht: die Variante unten aus dem spawn-Cfg fällt auf eine
    angenommene horizontal_aperture von 20.955 mm zurück, wenn das Feld fehlt oder in
    Isaac Lab 3.0 anders heißt. Dann stimmt das Sichtfeld nicht und jedes „IM BILD" ist
    wertlos. `data.intrinsic_matrices` kommt dagegen aus dem Renderer.
    """
    m = getattr(getattr(cam, "data", None), "intrinsic_matrices", None)
    if m is None:
        return None
    try:
        k = m[0].detach().cpu().numpy()
        fx, fy = float(k[0, 0]), float(k[1, 1])
    except Exception:  # noqa: BLE001
        return None
    if not (fx > 0 and fy > 0):
        return None
    return (width / (2 * fx), height / (2 * fy))


def frustum_report(cam_cfg, pos_w, quat_w, pts, width, height, tangents=None):
    """Projiziert Szenenpunkte in eine Kamera. -> Zeilen (name, u, v, dist, drin?).

    u/v sind auf [-1, 1] normiert (0 = Bildmitte). `tangents` aus der Renderer-Intrinsik
    hat Vorrang; ohne sie wird aus dem spawn-Cfg gerechnet (mit Annahme, siehe oben).
    """
    spawn = getattr(cam_cfg, "spawn", None)
    if tangents is not None:
        tan_h, tan_v = tangents
    else:
        f = float(getattr(spawn, "focal_length", 24.0))
        ap_h = float(getattr(spawn, "horizontal_aperture", 20.955))
        tan_h, tan_v = ap_h / (2 * f), (ap_h * height / width) / (2 * f)
    clip = getattr(spawn, "clipping_range", (0.0, 1e9))

    x = quat_rotate(quat_w, [1.0, 0.0, 0.0])   # Blickachse (convention="world")
    y = quat_rotate(quat_w, [0.0, 1.0, 0.0])
    z = quat_rotate(quat_w, [0.0, 0.0, 1.0])
    rows = []
    for name, p in pts.items():
        v = np.asarray(p, dtype=float) - np.asarray(pos_w, dtype=float)
        fwd = float(np.dot(v, x))
        if fwd <= 1e-6:
            rows.append((name, None, None, fwd, "hinter der Kamera"))
            continue
        u = -float(np.dot(v, y)) / fwd / tan_h
        w = float(np.dot(v, z)) / fwd / tan_v
        dist = float(np.linalg.norm(v))
        if not (clip[0] <= dist <= clip[1]):
            state = f"ausserhalb clipping {tuple(clip)}"
        elif abs(u) <= 1 and abs(w) <= 1:
            state = "IM BILD"
        else:
            state = "ausserhalb des Bildes"
        rows.append((name, u, w, dist, state))
    return rows


def set_dome_intensity(value: float) -> bool:
    """Intensität des Dome-Lights unter /World/Light zur Laufzeit setzen. -> Erfolg?

    Gibt bei jedem Fehlschlag laut aus. Der Sweep darf den Dump nicht abbrechen, aber
    er darf auch nicht stillschweigend nichts tun und dann vier identische Bilder als
    "kein Effekt" ausgeben.
    """
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    except Exception as e:  # noqa: BLE001
        print(f"  !! omni.usd nicht verfügbar ({e}) — Dome-Sweep nicht möglich.")
        return False
    prim = stage.GetPrimAtPath("/World/Light") if stage is not None else None
    if prim is None or not prim.IsValid():
        print("  !! /World/Light nicht gefunden — Dome-Sweep nicht möglich.")
        return False
    for attr_name in ("inputs:intensity", "intensity"):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            attr.Set(float(value))
            return True
    print(f"  !! /World/Light ohne intensity-Attribut: {list(prim.GetPropertyNames())}")
    return False


def report_frames(obs, out_dir: str, suffix: str = "") -> None:
    """PNGs schreiben und die Bildstatistik selbstbewertend ausgeben."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        print(f"\n[dump] Pillow fehlt ({e}) — keine PNGs.", flush=True)
        return
    # Bildstatistik direkt mitausgeben. Diese Kennzahlen haben am 2026-08-08 die
    # Diagnose getragen, waehrend die PNGs allein nur "sieht falsch aus" sagten:
    #   dunkel% = Anteil Pixel unter Helligkeit 100. Ein korrekter Frame (Referenz:
    #     Simulation/old_videos/12/_debug_obs_cam_left_high.png) hat 11-22 % — der
    #     Hintergrund ist dort dunkel. 0 % heisst: alles weiss, kein Kontrast.
    #   chroma = mittleres max(RGB)-min(RGB). Referenz 5.7-6.0 (farbige Wuerfel).
    #   min/max = Wertebereich. Referenz 32-239; ein Bereich wie 246-248 ist leer.
    print()
    print(f"  {'Kamera':<18}{'min':>5}{'median':>8}{'max':>5}{'chroma':>8}{'dunkel%':>9}"
          f"   Referenz Juni: 32/229/239, chroma 6.0, dunkel 11-22%")
    for key in sorted(k for k in (obs or {}) if k.startswith("video.")):
        cam = key.split(".", 1)[1]
        frame = obs[key][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
        Image.fromarray(frame).save(os.path.join(out_dir, f"{cam}{suffix}.png"))
        f = frame.reshape(-1, 3).astype(np.int16)
        chroma = float((f.max(1) - f.min(1)).mean())
        dark = float((f.mean(1) < 100).mean() * 100.0)
        verdict = "" if dark > 5.0 else "   <-- kein Kontrast, Bild praktisch leer"
        print(f"  {cam:<18}{frame.min():5d}{int(np.median(frame)):8d}{frame.max():5d}"
              f"{chroma:8.2f}{dark:8.1f}%{verdict}", flush=True)
    print(f"\n[dump] PNGs in {out_dir}", flush=True)


def main() -> None:
    cfg = G1Dex3BlockstackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    if args.no_dr:
        cfg.dr_enabled = False

    print("=" * 72)
    print(f"KAMERA-POSEN — konfiguriert vs. gerendert   (num_envs={args.num_envs}, "
          f"DR={'aus' if args.no_dr else 'an'})")
    print("=" * 72, flush=True)

    env = G1Dex3BlockstackEnv(cfg)
    env.reset()
    zero = torch.zeros((env.num_envs, 28), dtype=torch.float32, device=env.device)
    obs = None
    for _ in range(max(1, args.settle_steps)):
        obs, _, _, _, _ = env.step(zero)

    origins = env.scene.env_origins.detach().cpu().numpy()
    print("\nEnv-Ursprünge (env_spacing="
          f"{getattr(cfg.scene, 'env_spacing', '?')}):")
    for e, o in enumerate(origins):
        print(f"  env {e}: {np.round(o, 3)}")

    # Szenengeometrie zur Laufzeit — die nie geprüfte Annahme.
    pts = scene_points(env, origins)
    # BEIDE Spalten ausgeben. Vorher stand hier nur die env-relative Position unter der
    # Überschrift "WELTPOSITION" — das hat am 2026-08-08 zu einer kompletten Fehldiagnose
    # geführt ("alle Objekte am Weltursprung"), obwohl die Szene korrekt war. Die
    # Erwartungswerte aus der Config sind env-lokal, die Frustum-Rechnung unten arbeitet
    # dagegen in Weltkoordinaten — wer nur eine Spalte sieht, vergleicht Äpfel mit Birnen.
    print("\nSzenenobjekte zur Laufzeit (env 0):")
    print(f"  {'Objekt':<24}{'welt':<26}{'rel. zum Ursprung':<26}erwartet (env-lokal, Config)")
    expect = {"table": "(0.5, 0.0, 0.435)", "block_0": "(0.34, -0.15, 0.915)",
              "block_1": "(0.36, 0.0, 0.915)", "block_2": "(0.34, 0.15, 0.915)",
              "robot_root": "(0, 0, ~0.85)"}
    for name, p in pts.items():
        rel = p - origins[0]
        print(f"  {name:<24}{str(np.round(p, 3)):<26}"
              f"{str(np.round(rel, 3)):<26}{expect.get(name, '—')}")
    if not pts:
        print("  !! keine Objektposen auslesbar — Attributnamen der Isaac-Lab-Version prüfen.")

    for name, cam in env.cameras.items():
        cam_cfg = getattr(cfg.scene, name, None)
        off = getattr(cam_cfg, "offset", None) if cam_cfg is not None else None
        cfg_pos = np.asarray(off.pos, dtype=float) if off is not None else None
        cfg_rot = np.asarray(off.rot, dtype=float) if off is not None else None

        pos_key, pos_w = first_attr(cam.data, ["pos_w"])
        quat_key, quat_w = first_attr(
            cam.data, ["quat_w_world", "quat_w", "quat_w_ros", "quat_w_opengl"]
        )
        print("\n" + "-" * 72)
        print(f"{name}   (Prim: {getattr(cam_cfg, 'prim_path', '?')})")
        if pos_w is None or quat_w is None:
            print(f"  !! Pose nicht auslesbar. Verfuegbar: {sorted(dir(cam.data))[:40]}")
            continue
        pos_w = pos_w.detach().cpu().numpy()
        quat_w = quat_w.detach().cpu().numpy()
        print(f"  gelesen aus: data.{pos_key} / data.{quat_key}")
        if cfg_pos is not None:
            print(f"  konfiguriert: pos={np.round(cfg_pos, 3)}  rot={np.round(cfg_rot, 4)}")

        for e in range(min(len(pos_w), args.num_envs)):
            rel = pos_w[e] - origins[e]
            line = (f"  env {e}: welt={np.round(pos_w[e], 3)}  "
                    f"rel. zum Ursprung={np.round(rel, 3)}")
            if cfg_pos is not None and off is not None and "ENV_REGEX_NS" in str(
                getattr(cam_cfg, "prim_path", "")
            ):
                d = float(np.linalg.norm(rel - cfg_pos))
                line += f"  Δ zum Offset={d:.3f} m {'OK' if d < 0.02 else '<-- WEICHT AB'}"
            print(line)

        # Blickrichtung. In convention="world" ist die Blickachse +X.
        view = quat_rotate(quat_w[0], [1.0, 0.0, 0.0])
        print(f"  Blickrichtung (env 0, +X): {np.round(view, 3)}  "
              f"Pitch={np.degrees(np.arcsin(np.clip(view[2], -1, 1))):+.1f}°")
        # Roll mit ausgeben. Die Treffer-Matrix unten prueft NUR die Blickachse — ueber die
        # Drehung um diese Achse sagt sie nichts, und ein verdrehtes Bild waere fuer den
        # eingefrorenen Vision-Encoder ein massiver Domain-Shift. 0° = Bild-Oben zeigt
        # nach Welt-+Z. Bei den Handgelenkskameras haengt der Wert von der Armpose ab.
        cam_up = quat_rotate(quat_w[0], [0.0, 0.0, 1.0])
        cam_right = quat_rotate(quat_w[0], [0.0, 1.0, 0.0])
        ref = np.array([0.0, 0.0, 1.0]) - float(np.dot([0.0, 0.0, 1.0], view)) * np.asarray(view)
        if np.linalg.norm(ref) < 1e-6:
            print("  Roll: Blickachse fast senkrecht — Roll undefiniert")
        else:
            ref = ref / np.linalg.norm(ref)
            roll = np.degrees(np.arctan2(float(np.dot(ref, cam_right)),
                                         float(np.dot(ref, cam_up))))
            print(f"  Roll gegen Welt-Oben: {roll:+.1f}°"
                  + ("" if abs(roll) < 2.0 else "   <-- Bild ist gegen die Senkrechte verdreht"))
        if cfg_rot is None:
            continue
        want = quat_rotate(cfg_rot, [1.0, 0.0, 0.0])
        print(f"  SOLL-Richtung (aus der Config): {np.round(want, 3)}")

        # Welche Konvention/Achse trifft die Sollrichtung? Isaac Lab meldet dieselbe
        # physische Orientierung in mehreren Konventionen (world/ros/opengl), und die
        # "Vorwaerts"-Achse ist je Konvention eine andere: world +X, ROS +Z, OpenGL -Z.
        # Zeigt eine ANDERE Kombination als (world, +X) auf das Ziel, dann wendet Isaac Lab
        # unsere rot anders an als gemeint — dann ist `convention=` der Fehler, nicht die Pose.
        axes = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0],
                "-Y": [0, -1, 0], "+Z": [0, 0, 1], "-Z": [0, 0, -1]}
        variants = {}
        for qname in ("quat_w_world", "quat_w_ros", "quat_w_opengl", "quat_w"):
            if hasattr(cam.data, qname):
                variants[qname] = getattr(cam.data, qname).detach().cpu().numpy()[0]
        best = None
        print("  Treffer-Matrix (Winkel zur SOLL-Richtung, kleinster Wert gewinnt):")
        for qname, q in variants.items():
            row, cells = [], []
            for aname, ax in axes.items():
                a = np.degrees(np.arccos(np.clip(np.dot(quat_rotate(q, ax), want), -1, 1)))
                cells.append(f"{aname} {a:6.1f}°")
                row.append((a, qname, aname))
            print(f"    {qname:<14} " + "  ".join(cells))
            cand = min(row)
            best = cand if best is None or cand[0] < best[0] else best
        if best is not None:
            ang, qname, aname = best
            verdict = "OK" if (qname.endswith("world") and aname == "+X" and ang < 2) else \
                      "<-- Konvention passt NICHT zu convention='world'"
            print(f"  bester Treffer: {qname} / {aname} bei {ang:.1f}°   {verdict}")

        # Was MÜSSTE diese Kamera sehen? Deckt der Frustum-Test die Objekte ab, ist es ein
        # reines Render-Problem; deckt er sie nicht ab, stehen Kamera und Szene nicht
        # zueinander — dann ist die Pose zwar "richtig", aber die Szene woanders.
        if pts:
            cw = int(getattr(cam_cfg, "width", 640))
            ch = int(getattr(cam_cfg, "height", 480))
            tangents = camera_tangents(cam, cw, ch)
            if tangents is None:
                print("  !! keine intrinsic_matrices — Sichtfeld aus dem Cfg GESCHÄTZT "
                      "(horizontal_aperture-Annahme), 'IM BILD' entsprechend unsicher.")
            else:
                fov = (2 * np.degrees(np.arctan(tangents[0])),
                       2 * np.degrees(np.arctan(tangents[1])))
                print(f"  Sichtfeld aus der Renderer-Intrinsik: "
                      f"{fov[0]:.1f}° x {fov[1]:.1f}°")
            print("  Sichtbarkeit der Szenenobjekte (u/v normiert, 0 = Bildmitte):")
            for name, u, w, dist, state in frustum_report(
                cam_cfg, pos_w[0], quat_w[0], pts, cw, ch, tangents=tangents,
            ):
                uv = "        —      " if u is None else f"u={u:+6.2f} v={w:+6.2f}"
                print(f"    {name:<24}{uv}  d={dist:5.2f} m  {state}")

    # Standbilder — belegen, was die Kamera wirklich sieht.
    report_frames(obs, args.out_dir)

    # ── Dome-Light-Sweep ──────────────────────────────────────────────────────────
    # Alle Intensitäten im SELBEN Prozess: der App-Start kostet ~90 s, ein Sweep über
    # vier Werte in vier Läufen also 6 Minuten Leerlauf. Hier ist es einmal.
    # Hintergrund: die Juni-Referenzbilder (32/229/239) entstanden unter Isaac Sim 4.x,
    # die gleichmäßig hellen unter 6.0 — bei identischem DomeLightCfg(intensity=2000).
    # Wenn niedrigere Werte Kontrast zurückbringen, ist es die Belichtung und nicht die
    # Szene (die ist ab runs/20260808/09 nachgemessen korrekt: Würfel auf dem Tisch,
    # im Frustum, 0,66–0,71 m vor der Kamera).
    sweep = [v.strip() for v in (args.dome_sweep or "").split(",") if v.strip()]
    if sweep:
        base = os.environ.get("RL_DOME_INTENSITY", "2000 (Default)")
        print("\n" + "=" * 72)
        print(f"DOME-LIGHT-SWEEP — {len(sweep)} Werte, Basis: {base}")
        print("=" * 72, flush=True)
        settle = max(4, min(args.settle_steps, 20))
        for raw in sweep:
            try:
                val = float(raw)
            except ValueError:
                print(f"  !! '{raw}' ist keine Zahl — übersprungen.")
                continue
            if not set_dome_intensity(val):
                break
            for _ in range(settle):
                obs, _, _, _, _ = env.step(zero)
            print(f"\nintensity = {val:g}   ({settle} Steps gerendert)")
            report_frames(obs, args.out_dir, suffix=f"__dome{val:g}")

    env.close()
    print("\n[dump] fertig.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
