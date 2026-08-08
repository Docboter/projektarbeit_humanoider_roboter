#!/usr/bin/env python3
"""Live-Ansicht der Sim im Browser — MJPEG ueber HTTP ("Spur B" des Livestream-Plans).

Zweck: waehrend eines langen Laufs (v. a. RL-Fine-tuning ueber Stunden/Tage) im Browser
zusehen, ohne WebRTC/NVENC. Zustandslos: Tab schliessen und morgen wieder aufmachen
aendert am Lauf nichts, und beliebig viele Zuschauer sind moeglich.
Hintergrund + Abgrenzung zu Spur A: docs/weiterfuehrend/livestream-plan.md §4.

Endpunkte (Default-Port 8900):
    GET /                    HTML-Seite mit allen Kameras + Live-Metriken
    GET /stream.mjpg?cam=X   multipart/x-mixed-replace — der eigentliche Stream
    GET /frame.jpg?cam=X     letztes Einzelbild (fuer Skripte/curl)
    GET /meta.json           {iteration, step, reward_mean, success_rate, fps, ...}

Kostenmodell — warum das im RL-Lauf praktisch gratis ist:
    ALLE Kameras werden ohnehin JEDEN Env-Step gerendert (sie stehen in
    G1Dex3BlockstackEnv.cameras und laufen durch _get_observations/get_obs_batched
    mit). Es kommt also KEIN Render-Pass dazu, nur ein GPU->CPU-Copy (~1 MB je Kamera)
    und die JPEG-Kodierung. Letztere laeuft in einem eigenen Thread und NUR, solange
    tatsaechlich jemand zuschaut (Zaehler `_viewers`).

Kamera-Wahl: Default sind die vier per Overlay kalibrierten POLICY-Kameras
    (cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist) — sie zeigen genau
    die Modell-Eingabe. `cam_scene` ist eine unvalidierte Uebersichtskamera und lieferte
    am 2026-08-08 fast nur Hintergrund; Diagnose dazu: dump_camera_poses.py.

Robustheit: Diese Klasse darf einen mehrstuendigen Lauf niemals abschiessen. Jeder
Fehler (Port belegt, Pillow fehlt, Client bricht ab) fuehrt zu _disable() mit einer
Warnung — der Lauf laeuft ungestoert weiter. Bei enabled=False ist publish() ein
reiner Early-Return, das Verhalten also bit-identisch zu vorher.

Abhaengigkeiten: nur stdlib + Pillow. Pillow ist im Isaac-Sim-Python vorhanden
(harte Abhaengigkeit von torchvision und diffusers) — trotzdem lazy importiert und
mit sauberem Fallback, damit ein fehlendes Wheel den Lauf nicht kostet.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_BOUNDARY = "g1dex3frame"

_PAGE = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 1rem; background: #14161a; color: #e6e8eb;
         font: 14px/1.55 system-ui, -apple-system, sans-serif; }
  h1 { font-size: 1.05rem; margin: 0 0 .8rem; font-weight: 600; }
  #cams { display: flex; flex-wrap: wrap; gap: 1rem; }
  figure { margin: 0; }
  figcaption { font-size: .8rem; color: #9aa3ad; margin-top: .35rem; }
  img { display: block; max-width: 100%; border-radius: 6px; background: #000; }
  table { border-collapse: collapse; margin-top: 1.1rem;
          font-variant-numeric: tabular-nums; }
  td { padding: .15rem 1.1rem .15rem 0; }
  td:first-child { color: #9aa3ad; }
  #warn { color: #e0a030; margin-top: .8rem; min-height: 1.2em; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div id="cams">__FIGURES__</div>
<table id="meta"></table>
<div id="warn"></div>
<script>
async function tick() {
  try {
    const r = await fetch('/meta.json', {cache: 'no-store'});
    const m = await r.json();
    document.getElementById('meta').innerHTML = Object.keys(m).map(
      k => '<tr><td>' + k + '</td><td>' + m[k] + '</td></tr>').join('');
    document.getElementById('warn').textContent = '';
  } catch (e) {
    // Lauf beendet oder Netz weg — Seite bleibt stehen, wir versuchen es weiter.
    document.getElementById('warn').textContent =
      'keine Verbindung zum Lauf (beendet? Netz weg?) — versuche weiter …';
  }
}
tick(); setInterval(tick, 1000);
</script>
</body>
</html>
"""


def _to_uint8_hwc(arr):
    """Torch-Tensor oder numpy-Array -> (H, W, 3) uint8 numpy auf der CPU.

    Akzeptiert RGBA und schneidet den Alpha-Kanal ab. Gibt None zurueck, wenn die
    Form nicht als Bild interpretierbar ist (statt zu werfen — siehe Modul-Docstring).
    """
    if hasattr(arr, "detach"):  # torch.Tensor, ohne torch importieren zu muessen
        arr = arr.detach().cpu().numpy()
    if getattr(arr, "ndim", 0) != 3 or arr.shape[-1] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != "uint8":
        arr = arr.astype("uint8")
    return arr


class _Handler(BaseHTTPRequestHandler):
    """HTTP-Handler. `view` wird per Subklasse beim Serverstart gebunden."""

    view: "LiveView" = None
    server_version = "G1Dex3LiveView/1"

    def log_message(self, *args, **kwargs):
        # HTTP-Zugriffe NICHT ausgeben — sonst flutet jeder Frame das Trainings-Log.
        pass

    def do_GET(self):  # noqa: N802  (stdlib-vorgegebener Name)
        parsed = urlparse(self.path)
        cam = (parse_qs(parsed.query).get("cam") or [None])[0]
        try:
            if parsed.path in ("/", "/index.html"):
                self._send(self.view.index_html().encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/meta.json":
                self._send(json.dumps(self.view.meta()).encode("utf-8"), "application/json")
            elif parsed.path == "/frame.jpg":
                data, _seq = self.view.wait_jpeg(cam, -1, timeout=2.0)
                if data is None:
                    self.send_error(503, "noch kein Frame")
                else:
                    self._send(data, "image/jpeg")
            elif parsed.path == "/stream.mjpg":
                self._stream(cam)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser-Tab zu — normal, kein Fehlerfall

    def _send(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self, cam) -> None:
        view = self.view
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        view.viewer_enter()
        try:
            last = -1
            while not view.stopped():
                data, last = view.wait_jpeg(cam, last, timeout=5.0)
                if data is None:
                    continue  # nur ein Timeout — Verbindung offen halten
                self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            view.viewer_exit()


class LiveView:
    """Frame-Slot + HTTP-Server. Haelt immer nur das JEWEILS LETZTE Bild.

    Kein Puffer, keine Queue: ein langsamer Client kann die Sim damit nicht ausbremsen,
    er sieht schlicht Frames nicht, die inzwischen ueberschrieben wurden.
    """

    def __init__(
        self,
        enabled: bool = False,
        port: int = 8900,
        every_n: int = 1,
        cams=("cam_left_high", "cam_left_wrist"),
        env_index: int = 0,
        quality: int = 75,
        host: str = "0.0.0.0",
        title: str = "G1 DEX3 — Live-Ansicht",
    ):
        self.enabled = bool(enabled)
        self.port = int(port)
        self.every_n = max(1, int(every_n))
        self.cams = [str(c) for c in cams] or ["cam_left_high"]
        self.env_index = int(env_index)
        self.quality = int(quality)
        self.host = host
        self.title = title

        self._count = 0
        self._seq = 0
        self._raw: dict = {}
        self._cond = threading.Condition()

        self._jseq = 0
        self._jpeg: dict = {}
        self._jcond = threading.Condition()

        self._meta: dict = {}
        self._fps = 0.0
        self._last_pub = 0.0
        self._viewers = 0
        self._vlock = threading.Lock()

        self._stop = threading.Event()
        self._httpd = None
        self._warned = False

        if self.enabled:
            self._start()

    # ── Lebenszyklus ─────────────────────────────────────────────────────────
    def _disable(self, reason: str) -> None:
        self.enabled = False
        if not self._warned:
            self._warned = True
            print(f"[live] Live-Ansicht deaktiviert: {reason}", flush=True)

    def _start(self) -> None:
        try:
            from PIL import Image  # noqa: F401  (nur Verfuegbarkeitspruefung)
        except Exception as e:  # pragma: no cover — haengt am Image
            self._disable(f"Pillow (PIL) nicht importierbar: {e}")
            return
        bound = type("_BoundHandler", (_Handler,), {"view": self})
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), bound)
        except OSError as e:
            self._disable(f"Port {self.port} nicht bindbar ({e}) — LIVE_VIEW_PORT setzen.")
            return
        self._httpd.daemon_threads = True
        threading.Thread(
            target=self._httpd.serve_forever, name="live-view-http", daemon=True
        ).start()
        threading.Thread(target=self._encode_loop, name="live-view-jpeg", daemon=True).start()
        print(
            f"[live] Live-Ansicht aktiv auf Port {self.port} "
            f"(Kameras: {', '.join(self.cams)}, jedes {self.every_n}. Frame)",
            flush=True,
        )
        print(
            f"[live]   Browser:    http://<server-ip>:{self.port}/\n"
            f"[live]   nur SSH?    ssh -L {self.port}:localhost:{self.port} <server> "
            f"→ http://localhost:{self.port}/",
            flush=True,
        )

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        with self._jcond:
            self._jcond.notify_all()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        self.enabled = False

    def stopped(self) -> bool:
        return self._stop.is_set()

    # ── Publisher-Seite (wird aus der Rollout-Schleife gerufen) ──────────────
    def _due(self) -> bool:
        """Zaehlt den Aufruf und sagt, ob dieser Frame dran ist (LIVE_VIEW_EVERY_N)."""
        self._count += 1
        return (self._count - 1) % self.every_n == 0

    def publish(self, frames, **meta) -> None:
        """Frame(s) veroeffentlichen. `frames`: Array/Tensor oder {name: Array/Tensor}."""
        if not self.enabled or not self._due():
            return
        self._store(frames, meta)

    def publish_obs(self, obs, **meta) -> None:
        """Bequemer Weg aus dem Env-Obs-Dict: nimmt video.<cam> fuer env_index.

        Ist keine der konfigurierten Kameras vorhanden, wird einmalig auf die erste
        beliebige video.*-Kamera ausgewichen (und gewarnt) — eine falsch gesetzte
        LIVE_VIEW_CAMS soll nicht in einer schwarzen Seite enden.
        """
        if not self.enabled or not self._due():
            return
        frames = {}
        for cam in self.cams:
            t = obs.get(f"video.{cam}")
            if t is not None:
                frames[cam] = t[self.env_index]
        if not frames:
            fallback = [k for k in obs if k.startswith("video.")]
            if not fallback:
                self._disable(f"keine video.*-Kamera im Obs-Dict (vorhanden: {sorted(obs)})")
                return
            if not self._warned:
                self._warned = True
                print(
                    f"[live] Kameras {self.cams} nicht im Obs — weiche auf "
                    f"'{fallback[0]}' aus. Vorhanden: {sorted(fallback)}",
                    flush=True,
                )
            self.cams = [fallback[0].removeprefix("video.")]
            frames = {self.cams[0]: obs[fallback[0]][self.env_index]}
        self._store(frames, meta)

    def update_meta(self, **meta) -> None:
        """Metriken aktualisieren ohne neues Bild (z. B. am Iterations-Ende)."""
        if not self.enabled:
            return
        with self._cond:
            self._meta.update(meta)

    def _store(self, frames, meta: dict) -> None:
        try:
            if not isinstance(frames, dict):
                frames = {self.cams[0]: frames}
            raw = {}
            for name, arr in frames.items():
                conv = _to_uint8_hwc(arr)
                if conv is not None:
                    raw[name] = conv
            if not raw:
                return
            now = time.monotonic()
            with self._cond:
                if self._last_pub:
                    dt = now - self._last_pub
                    if dt > 0:
                        inst = 1.0 / dt
                        self._fps = inst if self._fps == 0.0 else 0.85 * self._fps + 0.15 * inst
                self._last_pub = now
                self._raw = raw
                self._seq += 1
                self._meta.update(meta)
                self._meta["frames_pro_s"] = round(self._fps, 2)
                self._cond.notify_all()
        except Exception as e:  # pragma: no cover — darf den Lauf nie abschiessen
            self._disable(f"publish() fehlgeschlagen: {e}")

    # ── Encoder-Thread ───────────────────────────────────────────────────────
    def _encode_loop(self) -> None:
        from io import BytesIO

        from PIL import Image

        last = 0
        while not self._stop.is_set():
            with self._cond:
                while self._seq == last and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                last = self._seq
                raw = self._raw
            # Nur kodieren, wenn jemand zuschaut. Das allererste Frame immer, damit ein
            # frisch verbundener Client sofort etwas sieht statt auf den naechsten Step
            # zu warten (bei langsamen Iterationen sonst spuerbar).
            with self._vlock:
                viewers = self._viewers
            if viewers <= 0 and self._jseq > 0:
                continue
            try:
                jpeg = {}
                for name, arr in raw.items():
                    buf = BytesIO()
                    Image.fromarray(arr).save(buf, format="JPEG", quality=self.quality)
                    jpeg[name] = buf.getvalue()
            except Exception as e:  # pragma: no cover
                self._disable(f"JPEG-Kodierung fehlgeschlagen: {e}")
                return
            with self._jcond:
                self._jpeg = jpeg
                self._jseq += 1
                self._jcond.notify_all()

    # ── Consumer-Seite (HTTP-Threads) ────────────────────────────────────────
    def viewer_enter(self) -> None:
        with self._vlock:
            self._viewers += 1

    def viewer_exit(self) -> None:
        with self._vlock:
            self._viewers = max(0, self._viewers - 1)

    def wait_jpeg(self, cam, last_seq: int, timeout: float = 5.0):
        """Blockiert bis ein Frame neuer als last_seq da ist. -> (bytes|None, seq)."""
        with self._jcond:
            if self._jseq == last_seq:
                self._jcond.wait(timeout=timeout)
            if self._jseq == last_seq:
                return None, last_seq
            jpeg, seq = self._jpeg, self._jseq
        name = cam if cam in jpeg else next((c for c in self.cams if c in jpeg), None)
        if name is None:
            name = next(iter(jpeg), None)
        return (jpeg.get(name) if name else None), seq

    def meta(self) -> dict:
        with self._cond:
            data = dict(self._meta)
        with self._vlock:
            data["zuschauer"] = self._viewers
        return data

    def index_html(self) -> str:
        figures = "".join(
            f'<figure><img src="/stream.mjpg?cam={c}" alt="{c}">'
            f"<figcaption>{c}</figcaption></figure>"
            for c in self.cams
        )
        return _PAGE.replace("__TITLE__", self.title).replace("__FIGURES__", figures)
