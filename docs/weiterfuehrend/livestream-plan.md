# Implementierungsplan — Live-Ansicht der Isaac-Lab-Sim

**Status:** **Spur B für den RL-Lauf ist gebaut** ([`live_view.py`](../../Simulation/g1_dex3_sim/live_view.py),
2026-08-08) · Spur A (WebRTC) unverändert offen und **nie auf Hardware getestet** ·
**Erstellt:** 2026-06-02 · **Revision v2:** 2026-08-07 · **v3:** 2026-08-08

> ## Umsetzungsstand
>
> | Teil | Stand |
> |---|---|
> | **Spur B — Frame-Stream, RL-Pfad** | ✅ **umgesetzt.** `live_view.py` + Hook in der Rollout-Schleife von [`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py), `LIVE_VIEW*`-Env-Vars, Port-Mapping in [`server_rl_run.sh`](../../Simulation/server_rl_run.sh). Modul isoliert getestet (Index, `meta.json`, `frame.jpg`, MJPEG-Strom, Kamera-Fallback, belegter Port). **Noch nicht auf der Server-GPU im echten Lauf gesehen.** |
> | **Option C — W&B-Video** | ✅ **umgesetzt.** `RL_WANDB_VIDEO_EVERY=N`, mit Fallback auf einen Bild-Filmstreifen (kein `moviepy` im Kit-Python). |
> | **Spur B — Sim-Eval + Baseline-Eval** | ⬜ offen. `live_view.py` ist lauf-agnostisch, es fehlt nur der Hook (§4.2). |
> | **Spur A — WebRTC** | ⬜ offen, bewusst zurückgestellt (D1–D4, D6). Begründung: §1 — für einen tagelangen RL-Lauf ist der Viewport der falsche Mechanismus, und er trägt das gesamte Isaac-Sim-6.0-Risiko. |
>
> Vorgehen bewusst so geschnitten: Beobachtbarkeit **vor** dem ersten großen RL-Lauf, weil sie
> sich nicht nachrüsten lässt — und mit dem billigsten Teil, nicht dem attraktivsten.

> **Ziel:** Das Live-Äquivalent zu den heutigen MP4s aus `/data/sim_videos`. Statt nach dem
> Lauf per `docker cp` Videos zu holen, soll der Roboter **während** des Laufs im Browser
> oder in einer Desktop-App beobachtbar sein — für Sim-Eval, Baseline-Eval **und** den
> langen RL-Lauf.

---

## 0. Was sich seit v1 geändert hat

v1 (§9 unten, weiterhin gültig für vast.ai) plante ausschließlich den **WebRTC-Viewport auf
einer vast.ai-Instanz**. Drei Dinge haben sich seither verschoben:

| Änderung | Konsequenz für den Livestream |
|---|---|
| **Primäre Plattform ist jetzt `ikr-ki-server-01`** (2× RTX PRO 6000 Blackwell, Treiber 610.43.02), erreichbar **direkt im Netz/VPN** — siehe [`server_rl_run.sh`](../../Simulation/server_rl_run.sh) | Das vast.ai-Kernproblem (zufälliges Port-Mapping ⇄ SDP-eingebetteter Port) **entfällt vollständig**. Ports sind frei wählbar, UDP funktioniert. Das ist die mit Abstand günstigste Ausgangslage für WebRTC. |
| **Isaac-Sim-6.0-Port** (Basis-Image `isaac-lab:3.0.0-beta2-post1` statt `2.3.2`), erzwungen durch den Blackwell-Segfault unter Treiber 610.x | Der Browser-Client auf **Port 8211 existiert in Isaac Sim 6.0 nicht mehr**; die Kit-Settings-Pfade haben sich geändert. Unser bestehender Livestream-Code zielt noch auf 4.5/5.x → **§2 Defekte**. |
| **Drei Lauf-Typen** sollen live beobachtbar sein: RL-Fine-tuning, Sim-Eval, Baseline-Eval | RL hat **keinerlei** Livestream-Logik ([`rl_finetune.py:150`](../../Simulation/g1_dex3_sim/rl_finetune.py#L150) hardcodet `AppLauncher(headless=True, …)`), und ein tagelanger RL-Lauf passt schlecht zum WebRTC-Modell („nur ein Client, keine Reconnect-Semantik"). → **Zwei Spuren, §1.** |

**Hardware-Voraussetzung ist erfüllt:** WebRTC braucht NVENC; die RTX PRO 6000 Blackwell hat
**vier NVENC-Engines (9. Generation)**. Die Sim-GPU-Regel (Ampere+ mit RT-Cores) und die
Streaming-Regel (NVENC) schließen weiterhin beide dieselben GPUs aus — A100/H100.
`NVIDIA_DRIVER_CAPABILITIES=all` ist in [`Dockerfile.vastai`](../../Simulation/Dockerfile.vastai)
bereits gesetzt und enthält `video` (NVENC) — kein Zusatz-Install nötig.

---

## 1. Zwei Spuren — und welche wofür

Der Viewport-Stream und ein leichtgewichtiger Frame-Stream lösen **nicht dasselbe Problem**.
Der Plan baut beide, aber mit klarer Zuständigkeit:

| | **Spur A — WebRTC-Viewport** | **Spur B — Frame-Stream** |
|---|---|---|
| **Was man sieht** | Vollständiger Isaac-Sim-Viewport: freie Kamera, Szene drehen/zoomen, Isaac-Sim-UI | Die gerenderten Kamera-Bilder (`cam_scene` + optional die 4 Policy-Kameras) + Live-Metriken |
| **Technik** | NVENC-Hardware-Encoder, WebRTC (TCP 49100 Signaling + **UDP 47998** Medien) | MJPEG über HTTP (`multipart/x-mixed-replace`), Python-stdlib, keine neue Dependency |
| **Client** | Native **Isaac Sim WebRTC Streaming Client** (Desktop) oder Web-Viewer (Docker Compose, Port 8210) | **Jeder Browser**, URL direkt öffnen |
| **Zuschauer** | genau **1** gleichzeitig | beliebig viele, jederzeit rein-/rausklinken |
| **Reconnect** | Session-gebunden, Abbruch = neu verbinden | zustandslos — Tab neu laden genügt |
| **Über SSH-Tunnel** | ✗ (UDP-Medien; TCP-only wird nicht unterstützt) | ✓ (reines HTTP) |
| **Kosten im Lauf** | zusätzlicher Viewport-Render-Pfad + Encode | ~0 — die Frames werden **ohnehin schon** gerendert (Policy-Obs/Video) |
| **Risiko** | mittel–hoch (Isaac-Sim-6.0-Port unverifiziert, Kit-Settings geändert) | niedrig (kein Isaac-Sim-Feature involviert) |

**Empfohlene Zuordnung:**

- **Sim-Eval + Baseline-Eval → Spur A** (WebRTC). Episodische Läufe, man schaut gezielt zu,
  freie Kamera ist beim Debuggen von Greifposen echtes Gold wert.
- **RL-Fine-tuning → Spur B** (Frame-Stream). Läuft Stunden bis Tage, soll nebenbei im Tab
  offen liegen, muss Netzabbrüche überleben, und mehrere Leute wollen draufschauen.
  Ein WebRTC-Viewport über 48 h zu halten ist der falsche Mechanismus.
- **Spur B ist gleichzeitig der Fallback für alles**, falls Spur A am Isaac-Sim-6.0-Port
  scheitert. Deshalb wird sie **zuerst** gebaut (§6, Phase 1) — danach existiert unabhängig
  vom WebRTC-Ausgang eine funktionierende Live-Ansicht.

---

## 2. Befund am bestehenden Code — 6 konkrete Defekte

Der v1-Code in [`entrypoint_sim.sh`](../../Simulation/scripts/entrypoint_sim.sh#L270-L296) und
[`entrypoint_baseline.sh`](../../Simulation/scripts/entrypoint_baseline.sh#L204-L210) ist
strukturell richtig (opt-in, `--headless` korrekt weggelassen wegen
[IsaacLab#381](https://github.com/isaac-sim/IsaacLab/issues/381)), hat aber sechs Punkte, die
vor dem ersten Test korrigiert bzw. verifiziert werden müssen:

| # | Defekt | Datei/Zeile | Fix |
|---|---|---|---|
| **D1** | **Browser-URL zeigt auf Port 8211** — in Isaac Sim 6.0 gibt es diesen Client nicht mehr. Der Nutzer bekommt eine URL, die garantiert ins Leere läuft. | [`entrypoint_sim.sh:286`](../../Simulation/scripts/entrypoint_sim.sh#L286) | Ausgabe auf **nativen WebRTC-Client** (`<IP>:49100`) umstellen; Web-Viewer (Port 8210) nur nennen, wenn er separat deployt wurde. |
| **D2** | **Kit-Settings-Pfad veraltet.** Wir setzen `--/app/livestream/port` + `--/app/livestream/publicEndpointAddress`. Isaac Sim 6.0 dokumentiert `--/exts/omni.kit.livestream.app/primaryStream/{signalPort,streamPort,publicIp}`. | [`entrypoint_sim.sh:277-280`](../../Simulation/scripts/entrypoint_sim.sh#L277-L280) | Im Container die tatsächlich akzeptierten Settings prüfen (Phase 0) und den Pfad versionsabhängig setzen. Falsche Kit-Settings werden von Kit **still ignoriert** — das wird sonst zur stundenlangen Fehlersuche. |
| **D3** | **Doppelte Port-Belegung.** Der Isaac-Lab-`AppLauncher` injiziert bei `livestream=1` selbst `--/app/livestream/port=49100`; unser `--kit_args` hängt einen zweiten, ggf. abweichenden Port an. Welcher gewinnt, hängt von der argv-Reihenfolge ab. | [`entrypoint_sim.sh:277`](../../Simulation/scripts/entrypoint_sim.sh#L277) | Auf dem Server `LIVESTREAM=2` (privat) nutzen — dort injiziert der AppLauncher **keinen** Port und der Konflikt entfällt. Abweichende Ports nur für den vast.ai-Sonderfall. |
| **D4** | **`curl ifconfig.me` läuft auch bei `LIVESTREAM=2`** (privates Netz), wo `PUBLIC_IP` bedeutungslos ist — im Institutsnetz ggf. ein 10-s-Timeout beim Start. | [`entrypoint_sim.sh:109-110`](../../Simulation/scripts/entrypoint_sim.sh#L109-L110) | Nur bei `LIVESTREAM=1` ausführen. |
| **D5** | ~~**RL kennt keinen Livestream.**~~ **Für Spur B erledigt (2026-08-08):** Publish-Hook in der Rollout-Schleife, `LIVE_VIEW*` als argparse-Defaults aus der Umgebung. `AppLauncher(headless=True, enable_cameras=True)` bleibt hartkodiert — für den Frame-Stream ist das genau richtig. | [`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py) | Spur A (RL im WebRTC-Viewport) bliebe offen: dafür müsste `AppLauncher` zusätzlich `livestream=` bekommen — **niedrige Priorität**, siehe §1. |
| **D6** | **Kein Server-Launcher für die Sim-Eval.** Es gibt [`server_rl_run.sh`](../../Simulation/server_rl_run.sh) und `server_robocasa_ref_run.sh`, aber **kein** `server_sim_run.sh` — die Eval hat auf `ikr-ki-server-01` also noch gar keinen Startweg, geschweige denn Port-Publishing. | — | Neues Skript `Simulation/server_sim_run.sh` nach dem Muster von `server_rl_run.sh` (Workbench-Container, `docker exec`), inkl. `-p`-Mappings. |

---

## 3. Spur A — WebRTC-Viewport

### 3.1 Funktionsweise, Ports, Clients (Isaac Sim 6.0)

| Port | Protokoll | Zweck | Pflicht? |
|---|---|---|---|
| **49100** | TCP | WebRTC-**Signaling** | ✅ |
| **47998** | UDP | WebRTC-**Medienstrom** | ✅ — TCP-only wird **nicht** unterstützt |
| 8210 | TCP | Web-Viewer (nur bei separatem Docker-Compose-Deployment, Ubuntu-only) | optional |
| ~~8211~~ | — | Browser-Client aus Isaac Sim ≤5.x — **in 6.0 entfallen** | ✗ |

`LIVESTREAM`-Werte (Isaac Lab 2.x/3.x — in Isaac Lab 1.x bedeutete `1` noch den heute
deprecateten Native-Client, deshalb kursieren widersprüchliche Tabellen im Netz):

| Wert | Bedeutung | AppLauncher-Verhalten |
|---|---|---|
| `0` | aus (Default) | — |
| `1` | WebRTC über **öffentliches** Netz | setzt `publicEndpointAddress=$PUBLIC_IP` + `port=49100`, aktiviert `omni.services.livestream.nvcf` |
| `2` | WebRTC über **lokales/privates** Netz | aktiviert `omni.services.livestream.nvcf`, **ohne** Endpunkt-/Port-Injektion |

Jeder Wert ≠ 0 erzwingt Headless — `--headless` darf **nicht zusätzlich** gesetzt werden.
`--enable_cameras` bleibt Pflicht (5 Kameras in der Szene).

### 3.2 Netzkonfiguration auf `ikr-ki-server-01`

Da der Server **direkt im Netz/VPN** erreichbar ist, ist die Konfiguration denkbar simpel —
kein `PUBLIC_IP`, kein Port-Rätselraten:

```bash
LIVESTREAM=2                 # privates Netz
# Container-Start (Ergänzung in server_sim_run.sh / server_rl_run.sh):
docker run … -p 49100:49100/tcp -p 47998:47998/udp …
```

Client: **Isaac Sim WebRTC Streaming Client** (native Desktop-App für Windows/macOS/Linux,
von der Isaac-Sim-Downloadseite) → Server-Adresse `<server-ip>:49100`. Läuft ohne lokale
GPU-Anforderung.

⚠️ Firewall: UDP 47998 muss zwischen Arbeitsrechner und Server **offen** sein. Das ist der
wahrscheinlichste Stolperstein im Institutsnetz und wird in Phase 0 zuerst geprüft
(`nc -u -z <ip> 47998` bzw. `iperf3 -u`).

### 3.3 Code-Änderungen

1. **[`entrypoint_sim.sh`](../../Simulation/scripts/entrypoint_sim.sh)** — D1–D4 beheben:
   Kit-Settings versionsabhängig, `curl` nur bei Modus 1, Client-Hinweis auf den nativen
   Client, Warn-Text „vast.ai" nur ausgeben, wenn tatsächlich vast.ai (z. B. via neuer Var
   `PLATFORM=server|vastai`).
2. **[`entrypoint_baseline.sh`](../../Simulation/scripts/entrypoint_baseline.sh)** — identische
   Anpassung (die Blöcke sind fast wortgleich; ggf. in ein gemeinsames `lib_livestream.sh`
   ausklammern, das beide Entrypoints sourcen — vermeidet, dass die Zwillinge auseinanderlaufen).
3. **Neu: [`Simulation/server_sim_run.sh`](../../Simulation/server_sim_run.sh)** (D6) — Workbench-Container
   nach dem Muster von `server_rl_run.sh`, mit `-p 49100:49100 -p 47998:47998/udp` und
   Durchreichen von `LIVESTREAM`, `NUM_EPISODES`, `CHECKPOINT_PATH`, `ASSET_PATH`.
4. **[`Dockerfile.vastai`](../../Simulation/Dockerfile.vastai)** — `EXPOSE 49100 8210` +
   `EXPOSE 47998/udp` ergänzen (Dokumentationswert), `LIVESTREAM_PORT`-Default beibehalten.
5. **Optional, niedrige Priorität — Echtzeit-Pacing:** Die Eval läuft so schnell wie möglich;
   zum Zuschauen kann ein `--realtime`-Flag pro Step auf `dt` drosseln
   ([`run_g1_dex3_sim_eval.py`](../../Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py), Rollout-Schleife).
   Erst nach funktionierendem Stream angehen.

### 3.4 vast.ai-Sonderfall

Auf vast.ai bleibt die v1-Logik nötig und unverändert gültig: `LIVESTREAM=1`, `PUBLIC_IP` via
`ifconfig.me`, und `LIVESTREAM_PORT` **auf den extern gemappten Port setzen** (intern == extern),
weil WebRTC den Port in die SDP-Verhandlung einbettet. Docker-Options der Instanz:
`--ipc=host --shm-size=16g -p 22 -p 49100 -p 47998/udp`. Das UDP-Mapping ist dort der
Hauptrisikofaktor — Spur B ist auf vast.ai daher der verlässlichere Weg.

---

## 4. Spur B — Frame-Stream (`live_view.py`)

### 4.1 Architektur

```
 Sim-Prozess (Isaac Lab)                          Browser (beliebig viele)
 ┌───────────────────────────────┐
 │ Rollout-Schleife              │
 │   obs = env.step(action)      │
 │   frame = obs["video.cam_…"]  │
 │   live.publish(frame, meta) ──┼──► LiveView (Hintergrund-Thread)
 └───────────────────────────────┘         │  http.server, stdlib
                                           │
                                   GET /            → HTML-Seite
                                   GET /stream.mjpg → multipart/x-mixed-replace  ──► <img src>
                                   GET /meta.json   → {step, episode, success, reward, fps}
```

**Bewusst minimal:** `http.server.ThreadingHTTPServer` + `imageio`/`PIL` für die JPEG-Kodierung
— beides ist im Image bereits vorhanden (`imageio` wird für die MP4s genutzt). **Keine neue
Dependency, kein Flask, kein WebSocket-Stack.** Der Publisher hält nur das *jeweils letzte*
Frame (Slot, kein Puffer) — langsame Clients bremsen die Sim damit nicht aus.

Neues Modul **`Simulation/g1_dex3_sim/live_view.py`**:

```python
class LiveView:
    def __init__(self, enabled: bool, port: int = 8900, every_n: int = 1): ...
    def publish(self, frame: np.ndarray, **meta) -> None: ...   # No-Op wenn disabled
    def close(self) -> None: ...
```

Ist `LIVE_VIEW=0`, ist `publish()` ein reiner Early-Return — das Default-Verhalten aller drei
Läufe bleibt bit-identisch.

### 4.2 Hook-Punkte (drei Stellen, je 1–3 Zeilen)

| Lauf | Datei / Zeile | Frame-Quelle |
|---|---|---|
| Sim-Eval | [`run_g1_dex3_sim_eval.py:187-189`](../../Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py#L187-L189) — direkt neben `frames.append(frame)` | `obs_step["video.cam_scene"]` (bereits gerendert, Fallback `cam_left_high`) |
| Baseline-Eval | derselbe Codepfad (nutzt dasselbe Eval-Skript) | dito |
| RL | Rollout-Schleife in [`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py#L230-L290), zusätzlich Metriken am Iterations-Ende (`reward_mean`, `success`, [Zeile 283](../../Simulation/g1_dex3_sim/rl_finetune.py#L283)) | `obs[f"video.{cam}"][0]` — Env 0 der vektorisierten Envs |

**Kostenpunkt — die offene Frage ist beantwortet (2026-08-08, Code-Befund):** `cam_scene` **wird
im RL-Lauf gerendert**. Sie steht in
[`G1Dex3BlockstackEnv.cameras`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L351)
(mit dem Kommentar „nur fürs Video"), `_get_observations()` iteriert über *alle* Einträge dieses
Dicts, und `get_obs_batched()` reicht sie als `video.cam_scene` an den RL-Trainer durch. Der
Rollout hat das Bild also ohnehin in der Hand — **kein zusätzlicher Render-Pass**, nur ein
GPU→CPU-Copy (~1 MB) und die JPEG-Kodierung. Letztere läuft in einem eigenen Thread und nur,
solange tatsächlich jemand zuschaut. `LIVE_VIEW_EVERY_N` (z. B. 2–5) drosselt zusätzlich.

Nebenbefund: Die Policy selbst nutzt `cam_scene` **nicht** (nur die vier Policy-Kameras gehen an
GR00T). Wer den Render-Durchsatz optimieren will, könnte sie im RL-Pfad also abschalten — dann
verschwindet aber genau dieses Gratis-Bild. Erst messen (Gruppe 0 im RL-Plan), dann entscheiden.

### 4.3 Neue Env-Vars

| Variable | Default | Zweck |
|---|---|---|
| `LIVE_VIEW` | `0` | `1` = Frame-Stream aktiv |
| `LIVE_VIEW_PORT` | `8900` | HTTP-Port des Frame-Streams |
| `LIVE_VIEW_EVERY_N` | `1` | nur jedes n-te Frame publizieren (RL-Drosselung) |
| `LIVE_VIEW_CAMS` | `cam_scene` | kommagetrennt; mehrere Kameras nebeneinander auf der Seite |
| `RL_WANDB_VIDEO_EVERY` | `0` | Option C (§5): alle N Iterationen einen Rollout ins W&B-Dashboard |

Alle fünf sind umgesetzt und stehen in der zentralen Referenz
[`docs/training/env-vars.md`](../training/env-vars.md). Container-Start ergänzen um
`-p 8900:8900` — [`server_rl_run.sh`](../../Simulation/server_rl_run.sh) macht das beim Anlegen
selbst und warnt, wenn ein älterer Container das Mapping nicht hat. Aufruf:
`http://<server-ip>:8900/` — und, falls mal nur SSH geht, `ssh -L 8900:localhost:8900 <server>`
und dann `http://localhost:8900/`.

**Umsetzungsdetail, das Zeit spart:** Die `LIVE_VIEW*`-Vars liest
[`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py) **selbst** als argparse-Defaults,
statt sie über `entrypoint_rl.sh` als CLI-Flags durchzureichen. Grund: `server_rl_run.sh` mountet
`g1_dex3_sim` live in den Container, `/scripts` dagegen nicht — so wirkt der Schalter ohne
Image-Rebuild (~30 min).

### 4.4 Warum Spur B für RL die richtige ist

Ein RL-Lauf dauert Stunden bis Tage. Der WebRTC-Viewport erlaubt **einen** Client, überlebt
keinen Netzabbruch sauber und hält währenddessen einen NVENC-Encode-Pfad offen. Der
Frame-Stream ist zustandslos: Tab zu, Laptop zu, morgen wieder auf — der Lauf merkt nichts
davon. Zusätzlich lässt sich auf derselben Seite direkt zeigen, was beim RL interessiert
(Reward-Kurve, Success-Rate, Iteration) — Bild **und** Zahlen an einer Stelle.

---

## 5. Option C — W&B als Zero-Effort-Semi-Live (nur RL) — ✅ umgesetzt

`RL_WANDB_VIDEO_EVERY=N` schneidet alle N Iterationen den Rollout von Env 0 mit und hängt ihn an
denselben `wandb.log()`-Aufruf wie `reward_mean`/`success_rate` (bewusst derselbe Aufruf — ein
zweiter würde einen eigenen W&B-Step erzeugen und Video und Metriken versetzt ablegen).

Das ist **nicht live** (Verzögerung = eine Iteration), dafür aber **bleibend**: im Dashboard
abrufbar, von überall erreichbar, kein Port, keine Firewall — und damit in der Projektarbeit
zitierbar. Genau darin ergänzt es den MJPEG-Stream, der flüchtig ist.

**Stolperstein, den die Umsetzung umgeht:** `wandb.Video` braucht für numpy-Eingaben `moviepy`,
und das ist im Isaac-Sim-Kit-Python **nicht** installiert. Statt dafür eine Dependency ins Image
zu ziehen, fällt `_rollout_video_payload()` auf einen Filmstreifen aus acht Einzelbildern zurück
(`wandb.Image` braucht nur Pillow) und protokolliert das. Ein fehlendes Wheel kostet damit
höchstens Komfort, nie den Lauf.

---

## 6. Umsetzungsreihenfolge

| Phase | Inhalt | Aufwand | Abbruch-/Weiter-Kriterium |
|---|---|---|---|
| **0 — Machbarkeit** | Auf `ikr-ki-server-01`: UDP 47998 zwischen Arbeitsplatz und Server prüfen; im Container `omni.services.livestream.nvcf` + akzeptierte Kit-Settings-Pfade verifizieren (D2); NVENC im Container prüfen (`nvidia-smi -q -d ENCODER`) | ~1–2 h | ⬜ offen — betrifft nur noch Spur A |
| **1 — Spur B bauen** ✅ | `live_view.py` (stdlib + Pillow, Latest-Frame-Slot, Encoder-Thread, MJPEG/`meta.json`/`frame.jpg`) — isoliert getestet inkl. Kamera-Fallback und belegtem Port | erledigt 2026-08-08 | ✅ |
| **2 — Spur B auf RL** ✅ | Hook in der Rollout-Schleife von `rl_finetune.py` + Metriken je Iteration, `LIVE_VIEW*` als Env-Defaults, `-p 8900` in `server_rl_run.sh`, Option C (§5) | erledigt 2026-08-08 | ✅ — offen bleibt die Messung von `LIVE_VIEW_EVERY_N` im echten Lauf |
| **2b — Spur B auf Sim-/Baseline-Eval** | Hook an derselben Stelle wie `frames.append(frame)` (§4.2); `live_view.py` ist lauf-agnostisch, es fehlt nur der Aufruf | ~1 h | ⬜ offen — nicht nötig für den RL-Lauf |
| **3 — Spur A reparieren** | D1–D4 in beiden Entrypoints; `server_sim_run.sh` (D6) mit Port-Mappings | ~3–4 h | ⬜ offen |
| **4 — Spur A testen** | `NUM_EPISODES=2`, `LIVESTREAM=2`, nativer WebRTC-Client → Viewport sichtbar und flüssig? | ~2 h | ⬜ offen. Bei Fehlschlag: Isaac-Sim-6.0-Web-Viewer (Port 8210, Docker Compose) als zweiter Versuch; sonst bleibt Spur B die Lösung |
| **5 — Doku** ✅ (RL-Teil) | [`rl-anleitung.md`](rl-anleitung.md) Schritt 6, [`env-vars.md`](../training/env-vars.md), [`CLAUDE.md`](../../CLAUDE.md). Offen: [`vastai-anleitung.md`](../simulation/vastai-anleitung.md) + [`umsetzungsnotizen.md`](../simulation/umsetzungsnotizen.md) (gehören zu Phase 2b/3) | erledigt 2026-08-08 | ✅ |

**Reihenfolge-Begründung:** Spur B zuerst, obwohl der Viewport das attraktivere Ziel ist —
weil Phase 1+2 mit hoher Sicherheit funktionieren und danach *unabhängig vom Isaac-Sim-6.0-Risiko*
eine Live-Ansicht existiert. Phase 0 ist trotzdem ganz vorne, weil ihr Ergebnis darüber
entscheidet, ob Phase 3/4 überhaupt sinnvoll sind.

**Abweichung vom Plan (2026-08-08):** Umgesetzt wurde nur der RL-Strang (Phase 1 + 2 + Doku),
nicht Phase 1's ursprünglicher Sim-Eval-Einstieg. Auslöser war der anstehende erste große RL-Lauf:
Beobachtbarkeit lässt sich in einen bereits laufenden Mehrstunden-Job nicht nachrüsten, und
`cam_scene` liegt im RL-Rollout ohnehin gerendert vor (§4.2). Aus den geplanten ~12–15 h wurden
so ~2 h für den Teil, der vor dem Lauf tatsächlich gebraucht wird.

**Image-Rebuild:** Alle Skripte unter `Simulation/scripts/` und `g1_dex3_sim/` werden ins Image
**kopiert**, nicht gemountet → nach jeder Änderung `./Simulation/update_sim_image.sh --vastai`.
Für die Iteration in Phase 1–2 lohnt ein Bind-Mount (`-v $(pwd)/Simulation/g1_dex3_sim:/workspace/g1_dex3_sim`).

---

## 7. Test- und Validierungsplan

### 7.0 Bereits abgehakt (2026-08-08, lokal ohne GPU)

`live_view.py` hängt an nichts Isaac-spezifischem und ließ sich deshalb komplett auf dem
Arbeitsrechner prüfen — mit synthetischen Frames statt Sim-Bildern:

| Geprüft | Ergebnis |
|---|---|
| `enabled=False` ist ein reiner No-Op | kein Server, kein Thread, keine Exception |
| `GET /` liefert die Seite mit `<img src="/stream.mjpg?cam=…">` je Kamera | ✅ |
| `GET /meta.json` enthält die gemergten Metriken + `frames_pro_s` | ✅ |
| `GET /frame.jpg` ist ein gültiges JPEG in Bildgröße, RGBA→RGB gestutzt | ✅ (64×48, `mode=RGB`) |
| `GET /stream.mjpg` liefert Boundary + JPEG-Teile, während nebenher publiziert wird | ✅ |
| falsch gesetzte `LIVE_VIEW_CAMS` | weicht mit Warnung auf eine vorhandene Kamera aus |
| Port bereits belegt | deaktiviert sich mit Meldung, wirft nicht |
| `LIVE_VIEW*` als argparse-Defaults (an/aus/Müllwerte/CLI schlägt Env) | ✅ |

**Was das nicht zeigt:** dass Pillow im Isaac-Sim-Kit-Python vorhanden ist (dort nicht explizit
installiert, kommt als harte Abhängigkeit von `torchvision` und `diffusers` — deshalb der
Fallback statt einer Annahme), und dass echte `cam_scene`-Bilder wie erwartet aussehen. Beides
klärt Punkt 3 unten auf der Server-GPU.

### 7.1 Auf Hardware zu prüfen

1. **Regression (beide Spuren):** `LIVESTREAM=0` + `LIVE_VIEW=0` → exakt das alte Verhalten
   (headless, MP4s in `/data/sim_videos`, `results.json`). Das ist das wichtigste Kriterium.
2. **Spur B, Replay-Lauf:** `entrypoint_replay.sh` mit `LIVE_VIEW=1` — Bild bewegt sich,
   Seite überlebt Reload, zweiter Browser-Tab funktioniert parallel.
3. **Spur B, RL:** `--check`-Lauf mit `LIVE_VIEW=1`; danach im echten Lauf messen, ob
   `LIVE_VIEW_EVERY_N=1` die Iterationszeit spürbar erhöht (Vergleich gegen `LIVE_VIEW=0`).
4. **Spur A, Viewport:** `NUM_EPISODES=2`, `LIVESTREAM=2` → nativer Client zeigt den Viewport
   in Echtzeit; Kamera lässt sich frei bewegen.
5. **Parallelität:** MP4-Aufzeichnung **und** Live-Ansicht gleichzeitig, ohne dass das
   Kamera-Rendering einbricht (Schrittzeit vergleichen).
6. **Frozen-Check (Spur A):** Aktualisiert der Stream pro Sim-Step oder friert er ein? Isaac Lab
   ruft den Render-Loop bei aktivem Livestream i. d. R. selbst auf — falls nicht,
   `simulation_app.update()` in der Rollout-Schleife ergänzen.

---

## 8. Risiken

| Risiko | Einschätzung / Gegenmaßnahme |
|---|---|
| **Isaac-Sim-6.0-Port insgesamt unverifiziert** | Der Basis-Image-Wechsel auf `isaac-lab:3.0.0-beta2-post1` ist noch nicht auf Hardware gelaufen (Beta; flash-attn gegen torch 2.10 gebaut, Isaac Sim 6.0 liefert 2.11). Der Livestream **erbt** dieses Risiko vollständig. → Livestream-Arbeit erst *nach* einem grünen `server_rl_run.sh check`. |
| **Kit ignoriert falsche Settings still** | Ein Tippfehler in `--/exts/…` führt nicht zu einem Fehler, sondern zu einem Stream, der auf dem Default-Port lauscht. → In Phase 0 die gesetzten Werte im Kit-Log gegenprüfen. |
| **UDP 47998 im Institutsnetz blockiert** | Trifft nur Spur A. Spur B (reines HTTP) ist davon nicht betroffen — deshalb wird sie zuerst gebaut. |
| **Nur ein WebRTC-Client gleichzeitig** | Akzeptabel für Eval-Debugging, nicht für RL → §1-Zuordnung. |
| **Frame-Stream bremst die Sim** | Nur letztes Frame wird gehalten, JPEG-Encode im Hintergrund-Thread, `LIVE_VIEW_EVERY_N` als Ventil. In Phase 2 messen statt schätzen. |
| **Zwei fast identische Entrypoint-Blöcke** (`sim` + `baseline`) | Laufen bei Änderungen auseinander → gemeinsames `lib_livestream.sh` (§3.3). |
| **Kein Auth vor dem Frame-Stream** | Der MJPEG-Port ist ungeschützt. Im VPN/Institutsnetz vertretbar; **nicht** auf einer öffentlichen vast.ai-IP exponieren (dort nur per SSH-Tunnel nutzen). |

---

## 9. Anhang — v1-Stand (vast.ai, WebRTC)

Der v1-Code ist umgesetzt und bleibt für vast.ai die Referenz:

- [x] `entrypoint_sim.sh` + `entrypoint_baseline.sh`: `LIVESTREAM`/`LIVESTREAM_PORT`/`PUBLIC_IP`,
      konditionales `--headless`/`--livestream`
- [x] `Dockerfile.vastai`: `LIVESTREAM`/`LIVESTREAM_PORT`-Defaults
- [x] Doku: [vastai-anleitung.md](../simulation/vastai-anleitung.md), Env-Var-Tabelle in [`CLAUDE.md`](../../CLAUDE.md)
- [ ] **Nie auf Hardware getestet** — weder auf vast.ai noch auf dem Server

Offene v1-Punkte sind in §6 als Phase 3–5 aufgegangen.

---

## Quellen

- [Livestream Clients — Isaac Sim 6.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/manual_livestream_clients.html) — Ports, Clients, NVENC-Anforderung, 6.0-Settings-Pfade
- [isaaclab.app — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.app.html) — `livestream` 0/1/2, `kit_args`, `PUBLIC_IP`
- [isaaclab.app.app_launcher — Quellcode](https://docs.robotsfan.com/isaaclab_official/main/_modules/isaaclab/app/app_launcher.html) — welche Kit-Args der Launcher selbst injiziert
- [Deep-dive into AppLauncher — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/launch_app.html)
- [Bug: wrong experience file when headless+livestream — IsaacLab#381](https://github.com/isaac-sim/IsaacLab/issues/381)
- [How to use WebRTC livestream in docker container — IsaacLab#4116](https://github.com/isaac-sim/IsaacLab/issues/4116)
- [Specify ports when running on a remote machine — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/specify-ports-when-running-on-a-remote-machine/342701)
- [RTX PRO 6000 Blackwell — NVENC-Spezifikation](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
