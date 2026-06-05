# Implementierungsplan — Live-Stream der Isaac-Lab-Sim (WebRTC)

**Status:** ✅ Code umgesetzt (Test auf vast.ai ausstehend) · **Erstellt:** 2026-06-02 · **Umgesetzt:** 2026-06-02

> **Umsetzungs-Entscheidung (vast.ai):** Gewählte Strategie = **WebRTC mit flexiblem Port**.
> Der intern gebundene Signaling-Port ist frei per `LIVESTREAM_PORT` setzbar; auf vast.ai
> wird er auf den **extern gemappten** Port gesetzt (intern == extern), damit der in der
> SDP eingebettete Port erreichbar ist. `publicEndpointAddress` = `PUBLIC_IP` (auto via
> `ifconfig.me`). Konkrete Bedien-Anleitung: [vastai-anleitung.md](../simulation/vastai-anleitung.md)
> → Abschnitt „Optional — Live-Stream des 3D-Viewports (WebRTC)".

Dieses Dokument plant das **Live-Streaming der laufenden Isaac-Lab-Sim-Eval** vom
Remote-GPU (vast.ai) auf den lokalen Rechner. Heute produziert die Sim-Eval nur
**aufgezeichnete MP4s** (`/data/sim_videos`), die erst *nach* dem Lauf via `docker cp`
geholt werden können (siehe [umsetzungsnotizen.md §9](../simulation/umsetzungsnotizen.md)). Ziel
ist eine **Echtzeit-Visualisierung des 3D-Viewports**, während die Eval läuft — zum
Debuggen von Greif-Verhalten, Kamera-Posen und Policy-Rollouts ohne Wartezeit.

> **Quelle der Wahrheit für den aktuellen Stand** bleibt
> [umsetzungsnotizen.md](../simulation/umsetzungsnotizen.md). Dieser Plan ist additiv und ändert
> nichts am bestehenden headless-Video-Pfad — Live-Stream wird **opt-in** über eine neue
> Env-Var `LIVESTREAM`.

---

## 1. Recherche — gängige Praxis für Isaac-Sim-Streaming

NVIDIA bietet für headless betriebene Isaac-Sim-/Isaac-Lab-Instanzen offiziell
**Livestreaming über WebRTC** an. Es gibt zwei Streaming-Wege (einer davon veraltet):

| Methode | Status | Client |
|---|---|---|
| **WebRTC Livestream** | ✅ empfohlen | (a) **Isaac Sim WebRTC Streaming Client** (native Desktop-App, kein leistungsfähiges GPU lokal nötig) <br> (b) **WebRTC Browser-Client** unter `http://<ip>:8211/streaming/webrtc-client?server=<ip>` |
| Omniverse Streaming Client | ⚠️ deprecated | alte Kit-Streaming-App, größere Port-Range |

**Aktivierung in Isaac Lab** — zwei äquivalente Wege:

- **Env-Var:** `LIVESTREAM={1,2}`
- **AppLauncher-Flag:** `--livestream {0,1,2}` (von `AppLauncher.add_app_launcher_args`
  bereitgestellt — unser Skript ruft das bereits auf, das Flag ist also schon verfügbar)

Bedeutung der Werte:

| Wert | Bedeutung |
|---|---|
| `0` | kein Stream (default) |
| `1` | WebRTC über **öffentliche** Netze |
| `2` | WebRTC über **private/lokale** Netze |

Wichtige Eigenschaften (aus der Recherche, Quellen unten):

- **Livestream impliziert Headless.** Sobald `LIVESTREAM ∈ {1,2}`, läuft die App
  zwangsweise headless — es darf kein zweites `--headless` Konflikte mit der
  Experience-File-Auswahl erzeugen (bekannter Bug
  [IsaacLab#381](https://github.com/isaac-sim/IsaacLab/issues/381)). → In unserem
  Entrypoint **`--headless` weglassen, wenn `LIVESTREAM` gesetzt ist**.
- **`--enable_cameras` bleibt erforderlich** (Szene enthält 5 Kameras) — passen wir bereits.
- **Remote-Endpunkt:** Für Internet-erreichbare Instanzen den öffentlichen Endpunkt setzen:
  ```
  --/app/livestream/publicEndpointAddress=<PUBLIC_IP>
  --/app/livestream/port=<PORT>
  ```
  oder über die Env-Var `PUBLIC_IP` (z. B. `PUBLIC_IP=$(curl -s ifconfig.me)`).
- **Nur ein Client gleichzeitig** pro Isaac-Sim-Instanz.

### 1.1 Ports

Die Port-Liste variiert je nach Isaac-Sim-Version (unser Basis-Image ist
`nvcr.io/nvidia/isaac-lab:2.3.2`). Konservativ alle relevanten Ports öffnen:

| Port | Protokoll | Zweck |
|---|---|---|
| **8211** | TCP/HTTP | WebRTC **Browser-Client** + Signaling |
| **49100** | TCP | WebRTC Streaming (Signaling/Steuerung) |
| **47998** | UDP | WebRTC Medien-Stream (Video) |
| 47995–48012, 49000–49007 | TCP/UDP | nur falls **alter** Omniverse-Client genutzt wird |

### 1.2 ⚠️ Harte GPU-Anforderung: NVENC

WebRTC-Streaming braucht den **NVENC-Hardware-Encoder** der GPU.

- **A100 hat KEIN NVENC** → Livestream technisch unmöglich (offiziell dokumentiert).
- H100 ebenfalls für Streaming nicht vorgesehen.

**Für dieses Projekt unkritisch:** Die Sim-Eval ist ohnehin auf **Ampere+/Ada mit RT-Cores**
beschränkt (L40, RTX 4090, A6000, RTX 3090 — siehe
[umsetzungsnotizen.md §1](../simulation/umsetzungsnotizen.md)). Diese GPUs haben **alle NVENC**.
Die Streaming-Anforderung **verschärft** die bestehende GPU-Regel also nur konsistent:
A100/H100 sind bereits aus zwei Gründen ausgeschlossen (keine RT-Cores **und** kein NVENC).

---

## 2. Architektur — Einordnung in den bestehenden Stack

```
┌─ vast.ai-Instanz (L40 / RTX 4090 / A6000) ───────────────────────────┐
│  Container: …-sim-vastai:latest                                       │
│                                                                       │
│   GR00T-Policy-Server  ──ZMQ:5555──►  Isaac-Lab-Sim-Client            │
│   (/app/Groot-1.6/.venv)              (isaaclab.sh -p, EGL-Render)     │
│                                          │                            │
│                                          │ LIVESTREAM=2               │
│                                          ▼                            │
│                                   WebRTC-Encoder (NVENC)              │
│                                   Ports 8211/49100 (TCP), 47998 (UDP) │
└───────────────────────────────────────────┼─────────────────────────┘
                                             │  WebRTC (Internet)
                                             ▼
                          Lokaler Rechner: WebRTC Streaming Client
                          ODER Browser:  http://<IP>:8211/streaming/webrtc-client
```

Der Video-Aufzeichnungs-Pfad (`save_episode_video` → `/data/sim_videos`) bleibt
**unverändert** und läuft parallel weiter — Live-Stream ersetzt ihn nicht, sondern ergänzt
ihn.

---

## 3. Geplante Code-Änderungen

> Alle Änderungen sind **opt-in** und ändern das Default-Verhalten (`LIVESTREAM=0`) nicht.

### 3.1 `Simulation/scripts/entrypoint_sim.sh`

1. **Neue Env-Var** dokumentieren + lesen (Default `0`):
   ```bash
   LIVESTREAM="${LIVESTREAM:-0}"
   LIVESTREAM_PORT="${LIVESTREAM_PORT:-49100}"
   ```
2. **PUBLIC_IP** automatisch ermitteln, wenn Stream aktiv und nicht gesetzt:
   ```bash
   if [[ "$LIVESTREAM" != "0" && -z "${PUBLIC_IP:-}" ]]; then
       PUBLIC_IP="$(curl -s ifconfig.me || true)"
   fi
   export LIVESTREAM PUBLIC_IP
   ```
3. **`--headless` konditional weglassen** und `--livestream` setzen. Aktuell wird
   `--headless` fest übergeben (Zeile ~234). Stattdessen Flags in einem Array bauen:
   ```bash
   APP_FLAGS=( --enable_cameras )
   if [[ "$LIVESTREAM" != "0" ]]; then
       APP_FLAGS+=( --livestream "$LIVESTREAM" )
       APP_FLAGS+=( --kit_args "--/app/livestream/publicEndpointAddress=${PUBLIC_IP} --/app/livestream/port=${LIVESTREAM_PORT}" )
   else
       APP_FLAGS+=( --headless )
   fi
   ```
   (Headless ist bei aktivem Livestream ohnehin impliziert — siehe §1.)
4. **Hinweis-Ausgabe** mit der Client-URL, damit der Nutzer direkt verbinden kann:
   ```bash
   if [[ "$LIVESTREAM" != "0" ]]; then
       log "Live-Stream aktiv (WebRTC). Verbinden via:"
       echo "    Browser:  http://${PUBLIC_IP}:8211/streaming/webrtc-client?server=${PUBLIC_IP}"
       echo "    Native:   Isaac Sim WebRTC Streaming Client → ${PUBLIC_IP}:${LIVESTREAM_PORT}"
   fi
   ```

### 3.2 `Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py`

- **`--livestream` ist bereits verfügbar** über `AppLauncher.add_app_launcher_args(parser)`
  (Zeile 76) — **keine neue CLI-Option nötig**.
- **Render-Schleife prüfen:** Im headless-Video-Modus genügt das Kamera-Rendering für die
  Frame-Sammlung. Für einen flüssigen Viewport-Stream muss der App-Render-Loop pro Step
  laufen. Verifizieren, dass die Eval-Schleife `env.sim.render()` bzw. `simulation_app.update()`
  pro Step aufruft (Isaac Lab tut das bei aktivem Livestream i. d. R. selbst — als
  Verifikationspunkt in §6 vermerkt).
- **Optionales Real-Time-Pacing** (`--realtime`): Die Eval läuft sonst so schnell wie
  möglich; für menschliches Zuschauen kann ein `time.sleep(dt - elapsed)` pro Step die Sim
  auf Echtzeit drosseln. **Niedrige Priorität** — erst nach funktionierendem Stream.

### 3.3 `Simulation/Dockerfile.vastai`

- `EXPOSE 8211 49100` und `EXPOSE 47998/udp` ergänzen (Dokumentationswert; auf vast.ai muss
  das Port-Mapping separat über die Docker-Options gesetzt werden, s. u.).
- `NVIDIA_DRIVER_CAPABILITIES=all` ist **bereits gesetzt** (Zeile 40) — enthält `video`
  (NVENC). Kein Zusatz-Install nötig; die WebRTC-Livestream-Extension ist im
  Isaac-Sim-Bundle enthalten.

### 3.4 vast.ai-Instanz-Konfiguration

In den **Docker-Options** der Instanz die Ports mappen (zusätzlich zu `-p 22` für SSH):
```
--ipc=host --shm-size=16g -p 22 -p 8211 -p 49100 -p 47998/udp
```

⚠️ **Kernproblem vast.ai:** vast.ai mappt Container-Ports auf **zufällige externe Ports**.
WebRTC bettet den Port aber in die Signaling-Verhandlung (SDP) ein → der vom Client
erwartete Port muss mit dem extern gemappten übereinstimmen. Lösungspfad:

1. Im vast.ai-Dashboard unter **"IP & Port Info"** die externen Ports ablesen, auf die
   `8211`/`49100`/`47998` gemappt wurden.
2. `LIVESTREAM_PORT` und ggf. `--/app/livestream/port=<extern>` auf den **extern gemappten**
   Wert setzen, `PUBLIC_IP` auf die öffentliche Instanz-IP.

---

## 4. Konfigurations-Referenz (geplante neue Env-Vars)

| Variable | Default | Zweck |
|---|---|---|
| `LIVESTREAM` | `0` | `0`=aus, `1`=WebRTC öffentlich, `2`=WebRTC privat/lokal |
| `LIVESTREAM_PORT` | `49100` | WebRTC-Streaming-Port (auf vast.ai = extern gemappter Port) |
| `PUBLIC_IP` | *(auto via `ifconfig.me`)* | Öffentliche IP der Instanz für den Remote-Endpunkt |

Nach Umsetzung in [vastai-anleitung.md](../simulation/vastai-anleitung.md) und der Env-Var-Tabelle in
[`CLAUDE.md`](../../CLAUDE.md) nachtragen.

---

## 5. Client-Setup (lokaler Rechner)

**Variante A — Browser (am einfachsten, kein Download):**
```
http://<PUBLIC_IP>:8211/streaming/webrtc-client?server=<PUBLIC_IP>
```

**Variante B — Native Isaac Sim WebRTC Streaming Client:**
- Von NVIDIA herunterladen (Isaac-Sim-Download-Seite), starten, als Server
  `<PUBLIC_IP>:<LIVESTREAM_PORT>` eingeben.
- Vorteil: stabiler bei höheren Auflösungen; läuft auch ohne lokale GPU.

---

## 6. Test- und Validierungsplan

1. **Lokal zuerst** (falls eine geeignete lokale GPU mit funktionierendem Vulkan/EGL
   verfügbar ist — siehe Vulkan-Einschränkung in
   [umsetzungsnotizen.md §3](../simulation/umsetzungsnotizen.md); auf WSL2 nicht möglich):
   `LIVESTREAM=2`, Verbindung vom **selben** Rechner via Browser-Client → Viewport sichtbar?
2. **vast.ai-Smoke-Test:** Kleine Eval (`NUM_EPISODES=2`) mit `LIVESTREAM=1`, Ports gemappt,
   `PUBLIC_IP` korrekt. Erfolgskriterium: Browser-Client zeigt den 3D-Viewport in Echtzeit.
3. **Render-Verifikation:** Prüfen, dass der Stream pro Sim-Step aktualisiert (nicht
   eingefroren) — ggf. `simulation_app.update()` / `sim.render()` in der Schleife ergänzen.
4. **Parallelität:** Sicherstellen, dass die MP4-Aufzeichnung **und** der Stream gleichzeitig
   laufen, ohne dass das Kamera-Rendering einbricht.
5. **Regression:** `LIVESTREAM=0` (Default) liefert exakt das alte Verhalten
   (headless + Videos, `results.json`).

---

## 7. Risiken & offene Fragen

| Risiko | Einschätzung / Gegenmaßnahme |
|---|---|
| **vast.ai UDP-Port-Mapping** | WebRTC-Medien laufen über UDP; vast.ai-UDP-Mapping ist weniger zuverlässig als TCP. Fallback: bei reinem TCP-Signaling kann das Bild über TURN/Relay laufen — sonst auf aufgezeichnetes Video zurückfallen. |
| **Port-Mismatch (zufälliges Mapping)** | Externen Port aus dem Dashboard ablesen und `LIVESTREAM_PORT` darauf setzen (§3.4). |
| **Bekannte Cloud-Streaming-Bugs** | In den Isaac-Lab-Discussions sind Disconnect-/Netzwerk-Probleme auf Headless-Cloud-Instanzen dokumentiert; häufig VPN/Tunnel-IP-Konflikte (z. B. ZeroTier). → echte öffentliche IP nutzen, kein VPN. |
| **Nur ein Client gleichzeitig** | Akzeptabel für Debugging. |
| **Bandbreite** | WebRTC passt Bitrate adaptiv an; bei schwacher Leitung Auflösung/FPS senken. |
| **Isaac-Sim-Version vs. Port-Set** | `isaac-lab:2.3.2` → konkrete Isaac-Sim-Version verifizieren; Port-Set ggf. anpassen (§1.1). |

---

## 8. Umsetzungs-Reihenfolge (Checkliste)

- [x] `entrypoint_sim.sh`: `LIVESTREAM`/`LIVESTREAM_PORT`/`PUBLIC_IP`-Logik + konditionales
      `--headless`/`--livestream` (§3.1)
- [x] `Dockerfile.vastai`: `EXPOSE`-Ports + `LIVESTREAM`/`LIVESTREAM_PORT`-Defaults ergänzt (§3.3)
- [x] Image neu bauen + pushen (`Simulation\update_sim_image.ps1 -VastAI`)
- [ ] vast.ai-Instanz mit gemappten Ports starten, externe Ports ablesen
- [ ] Smoke-Test `NUM_EPISODES=2 LIVESTREAM=1` → Browser-Client verbinden (§6)
- [ ] Render-/Parallelitäts-Verifikation (§6.3–6.4)
- [ ] Optional: `--realtime`-Pacing (§3.2)
- [x] Doku nachziehen: [vastai-anleitung.md](../simulation/vastai-anleitung.md) (neuer Live-Stream-Abschnitt),
      Env-Var-Tabelle in [`CLAUDE.md`](../../CLAUDE.md). Offen: [umsetzungsnotizen.md](../simulation/umsetzungsnotizen.md)

---

## Quellen

- [Livestream Clients — Isaac Sim 5.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/manual_livestream_clients.html)
- [Livestream Clients — Isaac Sim 4.5 Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/manual_livestream_clients.html)
- [Deep-dive into AppLauncher — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/launch_app.html)
- [isaaclab.app — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.app.html)
- [How to do web streaming through WebRTC for IsaacLab? — Discussion #4361](https://github.com/isaac-sim/IsaacLab/discussions/4361)
- [WebRTC Visualization Error in Isaac Lab 2.1 Docker Container — Discussion #3192](https://github.com/isaac-sim/IsaacLab/discussions/3192)
- [Bug: wrong experience file when headless+livestream — Issue #381](https://github.com/isaac-sim/IsaacLab/issues/381)
- [Isaac Sim headless docker streaming to WebRTC Streaming Client — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/isaac-sim-headless-docker-streaming-to-webrtc-streaming-client/330619)
