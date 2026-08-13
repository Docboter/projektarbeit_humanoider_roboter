# Live-Ansicht — Isaac Sim auf dem eigenen Rechner öffnen

**Was das ist:** Statt nach dem Lauf MP4s aus dem Container zu holen, streamt Isaac Sim
seinen **3D-Viewport live per WebRTC**. Auf dem Arbeitsrechner öffnet ihn die App *Isaac Sim
WebRTC Streaming Client* — mit **freier Kamera**: Szene drehen, zoomen, dem Greifer über die
Schulter schauen, während die Policy läuft. Die Simulation selbst rechnet weiter auf dem
Server; lokal wird nur ein H.264-Videostrom dekodiert (keine lokale GPU-Anforderung).

Das ist „Spur A" aus dem [Livestream-Plan](../weiterfuehrend/livestream-plan.md).
Umgesetzt am 2026-08-13.

> **Ehrlicher Reifegrad:** Der Code ist vollständig und die Fehler D1–D4/D6 des Plans sind
> behoben, aber **auf Hardware ist die Live-Variante noch nie gelaufen**. Sie erbt zusätzlich
> das Restrisiko des Isaac-Sim-6.0-Ports. Deshalb steht [Schritt 2](#schritt-2--phase-0-prüfen-livecheck)
> (`livecheck`) vor dem ersten echten Lauf, und bei `LIVESTREAM=0` bleibt das bisherige
> Verhalten **bit-identisch** — die Live-Variante kann nichts kaputt machen, was vorher lief.

---

## Zwei Live-Wege — welcher wofür

| | **Spur A — Isaac-Sim-Viewport** (dieses Dokument) | **Spur B — Bild-Stream im Browser** ([`live_view.py`](../../Simulation/g1_dex3_sim/live_view.py)) |
|---|---|---|
| Schalter | `LIVESTREAM=2` | `LIVE_VIEW=1` |
| Was man sieht | kompletter Isaac-Sim-Viewport, freie Kamera, Isaac-Sim-UI | die gerenderten Kamerabilder (= Modell-Eingabe) + Live-Metriken |
| Client | Desktop-App *Isaac Sim WebRTC Streaming Client* | jeder Browser, URL öffnen |
| Zuschauer | genau **einer** | beliebig viele |
| Netzabbruch | Sitzung weg, neu verbinden | egal — Tab neu laden |
| Über `ssh -L` | ✗ (UDP-Medien) | ✓ (reines HTTP) |
| Passt zu | `eval`, `grasp`, `baseline` — zuschauen, Greifposen debuggen | `rl` — Lauf über Stunden/Tage nebenbei offen |

Beide lassen sich **gleichzeitig** einschalten. Für den langen RL-Lauf bleibt Spur B die
robustere Wahl; Spur A ist der Weg, wenn man sich die Szene wirklich *ansehen* will.

---

## Voraussetzungen

| | |
|---|---|
| **GPU** | NVENC-Encoder nötig. Die RTX PRO 6000 Blackwell hat vier NVENC-Engines ✓. A100/H100 haben **keine** — dieselben GPUs, die schon am RT-Core-Rendering scheitern. |
| **Ports** | `49100/tcp` (Signaling) **und** `47998/udp` (Video). Beide `intern == extern` gemappt — WebRTC bettet den Port in die SDP-Aushandlung ein. `server_rl_run.sh` mappt sie beim **Anlegen** des Containers automatisch. |
| **Firewall** | UDP 47998 muss zwischen Arbeitsrechner und Server offen sein. Das ist der wahrscheinlichste Stolperstein im Institutsnetz. |
| **Client** | Isaac Sim WebRTC Streaming Client (Windows/Linux/macOS), Download über die [Isaac-Sim-Doku → Livestream Clients](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/manual_livestream_clients.html) |

---

## Schritt 1 — Client auf dem Arbeitsrechner installieren

1. **Isaac Sim WebRTC Streaming Client** herunterladen (Windows-Installer bzw. AppImage) und starten.
2. Im Feld *Server* die Server-Adresse eintragen — **ohne** `http://`:
   ```
   <server-ip>:49100
   ```
3. *Connect*. Solange auf dem Server kein Lauf mit `LIVESTREAM≠0` aktiv ist, bleibt das
   Fenster leer — das ist normal.

> **WSL2 als Client:** Windows-Client bevorzugen. WSL2 hängt hinter einer eigenen NAT; UDP
> nach außen funktioniert dort zwar meist, ist aber eine zusätzliche Fehlerquelle, die man
> sich beim ersten Test spart.

---

## Schritt 2 — Phase 0 prüfen (`livecheck`)

Bevor eine ganze Eval für einen Stream läuft, die drei Dinge prüfen, die ihn im Container
scheitern lassen — NVENC, Livestream-Extension, Ports:

```bash
./Simulation/server_rl_run.sh livecheck
```

Ausgegeben werden Isaac-Sim-Version (entscheidet über die gültigen Kit-Settings-Pfade),
NVENC-Status, die gefundenen Livestream-Extensions und ob der Container beide Ports
veröffentlicht hat. Fehlt ein Port-Mapping, hilft nur ein einmaliges Neuanlegen:

```bash
./Simulation/server_rl_run.sh clean      # Daten unter $RL_HOST_DATA_DIR bleiben erhalten
```

Vom Arbeitsrechner aus zusätzlich die Firewall prüfen:

```bash
nc -vz  <server-ip> 49100      # Signaling
nc -vzu <server-ip> 47998      # Medien (UDP) — der kritische
```

---

## Schritt 3 — Lauf mit Live-Variante starten

`LIVESTREAM=2` (privates Netz/VPN) ist auf dem eigenen Server der richtige Wert. Er wirkt
auf **alle vier Läufe**:

```bash
# Closed-Loop-Eval — kurz halten beim ersten Versuch
HF_TOKEN=hf_... LIVESTREAM=2 NUM_EPISODES=2 EPISODE_LENGTH_S=120 \
    ./Simulation/server_rl_run.sh eval

# Greif-Physik-Test (Open-Loop-Replay) — hier ist die freie Kamera am wertvollsten
HF_TOKEN=hf_... LIVESTREAM=2 GRASP_MODE=hold ./Simulation/server_rl_run.sh grasp

# Aufbau-Smoke-Test: Isaac Sim startet im Livestream-Modus (Sekunden, kein Training)
HF_TOKEN=hf_... LIVESTREAM=2 ./Simulation/server_rl_run.sh check

# RL-Lauf (für Tage lieber LIVE_VIEW=1; beides zusammen geht)
HF_TOKEN=hf_... WANDB_API_KEY=... LIVESTREAM=2 LIVE_VIEW=1 \
    RL_NUM_ENVS=4 ./Simulation/server_rl_run.sh rl
```

Baseline-Eval (stock G1 + Dex1-Greifer) läuft über denselben Schalter, wenn
[`entrypoint_baseline.sh`](../../Simulation/scripts/entrypoint_baseline.sh) mit
`SIM_MODE=baseline` und `-e LIVESTREAM=2` gestartet wird.

Der Startlauf gibt den Verbindungshinweis mit der konkreten Server-IP aus. Der Viewport
erscheint erst, wenn Isaac Sim die Szene geladen hat — das dauert rund eine Minute.

### Live **statt** Video (und wie man beides bekommt)

Mit aktiver Live-Variante wird `--video-dir` **leer** übergeben. Die Eval-/Replay-Skripte
schalten daraufhin Frame-Sammeln, MP4-Ausgabe und den `rgb_array`-Render-Mode ab
(`record = bool(args.video_dir)`) — das spart pro Step eine GPU→CPU-Kopie. Wer beides will:

```bash
LIVE_KEEP_VIDEO=1 LIVESTREAM=2 ./Simulation/server_rl_run.sh eval
```

Zwei Dinge bleiben davon **unberührt**, weil sie nicht am `--video-dir` hängen:
`RL_WANDB_VIDEO_EVERY` (RL-Rollout ins W&B-Dashboard) und `LIVE_VIEW=1` (Spur B).

---

## Tempo — warum der Roboter in Zeitlupe läuft

Die Sim rechnet langsamer als Echtzeit. **Erste Messung auf Hardware (2026-08-13,
`LIVESTREAM=2`, 1 Env):**

```
… Step 150/3600 (17s, 8.6 Steps/s, 3.5x Echtzeit) | Obs 0% · Inferenz 5% · Sim+Render 94%
```

Bei 30 Hz Policy-Takt sind 8,6 Steps/s also **3,5-fache Zeitlupe** — und die Zeit geht fast
vollständig ins **Rendern**, nicht ins Modell. Das ist die entscheidende Zahl, weil sie die
naheliegende Maßnahme ausschließt: `EXECUTION_HORIZON` hoch zu setzen spart nur an den 5 %
Inferenz und bringt praktisch nichts.

Die Zeile schreibt der Eval-Runner alle 25 Schritte; dieselben Werte stehen je Episode als
`steps_per_s` und `time_share` in `results.json`.

| Hebel | Wirkung | Kosten |
|---|---|---|
| `SCENE_CAM=0` | Eine von fünf Kameras weniger je Step (~20 % der Kameralast) | **Keine.** `cam_scene` geht nur ins MP4 — die Policy sieht sie nie, Erfolgsraten bleiben vergleichbar. Es entfällt nur das Übersichtsbild |
| `CAM_RES_SCALE=0.5` | Stärkster Hebel — Pixelzahl geht quadratisch ein | **Ändert die Modell-Eingabe.** Die 640×480 sind gegen die Dataset-Referenzframes kalibriert; ein skalierter Lauf ist zum **Zuschauen**, seine Erfolgsrate nicht mit anderen Läufen vergleichbar |
| `RL_AA_MODE=Off` | Anti-Aliasing aus | Bild wird kantiger; verändert die Modell-Eingabe leicht |
| `LIVESTREAM=0` gegenprüfen | Zeigt, wie viel der WebRTC-Viewport selbst kostet (eigener Render-Pass + NVENC je `sim.render()`) | Nur ein Vergleichslauf |

Zum flüssigen Zuschauen also:

```bash
HF_TOKEN=hf_... LIVESTREAM=2 SCENE_CAM=0 CAM_RES_SCALE=0.5 \
    NUM_EPISODES=2 EPISODE_LENGTH_S=120 ./Simulation/server_rl_run.sh eval
```

Für **Messläufe** nur `SCENE_CAM=0` verwenden — alles andere verändert, was das Modell sieht.

Was **nicht** hilft: mehr GPU-Leistung. Renderer und Policy laufen strikt abwechselnd, jede
Seite wartet auf die andere — eine GPU-Auslastung um 50 % ist hier der Normalzustand und kein
Symptom.

### Für echte Hardware zählt eine andere Zahl

Auf einem realen Roboter entfällt das Rendering ersatzlos — die Kameras liefern ihre Bilder
selbst. Übrig bleiben die 5 %: die Zeit vom Observation-Dict bis zum Action-Chunk. Die misst
[`policy_latency.py`](../../Simulation/scripts/policy_latency.py) isoliert, in-process, ohne
Sim und ohne ZMQ:

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh latency
```

Maßstab ist das Chunk-Budget: ein Aufruf deckt `EXECUTION_HORIZON` Schritte ab, bei 8 Schritten
und 30 Hz also 267 ms. Berichtet werden Mittel, Median, p95 und **Maximum** — im Echtzeitbetrieb
ist der schlechteste Aufruf die relevante Zahl, weil ein einzelner Ausreißer über dem Budget
eine Lücke in der Aktionsfolge bedeutet.

Die Differenz zwischen dieser Zahl und dem `Inferenz … ms/Aufruf` aus der Eval ist der
**Transport-Overhead** des ZMQ-Wegs (4 unkomprimierte Bilder ≈ 3,7 MB je Aufruf).

> Die Balance des Roboters hängt **nicht** an dieser Schleife: der Whole-Body-Controller läuft
> entkoppelt mit eigener, deutlich höherer Rate und folgt nur Geschwindigkeitsbefehlen der VLA
> ([lokomotion-recherche.md §3.1](../weiterfuehrend/lokomotion-recherche.md)). Eine zu langsame
> Policy erzeugt ruckelige Bewegung, keinen Sturz.

## Was unter der Haube passiert

| Bestandteil | Wo |
|---|---|
| Gemeinsame Logik (Flags, Kit-Settings, Video-an/aus, Verbindungshinweis) | [`Simulation/scripts/lib_livestream.sh`](../../Simulation/scripts/lib_livestream.sh) |
| Sourcen + Aufruf | [`entrypoint_sim.sh`](../../Simulation/scripts/entrypoint_sim.sh), [`entrypoint_baseline.sh`](../../Simulation/scripts/entrypoint_baseline.sh), [`entrypoint_replay.sh`](../../Simulation/scripts/entrypoint_replay.sh), [`entrypoint_rl.sh`](../../Simulation/scripts/entrypoint_rl.sh) |
| RL-Trainer (`AppLauncher` statt CLI-Flags) | [`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py) |
| Ports, Durchreichen, `livecheck`, `grasp`-Flags | [`server_rl_run.sh`](../../Simulation/server_rl_run.sh) |

Zwei Details, die sonst Stunden kosten:

- **`--headless` und `--livestream` schließen sich aus.** Beides zusammen lässt Isaac Lab das
  falsche Experience-File wählen ([IsaacLab#381](https://github.com/isaac-sim/IsaacLab/issues/381))
  — der Stream bliebe schwarz. Jeder Livestream-Modus impliziert Headless ohnehin.
  `livestream_app_flags` setzt deshalb immer nur eines von beiden.
- **Kit ignoriert falsche Settings still.** Isaac Sim 6.0 hat die Pfade umbenannt
  (`--/exts/omni.kit.livestream.app/primaryStream/...` statt `--/app/livestream/...`). Ein
  Tippfehler führt nicht zu einem Fehler, sondern zu einem Stream auf dem Default-Port.
  `livestream_kit_args` liest deshalb die Isaac-Sim-`VERSION` und wählt den Pfad danach; ohne
  erkennbare Version setzt es **beide** (der falsche verpufft folgenlos). Bei Default-Ports im
  privaten Netz werden gar keine Kit-Settings gebraucht — dann bleibt der String leer, was
  zugleich die Doppelbelegung durch den `AppLauncher` (Defekt D3) vermeidet.

---

## Fehlersuche

| Symptom | Ursache / Abhilfe |
|---|---|
| Client verbindet, Bild bleibt **schwarz** | UDP 47998 kommt nicht durch (Firewall) oder ist nicht gemappt. `livecheck` prüft das Mapping, `nc -vzu` die Strecke. |
| *Connection refused* auf 49100 | Kein Lauf aktiv, Port nicht veröffentlicht (`livecheck`) oder Isaac Sim noch am Laden. |
| Stream steht, Szene **friert ein** | Der Render-Loop wird nicht getrieben. Im RL-Pfad regelt das `LIVESTREAM_UPDATE_EVERY_N` (Default 1 = jeder Rollout-Step ein `simulation_app.update()`); `0` schaltet es ab. |
| `docker run` meldet *port is already allocated* | Ein anderer Prozess hält 49100/47998. `LIVESTREAM_PORT=49101 ./Simulation/server_rl_run.sh clean` und neu anlegen. `ensure_container` legt den Container notfalls ohne die Livestream-Ports an, statt den ganzen Workflow zu blockieren — dann warnt es. |
| Videos fehlen nach dem Lauf | So gewollt: live **statt** Video. `LIVE_KEEP_VIDEO=1` schreibt beides. |
| Läuft mit `LIVESTREAM=1` auf vast.ai nicht | Dort muss `LIVESTREAM_PORT` der **extern gemappte** Port sein (intern == extern) und `PUBLIC_IP` stimmen — siehe [vastai-anleitung.md](vastai-anleitung.md). Auf einer öffentlichen IP ist der Viewport **ungeschützt**; besser per SSH-Tunnel arbeiten oder Spur B nutzen. |
| Kit-Settings greifen nachweislich nicht | `LIVESTREAM_SETTINGS_STYLE=new\|old\|both` erzwingen, oder die Zeile komplett selbst setzen: `LIVESTREAM_KIT_ARGS="--/… "`. |

---

## Env-Var-Referenz

| Variable | Default | Zweck |
|---|---|---|
| `LIVESTREAM` | `0` | `0`=aus (headless), `1`=WebRTC öffentlich, `2`=WebRTC privat/lokal |
| `LIVESTREAM_PORT` | `49100` | Signaling, TCP. Intern == extern mappen |
| `LIVESTREAM_MEDIA_PORT` | `47998` | Medien, UDP. Muss durch die Firewall |
| `LIVE_KEEP_VIDEO` | `0` | `1` = zusätzlich MP4s schreiben |
| `LIVESTREAM_UPDATE_EVERY_N` | `1` | Nur RL: alle n Rollout-Steps `simulation_app.update()`; `0` = nie |
| `LIVESTREAM_SETTINGS_STYLE` | `auto` | `auto\|new\|old\|both` — welche Kit-Settings-Pfade gesetzt werden |
| `LIVESTREAM_KIT_ARGS` | — | Manueller Override der kompletten Kit-Settings-Zeile |
| `PUBLIC_IP` | — | Nur `LIVESTREAM=1`; wird sonst gar nicht erst ermittelt |

Siehe auch: [env-vars.md](../training/env-vars.md) · [livestream-plan.md](../weiterfuehrend/livestream-plan.md) ·
[rl-anleitung.md](../weiterfuehrend/rl-anleitung.md)
