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

## Drei Live-Wege — welcher wofür

Spur A gibt es in **zwei Ausführungen**: als native App und — seit 2026-08-17 — im Browser.
Beide zeigen denselben WebRTC-Stream aus demselben Lauf; sie unterscheiden sich nur im
Client. Spur B ist etwas ganz anderes: kein WebRTC, sondern Einzelbilder über HTTP.

| | **A1 — Viewport, native App** | **A2 — Viewport im Browser** | **B — Bild-Stream im Browser** |
|---|---|---|---|
| Schalter | `LIVESTREAM=2` | `LIVESTREAM=2` + `webview` | `LIVE_VIEW=1` |
| Was man sieht | kompletter Isaac-Sim-Viewport, freie Kamera, Isaac-Sim-UI | dasselbe | die gerenderten Kamerabilder (= Modell-Eingabe) + Live-Metriken |
| **Steuern** | ✓ Maus + Tastatur | ✓ Maus + Tastatur | ✗ nur zuschauen |
| Client | Desktop-App *Isaac Sim WebRTC Streaming Client* | Chromium/Chrome/Edge, URL öffnen | jeder Browser, URL öffnen |
| Installation nötig | ja | **nein** | nein |
| Zuschauer | genau **einer** | genau **einer** | beliebig viele |
| Netzabbruch | Sitzung weg, neu verbinden | Sitzung weg, Tab neu laden | egal — Tab neu laden |
| Braucht UDP 47998 | ✓ | ✓ (spart die App, **nicht** den Port) | ✗ |
| Über `ssh -L` | ✗ | ✗ | ✓ (reines HTTP) |
| Passt zu | `view`, `eval`, `grasp`, `baseline` | dito, wenn man nichts installieren will/darf | `rl` — Lauf über Stunden/Tage nebenbei offen |

Der schnellste Einstieg in alle drei ist [`view`](#schritt-3--erster-blick-ohne-jede-gewichte-view):
die Szene ohne Modell, ohne Checkpoint, ohne `HF_TOKEN`.

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

## Schritt 3 — Erster Blick ohne jede Gewichte (`view`)

Bevor ein Lauf mit Modell für den Stream herhalten muss, gibt es den Weg ganz ohne:

```bash
./Simulation/server_rl_run.sh view
```

Kein `HF_TOKEN`, kein Checkpoint, kein GR00T-Server. Aufgebaut wird nur die Isaac-Lab-Env
— Roboter, Tisch, Würfel, Licht, Kameras —, und der Roboter hält still seine Home-Pose.
„Leer" heißt hier **leer an Policy, nicht an Szene**.

Das ist der billigste Weg zum ersten Bild und beantwortet auf einmal drei Fragen, die man
sonst erst nach einem 10-GB-Download stellt: Kommt Isaac Sim im Livestream-Modus hoch?
Rendert diese GPU überhaupt (RT-Cores)? Sieht die Szene so aus, wie sie soll?

Ohne `LIVESTREAM` oder `LIVE_VIEW` setzt `view` selbst `LIVESTREAM=2` — headless wäre
sinnlos, es gäbe ja nichts zu sehen. Beide Wege gehen auch hier einzeln oder zusammen:

```bash
LIVESTREAM=0 LIVE_VIEW=1 ./Simulation/server_rl_run.sh view    # nur Spur B (Browser)
LIVESTREAM=2 LIVE_VIEW=1 ./Simulation/server_rl_run.sh view    # beides
VIEW_NUM_ENVS=4 ./Simulation/server_rl_run.sh view             # das Klon-Gitter ansehen
VIEW_DURATION_S=14400 ./Simulation/server_rl_run.sh view       # 4 h statt 1 h
```

**Woher das USD kommt.** `view` sucht der Reihe nach im Container (Checkpoint-Verzeichnis,
`/workspace/assets`, `/data/assets`) und kopiert erst als letzten Schritt vom Host hinein:
`data/g1_dex3.usd` plus `data/configuration/` aus dem Repo, zusammen ~40 MB. Beide gehören
zusammen — der Wrapper referenziert `configuration/` relativ zu sich selbst. Das Ziel
`/data/assets` liegt im Bind-Mount und überlebt damit ein `clean`. Andere Quelle:
`VIEW_HOST_ASSET_DIR=/pfad/zu/usd`.

**Eigene Defaults.** `view` misst nichts — keine Erfolgsrate, keine `results.json`, kein
Vergleich mit anderen Läufen. Deshalb sind hier Sparhebel erlaubt, die in einem Messlauf
die Zahlen unvergleichbar machen würden:

| | `view`-Default | sonst | warum hier erlaubt |
|---|---|---|---|
| `SCENE_CAM` | `0` | `1` | `cam_scene` geht nur ins MP4, das hier gar nicht entsteht |
| `CAM_RES_SCALE` | `0.5` | `1` | es gibt keine Modell-Eingabe, die kalibriert bleiben müsste |
| `DR_ENABLED` | `0` | `1` | stabile Beleuchtung ist zum Ansehen nützlicher als eine je Episode gewürfelte |

Alle drei sind überschreibbar (`SCENE_CAM=1 ./Simulation/server_rl_run.sh view`).

Nach `VIEW_DURATION_S` Sekunden (Default 3600) endet der Lauf von selbst — damit ein
abgerissenes SSH-Fenster keine GPU dauerhaft blockiert. Alle 300 s (`EPISODE_LENGTH_S`)
setzt die Env zurück und würfelt die Würfelpositionen neu.

**Die Laufzeit ist immer endlich — einen Dauerlauf gibt es nicht.** Wer länger zusehen
will, setzt eine größere Zahl (`VIEW_DURATION_S=14400` für 4 h). `VIEW_DURATION_S=0`
stand hier früher für „bis Strg-C" und war unbrauchbar: der Lauf hängt an einem
`docker exec` **ohne TTY**, Strg-C beendet nur den Client auf dem Host. Im Container liefe
die Sim weiter und hielte die GPU, während der Wrapper den Lauf mangels Erfolgsmarker
`[view] fertig` als gescheitert meldet. `0` wird deshalb mit einer Warnung auf 3600 s
zurückgedreht. Vorzeitig beenden geht sauber über
`docker exec groot-rl pkill -f view_sim.py`.

> Wie die restliche LIVE-Variante ist auch `view` **auf Hardware noch nicht gelaufen**.
> Es ist aber der Lauf mit den wenigsten beweglichen Teilen — scheitert er, liegt es an
> Isaac Sim, der GPU oder den Ports, nicht am Modell.

---

## Schritt 4 — Lauf mit Live-Variante starten

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

## Variante A2 — derselbe Viewport, aber im Browser (`webview`)

Wer die native App nicht installieren will oder darf, bekommt denselben Viewport samt
Maus- und Tastatursteuerung im Browser:

```bash
# Terminal 1 — der Stream (wie gehabt; 14400 s = 4 h statt der 1 h Default)
LIVESTREAM=2 VIEW_DURATION_S=14400 ./Simulation/server_rl_run.sh view

# Terminal 2 — die Seite dazu (einmalig Build, danach Sekunden)
./Simulation/server_rl_run.sh webview
```

Dann `http://<server-ip>:8210/` in **Chromium, Chrome oder Edge** öffnen. Firefox wird von
NVIDIAs Streaming-Bibliothek nicht unterstützt.

Der Build dauert einmalig rund 30 Sekunden und braucht Zugang zu **zwei** Registries:

| Registry | wofür |
|---|---|
| `registry.npmjs.org` | Vite und die übrigen JS-Abhängigkeiten |
| `edge.urm.nvidia.com/artifactory/…/omniverse-client-npm` | der `@nvidia`-Scope — **anonym lesbar, kein Token** |

Die zweite ist der wahrscheinlichste Stolperstein im Institutsnetz. Ein
`npm error 404 '@nvidia/create-ov-web-rtc-app@1.14.2' is not in this registry` bedeutet
genau das: der Scope liegt nicht auf npmjs.org, und die NVIDIA-Adresse ist nicht erreichbar
oder gefiltert. Paketname und Version sind dann trotzdem richtig.

**Was das ist.** Isaac Sim 6.0 hat keinen eingebauten Browser-Client mehr — der alte auf
Port 8211 entfiel mit 5.x (Defekt D1). NVIDIA liefert stattdessen einen separaten
Web-Viewer, eine kleine React/Vite-App. [`Dockerfile.webviewer`](../../Simulation/Dockerfile.webviewer)
baut sie; `webview` startet sie als **eigenen kleinen Container** neben `groot-rl`.

**Wichtig für das mentale Modell:** dieser Container enthält *keinen Simulator*. Er
serviert nur eine Webseite. Der Browser verbindet sich anschließend **direkt** auf
`49100/tcp` + `47998/udp` des Sim-Containers. Der Web-Viewer ist ein zweiter *Client* für
denselben Stream, kein zweiter Server.

Daraus folgen die beiden Dinge, die man am ehesten falsch erwartet:

1. **`webview` allein zeigt nichts.** Ohne parallel laufenden `LIVESTREAM=2`-Lauf ist die
   Seite da, aber leer.
2. **Die Firewall-Anforderung bleibt identisch.** Der Browser spart die App-Installation,
   nicht den UDP-Port. Ist 47998/udp zu, bleibt das Bild hier genauso schwarz. Wer *das*
   Problem lösen will, braucht Spur B (`LIVE_VIEW=1`, reines HTTP, `ssh -L`-tunnelbar).

**Die IP steckt im JS-Bundle.** NVIDIAs Build backt `ISAACSIM_HOST` und die beiden Ports
fest in die App ein — sie sind Build-Zeit, nicht Laufzeit. `webview` schreibt sie deshalb in
den Image-Tag (`groot-webview:<ip>-49100-47998`): ändert sich die Server-IP, entsteht
automatisch ein anderer Tag, es wird neu gebaut und ein Container aus dem alten Tag wird
ersetzt. Ohne das zeigte eine alte Seite stumm ins Leere.

Der Build patcht die generierte `src/main.ts` per `sed` — ein fragiler Griff in generierten
Code. Unser Dockerfile **prüft danach, ob der Patch gegriffen hat**, und bricht sonst ab;
NVIDIAs Original tut das nicht und lieferte im Fehlerfall eine App, die still auf
`127.0.0.1` zeigt. Der Generator ist deshalb auf `@nvidia/create-ov-web-rtc-app@1.14.2`
gepinnt.

Genau diese Prüfung hat sich sofort bezahlt gemacht: `--name` legt ein **Unterverzeichnis**
an, die App landet also in `/app/web-viewer` und nicht in `/app`. NVIDIAs eigenes Dockerfile
arbeitet an der Stelle mit `WORKDIR /app` — dort gibt es die Datei gar nicht. Unser Build
setzt `WORKDIR /app/web-viewer`. Zwei weitere Abweichungen kamen beim Testen dazu: kein
`apt-get` (der Generator braucht weder git noch curl, und der apt-Schritt war die einzige
Stelle, die an einem gefilterten Netz scheiterte) und ein Healthcheck über Node statt curl.

```bash
./Simulation/server_rl_run.sh webview stop      # beenden (Image bleibt)
./Simulation/server_rl_run.sh clean             # entfernt ihn mit
WEBVIEW_PORT=8211 ./Simulation/server_rl_run.sh webview   # anderer Port
```

> **Reifegrad (2026-08-17).** Geprüft ist mehr als beim Rest der LIVE-Variante — der Build
> braucht keine GPU und lief lokal komplett durch:
>
> | | |
> |---|---|
> | Image baut durch (alle 7 Stufen) | ✓ ~30 s |
> | Patch-Verifikation meldet Erfolg | ✓ |
> | Seite liefert HTTP 200, Healthcheck `healthy` | ✓ |
> | IP/Ports **nachweislich im ausgelieferten JS-Bundle** (`signalingServer:"…"`, `mediaPort:47998`) | ✓ |
> | WebRTC-Handshake mit Isaac Sim | ✗ **offen** — braucht GPU + laufenden Sim |
> | Bibliothek gegen unser Kit aus dem *isaac-lab*-Image (NVIDIA testet gegen *isaac-sim*) | ✗ offen |
>
> Der letzte Schritt ist derselbe, der auch für den nativen Client noch aussteht.

### Falls der Viewport schwarz bleibt, obwohl UDP offen ist

NVIDIAs Docker-Anleitung sagt zu Isaac Sim 6.0 ausdrücklich: **„`--network=host` is required
for WebRTC livestreaming"** — das Streaming-SDK brauche direkten Zugriff auf die
Netzwerk-Interfaces, um seine UDP-Sockets korrekt zu binden. Wir fahren seit jeher Bridge
mit 1:1-Port-Mapping, was theoretisch reichen sollte (intern == extern, damit die
SDP-Aushandlung stimmt) — verifiziert ist es für WebRTC bei uns aber nicht.

Bleibt das Bild schwarz, obwohl `nc -vzu` durchkommt, ist das der erste Verdacht:

```bash
RL_NETWORK_MODE=host ./Simulation/server_rl_run.sh clean
RL_NETWORK_MODE=host LIVESTREAM=2 ./Simulation/server_rl_run.sh view
```

`--network` wirkt nur beim **Anlegen** des Containers, daher das `clean` (Daten bleiben).
Im Host-Modus entfallen die Port-Mappings — alles lauscht direkt auf den Host-Ports.
Default bleibt bewusst `bridge`: ein stiller Wechsel des Netzwerkmodells wäre später nicht
mehr aus den Messergebnissen herauszurechnen.

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

Die reine Policy-Latenz auf echter Hardware (ohne Rendering, `policy_latency.py`) und der
Denoising-Schritte-Sweep sind reine Modell-Performance-Analysen ohne Sim-Bezug und stehen
seit 2026-08-18 als eigener Abschnitt in
[inferenz-optimierung.md](inferenz-optimierung.md#latenz-messung-und-denoising-schritte) —
dort passen sie besser neben die anderen Inferenz-Optimierungen (TensorRT, `torch.compile`).

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
| `LIVESTREAM_UPDATE_EVERY_N` | `1` | Nur `rl` und `view`: alle n Steps `simulation_app.update()`; `0` = nie |
| `LIVESTREAM_SETTINGS_STYLE` | `auto` | `auto\|new\|old\|both` — welche Kit-Settings-Pfade gesetzt werden |
| `LIVESTREAM_KIT_ARGS` | — | Manueller Override der kompletten Kit-Settings-Zeile |
| `PUBLIC_IP` | — | Nur `LIVESTREAM=1`; wird sonst gar nicht erst ermittelt |
| `LIVESTREAM_HOST_ADDR` | auto | Server-Adresse für Hinweistext **und** für den `webview`-Build. Auto = erste Zeile von `hostname -I` |
| `RL_NETWORK_MODE` | `bridge` | `host` = Container mit `--network=host` anlegen. Erster Verdacht, wenn der Viewport trotz offenem UDP schwarz bleibt. Verlangt `clean` |

Nur für `webview` (Variante A2, Viewport im Browser):

| Variable | Default | Zweck |
|---|---|---|
| `WEBVIEW_PORT` | `8210` | Host-Port der Viewer-Seite |
| `WEBVIEW_IMAGE` | `groot-webview` | Image-Name; der Tag wird aus IP + Ports gebildet |
| `WEBVIEW_CONTAINER` | `groot-webview` | Container-Name |

Nur für `view` (Schritt 3, Szene ohne Gewichte):

| Variable | Default | Zweck |
|---|---|---|
| `VIEW_NUM_ENVS` | `1` | Parallele Envs. `>1` zeigt das Klon-Gitter |
| `VIEW_DURATION_S` | `3600` | Laufzeit, danach sauberes Ende. Muss `> 0` sein — kein Dauerlauf (`0` wird auf 3600 zurückgedreht) |
| `VIEW_HOST_ASSET_DIR` | `<repo>/data` | Wo `g1_dex3.usd` + `configuration/` auf dem Host liegen |
| `VIEW_ASSET_DIR` | `/data/assets` | Ziel im Container (im Bind-Mount, überlebt `clean`) |
| `EPISODE_LENGTH_S` | `0` | `0` = cfg-Default 300 s bis zum Auto-Reset (Würfel neu gewürfelt) |
| `SCENE_CAM` | `0` | Abweichender Default — sonst `1`. `cam_scene` geht nur ins MP4 |
| `CAM_RES_SCALE` | `0.5` | Abweichender Default — sonst `1`. Hier folgenlos: es wird nichts gemessen |
| `DR_ENABLED` | `0` | Abweichender Default — sonst `1`. Stabile Beleuchtung zum Ansehen |

Siehe auch: [env-vars.md](../training/env-vars.md) · [livestream-plan.md](../weiterfuehrend/livestream-plan.md) ·
[rl-anleitung.md](../weiterfuehrend/rl-anleitung.md)
