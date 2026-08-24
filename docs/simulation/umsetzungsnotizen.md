# Sim-Eval Implementation Notes — Lessons Learned & aktueller Stand

Dieses Dokument fasst Erkenntnisse zusammen, die beim Aufbau des Sim-Eval-Workflows
gewonnen wurden und in keinem anderen Dokument stehen. Stand: 2026-05-31
(Abschnitte 10–16 ergänzt bis 2026-06-05).

**Lesehilfe:**
- Kalibrierung & aktueller Stand: §10–§11, §13–§15
- Build-/Runtime-Fixes (Referenz): §1–§9
- Doku-Meta: §16

> **Neuere Erkenntnisse (August 2026)** — Isaac-Sim-6.0-Migration, Kamera-Pose-Bug und
> Neukalibrierung (Sichtfeld, Montagepunkt, Stereobasis), Domain-Gap-Neumessung sowie die
> Greif-Diagnostik-Kette (Läufe 9–28) sind in
> [../ergebnisse/diagnose-chronik.md](../ergebnisse/diagnose-chronik.md) dokumentiert.
> Insbesondere gilt der Befund „Greifen validiert" aus §13 unter Isaac Sim 6.0 nicht mehr
> uneingeschränkt (siehe Hinweis dort).

---

## 1. GPU-Kompatibilität — Korrekturen zu `archiv/gpu-kompatibilitaet.md`

### Quadro RTX 5000 (KISSKI `jupyter`-Partition) — **nicht kompatibel**

[`archiv/gpu-kompatibilitaet.md`](archiv/gpu-kompatibilitaet.md) klassifiziert die RTX 5000 als prinzipiell geeignet.
Das ist **falsch für Isaac Lab 2.3.2 / Isaac Sim 5.1**:

- Isaac Sim 4.x setzt **Ampere (RTX 30xx) als Mindestanforderung** voraus.
- Die RTX 5000 ist **Turing** (SM 7.5) — eine Generation zu alt.
- RT-Cores sind vorhanden, aber die Compute-Capability und Treiber-Features reichen
  für Isaac Sim 4.x nicht aus.
- **Konsequenz:** Die Closed-Loop-Sim auf KISSKI ist mit dem aktuellen Isaac-Lab-Image
  nicht möglich. Optionen:
  - Älteres Isaac-Lab-Image (1.x oder frühe 2.x) testen, das Turing noch unterstützt.
  - Externen Anbieter (vast.ai) für die Sim-Eval nutzen (bevorzugte Lösung, s. u.).

### A100 / H100 (KISSKI `kisski`/`kisski-h100`-Partitionen) — keine RT-Cores

Unverändert korrekt: A100/H100 haben keine RT-Cores → Kamera-Rendering scheitert.
Für GR00T-Inference (ohne Sim) auf A100 geeignet, nicht für die Closed-Loop-Sim.

### L40 / RTX 4090 / A6000 (vast.ai) — **empfohlen**

- **L40**: Ada Lovelace, 48 GB GDDR6, 3rd-gen RT-Cores → ideal (~0,7–0,9 $/h)
- **RTX 4090**: Ada Lovelace, 24 GB, RT-Cores → günstiger, VRAM etwas knapp
- **A6000**: Ampere, 48 GB, RT-Cores → ebenfalls geeignet

---

## 2. Container-Architektur — kombinierter Container für vast.ai

### Geplante vs. implementierte Architektur

[`archiv/sim-docker-build.md`](archiv/sim-docker-build.md) beschreibt die ursprünglich geplante **Zwei-Container-Architektur**
(GR00T-Server und Isaac-Lab-Sim-Client als getrennte Images). Diese ist korrekt für KISSKI
(zwei Apptainer-SIFs in einem SLURM-Job).

Für **vast.ai** wurde eine **kombinierte Ein-Container-Architektur** implementiert:

```
Dockerfile.vastai
└── nvcr.io/nvidia/isaac-lab:2.3.2  (Basis)
    ├── Isaac Lab / Isaac Sim        (bereits im Basis-Image)
    └── GR00T N1.6 Policy           (isolierte venv: /app/Groot-1.6/.venv)
```

Beide Prozesse starten autonom via `/scripts/entrypoint_sim.sh`:
1. GR00T-Server im Hintergrund (ZMQ REP auf Port 5555)
2. Isaac-Lab-Sim-Client im Vordergrund (ZMQ REQ)

**Dependency-Isolation:** Kein Konflikt zwischen Isaac-Sim-Python-Bundle und GR00T-venv,
weil jeder Prozess seinen eigenen Python-Interpreter nutzt:
- GR00T: `/app/Groot-1.6/.venv/bin/python`
- Isaac Lab: `${ISAACLAB_PATH}/isaaclab.sh -p` (Isaac Sims eingebettetes Python)

---

## 3. Bekannte Fixes und Stolperfallen

### Fix: `unset VIRTUAL_ENV` vor `isaaclab.sh -p`

**Problem:** Das Dockerfile setzt `VIRTUAL_ENV=/app/Groot-1.6/.venv`. `isaaclab.sh -p`
prüft diese Variable und nutzt dann das GR00T-venv statt seines eigenen Python-Bundles →
`ModuleNotFoundError: No module named 'isaaclab'`.

**Fix:** Vor jedem `isaaclab.sh`-Aufruf:
```bash
unset VIRTUAL_ENV
${ISAACLAB_PATH}/isaaclab.sh -p ...
```

Bereits in `entrypoint_sim.sh` eingebaut. Bei manuellen Kommandos im Container immer
zuerst `unset VIRTUAL_ENV` ausführen.

### Fix: USD-Konvertierung — output-Pfad darf kein `#` enthalten

**Problem:** Beim ersten Konvertierungsversuch wurde `--output /data/g1_dex3.usd#`
übergeben (Bash-Kommentarzeichen am Zeilenende durch Copy-Paste). Das `#` wurde
wörtlich in den Dateinamen übernommen → `g1_dex3.usd#` erzeugt, USD nicht ladbar.

**Fix:** Pfad ohne `#` übergeben. Bereits in `vastai-anleitung.md` korrigiert.

### USD-Asset ist ein Dateibündel, nicht eine einzelne Datei

Die Konvertierung erzeugt **5 Dateien** die zusammen gehören:

```
g1_dex3.usd                          (1,4 KB — Root, relative Referenzen)
configuration/
  g1_dex3_base.usd                   (38 MB — embedded Mesh-Geometrie)
  g1_dex3_physics.usd                (15 KB)
  g1_dex3_robot.usd                  (3,6 KB)
  g1_dex3_sensor.usd                 (0,7 KB)
```

`g1_dex3.usd` referenziert die `configuration/`-Dateien mit **relativen Pfaden** →
`configuration/` muss immer im selben Verzeichnis wie `g1_dex3.usd` liegen.

Alle 5 Dateien sind auf HuggingFace unter `luca-mue/groot-g1dex3-checkpoint` abgelegt.

### Vulkan in WSL2-Docker — nicht nutzbar für Isaac Sim

Auf Windows mit Docker Desktop (WSL2-Backend):
- CUDA funktioniert (NVIDIA-Container-Runtime)
- Vulkan ist **nicht** im WSL2-Treiber enthalten (`libnvidia-vulkan-producer.so` fehlt)
- `gfxstream_vk_icd.json` (WSL2 Vulkan-Schicht) unterstützt nicht die Vulkan-Extensions
  die Isaac Sim 5.x benötigt

**Konsequenz:** Isaac Sim in Docker auf Windows/WSL2 nicht für die Sim-Eval nutzbar.
USD-Konvertierung läuft trotz Vulkan-Fehlern durch (Fehler sind nicht-fatal für diesen
Use-Case), Sim-Eval hingegen scheitert.

**Empfehlung:** vast.ai für alle Isaac-Sim-Läufe nutzen.

---

## 4. USD-Asset erzeugen — lokale Methode (Windows + Docker Desktop)

Trotz Vulkan-Problemen funktioniert die **URDF→USD-Konvertierung** lokal auf Windows
mit Docker Desktop, weil sie kein funktionierendes Rendering braucht:

```powershell
# Im Projektrepo-Root (PowerShell):
docker run -it --rm --gpus all --ipc=host --shm-size=8g `
  -v "${PWD}\data:/data" `
  -e NVIDIA_DRIVER_CAPABILITIES=all `
  --entrypoint bash `
  lucam03/projekt-humanoider-roboter-sim-vastai:latest
```

```bash
# Im Container:
unset VIRTUAL_ENV
${ISAACLAB_PATH}/isaaclab.sh -p \
    /workspace/g1_dex3_sim/convert_urdf_to_usd.py \
    --headless \
    --urdf /data/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf \
    --output /data/g1_dex3.usd
```

Voraussetzung: `git submodule update --init data/unitree_ros`

---

## 5. Fix: GR00T-Server startet nicht — `transformers`-Import scheitert an Triton-gcc

**Symptom:** Beim Start des GR00T-Servers importiert `transformers` → Triton → versucht
`cuda_utils.cpython-312-*.so` mit gcc zu kompilieren → `returned non-zero exit status 1` →
`RuntimeError: Failed to import transformers.modeling_utils`.

Am lokal gebauten Image (`docker run --entrypoint bash …`) **definitiv verifiziert** — drei
nacheinander auftretende, unabhängige Ursachen (1+2 am gcc-Befehl ablesbar:
`-I/usr/include/python3.12 … -lcuda`, 3 nach deren Behebung):

### Ursache 1 (primär): Python-Versions-Mismatch 3.12 vs. 3.10

| | Wert |
|---|---|
| venv-Python | **3.12.3** (`/app/Groot-1.6/.venv/lib/python3.12/`) |
| installierte Header | nur `python3.10-dev` → `/usr/include/python3.10/Python.h` |
| `/usr/include/python3.12/Python.h` | **fehlt** |

Das Dockerfile installiert absichtlich `python3.10` + `python3.10-dev` („kompatibel mit
uv.lock des Training-Containers"), aber `uv sync` bekam **nie `--python` gesagt**. Da das
isaac-lab-Basisimage (Ubuntu 24.04) als `python3` die **3.12** mitbringt, baute uv eine
3.12-venv. Triton ruft dann `gcc -I/usr/include/python3.12` auf → `Python.h` fehlt → gcc scheitert.

**Das erklärt, warum der Training-Container funktioniert, der Sim-Container nicht:**
Training-Basis liefert Python 3.10 → passende Header; Sim-Basis liefert 3.12 → keine Header.

**Fix:** venv explizit mit 3.10 anlegen, bevor `uv sync` läuft:
```dockerfile
RUN uv venv --python /usr/bin/python3.10 /app/Groot-1.6/.venv
RUN ... uv sync --frozen --no-install-project --extra dev --no-cache
```
Plus Build-Time-Smoke-Test, der `Python.h` zur venv-Version prüft → Build bricht lokal
(kostenlos) ab statt erst auf vast.ai.

### Ursache 2 (sekundär): `libcuda.so` (unversioniert) fehlt — für `-lcuda`

Im isaac-lab-Image gibt es **kein** `libcuda.so`, `libcuda.so.*` und kein
`/usr/local/cuda/lib64/stubs/`. Zur Laufzeit injiziert das NVIDIA-Container-Runtime nur
`libcuda.so.1`. gcc braucht für `-lcuda` aber die unversionierte `libcuda.so`.

**Fix (Runtime, in `entrypoint_sim.sh`):** Symlink `libcuda.so → libcuda.so.1` anlegen,
nachdem der Container mit GPU gestartet ist (vorher existiert `libcuda.so.1` nicht):
```bash
LIBCUDA_SO1=$(ldconfig -p | grep "libcuda\.so\.1" | awk '{print $NF}' | head -1 || true)
ln -sf "$LIBCUDA_SO1" /usr/lib/x86_64-linux-gnu/libcuda.so
```

> Hinweis: Frühere Versuche mit `LD_LIBRARY_PATH` bzw. `LIBRARY_PATH=/usr/local/cuda/lib64/stubs`
> waren wirkungslos — das Stubs-Verzeichnis existiert in diesem Image gar nicht.

### Ursache 3: DeepSpeed verlangt CUDA-Toolchain — `CUDA_HOME does not exist`

Nachdem Ursache 1+2 behoben waren, scheiterte der Import an einer dritten Stelle:
```
RuntimeError: Failed to import transformers.modeling_utils ...
CUDA_HOME does not exist, unable to compile CUDA op(s)
```
Quelle: `deepspeed/ops/op_builder/builder.py` → `installed_cuda_version()`. `transformers.modeling_utils`
macht beim Import `if is_deepspeed_available(): import deepspeed`; `import deepspeed` prüft die
CUDA-Toolchain. Das isaac-lab-Image hat **kein** `nvcc`, **kein** `/usr/local/cuda`, **kein**
`CUDA_HOME` → `MissingCUDAException`.

Der Training-Container funktioniert, weil dessen CUDA-devel-Basisimage `nvcc` + `CUDA_HOME` mitbringt.

**Fix:** `deepspeed` aus der Sim-venv entfernen (Dockerfile, nach `uv sync`):
```dockerfile
RUN uv pip uninstall --python /app/Groot-1.6/.venv/bin/python deepspeed || true
```
DeepSpeed ist eine reine Trainings-Bibliothek (ZeRO); GR00T importiert sie für die Inferenz
nirgends direkt. Ohne das Paket liefert `is_deepspeed_available()==False` → transformers
überspringt den Import. Alternative (CUDA-Toolkit installieren) würde das Image um >2 GB aufblähen.
Build-Time-Smoke-Test prüft, dass `import deepspeed` fehlschlägt.

### Lokal auf GPU verifiziert ✅

Auf einer lokalen GPU (RTX 4070 Laptop, mit `docker run --gpus all`) end-to-end bestätigt
(Image mit Python-3.10-Fix; libcuda-Symlink + deepspeed-uninstall wie im Entrypoint/Dockerfile):
- `from transformers import PreTrainedModel` → **OK**
- `import gr00t.model` (volle Server-Importkette) → **OK**
- `run_gr00t_server.py --help` → tyro parst `ServerConfig`, `NEW_EMBODIMENT` ist gültig, kein `--no-flash-attn`

> **Noch offen (nur mit echter Sim-GPU testbar):** das eigentliche Modell-Gewichte-Laden
> (~6–8 GB VRAM) und die Isaac-Lab-Sim-Schleife. Der komplette Server-**Start**pfad bis zum
> Modell-Load ist aber lokal grün.

---

## 6. Fix: `NO_FLASH_ATTN` / `--no-flash-attn` ist kaputt — Flash-Attn ist Pflicht

**Symptom (latent):** Der Entrypoint übergab bei `NO_FLASH_ATTN=1` das Flag `--no-flash-attn`
an `run_gr00t_server.py`. Die Anleitung empfahl genau das als Fix bei „flash-attn-Fehlern".

**Verifiziert am Image:** Zwei Gründe, warum das nicht funktioniert:
1. `run_gr00t_server.py` nutzt eine `tyro`-`ServerConfig`-Dataclass **ohne** Flash-Attn-Feld →
   `--no-flash-attn` ist ein unbekanntes Argument → Server crasht sofort.
2. `gr00t/model/modules/eagle_backbone.py` enthält ein hartes
   `assert use_flash_attention, "nvidia/Eagle-Block2A-2B-v2 requires flash attention by default"`
   → das Modell **kann** gar nicht ohne Flash-Attention 2 laufen.

`flash_attn 2.7.4.post1` ist im venv installiert.

**Fix:** Flash-Attn-Flag-Logik aus dem Entrypoint entfernt; `NO_FLASH_ATTN` wird nur noch mit
Warnung ignoriert. `NO_FLASH_ATTN`-Zeile + Troubleshooting-Tipp aus `vastai-anleitung.md`
entfernt. **Konsequenz:** Nur GPUs mit Flash-Attn-Support (Ampere+) sind nutzbar — bestätigt die
ohnehin bestehende GPU-Anforderung.

---

## 7. Fix: Sim-Client — `ModuleNotFoundError: No module named 'zmq'`

Nachdem der **Server** sauber startete (alle Server-Fixes wirken), scheiterte **Schritt 3/3**
(Isaac-Lab-Sim-Client) sofort beim Import:
```
File "/workspace/g1_dex3_sim/client.py", line 21, in <module>
    import zmq
ModuleNotFoundError: No module named 'zmq'
```

**Ursache:** Server und Sim-Client laufen in **zwei verschiedenen Python-Umgebungen**:
- GR00T-Server: `/app/Groot-1.6/.venv/bin/python` (3.10) — hier wurde `pyzmq` installiert
- Sim-Client: Isaac Sims gebündeltes Python `/workspace/isaaclab/_isaac_sim/python.sh` (**3.11**)
  via `isaaclab.sh -p` — hier fehlte `pyzmq`.

`client.py` braucht `msgpack`, `numpy`, `zmq`. In Isaac Sims Python sind `numpy` (1.26) und
`msgpack` (1.1.2) bereits vorhanden, nur `pyzmq` fehlte.

**Fix (Dockerfile):** `pyzmq` zusätzlich in Isaac Sims Python installieren:
```dockerfile
RUN unset VIRTUAL_ENV && /workspace/isaaclab/_isaac_sim/python.sh -m pip install pyzmq msgpack
```
Plus Build-Time-Smoke-Test (`import zmq, msgpack, numpy` unter `python.sh`). Am Image verifiziert:
nach Installation `import zmq` → OK (pyzmq 27.1.0).

### Zusätzliche Härtung: maskierte Crashes

`isaaclab.sh` **schluckt den Exit-Code** des Python-Sim-Clients — der zmq-Crash führte trotzdem
zu „Sim-Eval abgeschlossen" + Exit 0. Im Entrypoint deshalb nach dem `isaaclab.sh`-Aufruf
geprüft, ob `results.json` tatsächlich (nicht leer) existiert; sonst Fehler + Exit 1.

---

## 8. Fix: SSH-Server im Entrypoint

**Problem:** Docker-ENTRYPOINT-Modus auf vast.ai startet keinen SSH-Server → `vastai ssh-url`
schlägt fehl mit "ssh port not found". OpenSSH war im isaac-lab-Image gar nicht installiert.

**Fix (drei Teile):**
1. `openssh-server` im Dockerfile installieren + `ssh-keygen -A` (Host-Keys erzeugen).
2. Im Entrypoint `$PUBLIC_KEY` (von vast.ai injiziert) in `authorized_keys` schreiben und
   `/usr/sbin/sshd` direkt starten (kein `service`/init.d im Container):
   ```bash
   [[ -n "${PUBLIC_KEY:-}" ]] && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
   /usr/sbin/sshd
   ```
3. In den Docker-Options der Instanz Port 22 freigeben: `--ipc=host --shm-size=16g -p 22`.
   Ohne `-p 22` mappt vast.ai keinen externen Port und `vastai ssh-url` findet nichts.

SSH-Verbindung danach via `vastai set api-key <key>` + `vastai ssh <instance-id>`.

---

## 9. Fix: Eval-Schleife lauffähig + beobachtbar (Tuple/List, Videos, Buffering)

Nachdem der Sim-Client startete, drei Probleme in der Eval-Schleife selbst:

### 9a. `get_action`: Tuple wird als Liste übertragen
`BasePolicy.get_action` (policy.py) gibt `(action_dict, info_dict)` zurück; msgpack serialisiert
das Tuple als **Liste** `[action_dict, info_dict]`. Der Client (`client.py`) griff mit
`resp["action.left_arm"]` zu → `TypeError: list indices must be integers or slices, not str`.
**Fix:** `action_dict = resp[0] if isinstance(resp, (list, tuple)) else resp`.

### 9b. Videos wurden nie gespeichert
`save_episode_video()` war definiert, aber **nie aufgerufen** — die Frames wurden in
`run_episode` gesammelt und beim Return verworfen. **Fix:** `run_episode` bekommt `episode` +
`video_dir` und ruft `save_episode_video(frames, episode, video_dir)` vor dem Return.
`imageio` (2.37) + `imageio_ffmpeg` (0.6) sind in Isaac-Sim-Python vorhanden.

### 9c. Kein sichtbarer Fortschritt (Buffering)
`isaaclab.sh` führt Pythons stdout als Pipe → `print()` block-gepuffert; sah aus wie ein Hang.
**Fixes:** `export PYTHONUNBUFFERED=1` im Entrypoint + Fortschritts-Print alle 25 Steps in
`run_episode` (`flush=True`). Zusätzlich: `isaaclab.sh` schluckt den Python-Exit-Code →
Entrypoint prüft jetzt, ob `results.json` (nicht leer) existiert, sonst Fehler.

> **Meilenstein:** Mit 9a–9c lief die Pipeline erstmals **vollständig end-to-end** durch —
> 20 Episoden, `results.json` + Videos geschrieben, kein Crash (RTX 6000 Ada / L40S).
> 0/20 Erfolge mit checkpoint-3000 (erwartet, s. u.).

---

## 10. Kamera-Rekonstruktion aus dem Dataset

**Befund (aus dem aufgenommenen Video):** Die High-Kameras zeigten nur den Boden, verkippt.
Ursache zweifach in `g1_dex3_cfg.py`:
- Position `x=−0.5` (hinter dem Ursprung; der Arbeitsbereich liegt bei **+X**: Tisch (0.5,0,0.37), Würfel z≈0.77).
- Rotation `(0.924,−0.383,0,0)` = reine **Roll-Drehung um die +X-Blickachse**
  (`convention="world"` → Blick=+X, oben=+Z), **kein Pitch nach unten**.

Da dieselben Kamerabilder als Policy-Observation dienen, war die 0-%-Eval damit **kein
gültiges Modell-Urteil** — die Policy bekam Boden-Bilder.

**Rekonstruktion:**
- Dataset: `unitreerobotics/G1_Dex3_BlockStacking_Dataset` (LeRobot **v3.0**, 30 fps, 480×640, 4 Kameras).
- Video-Pfad: `videos/observation.images.<cam>/chunk-000/file-000.mp4` (~500 MB, **faststart**).
  Referenz-Frame (Frame 0) je Kamera per **HTTP-Byte-Range (~12 MB)** gezogen — kein Voll-Download nötig.
  Versioniert abgelegt unter `Simulation/camera_reference/dataset_cam_*.png`.
- Echte High-Sicht: Kopf-Stereo-Paar, vorne-oben, ~50° nach unten auf den Tisch; beide Hände von unten im Bild.
- Fix: Helper `look_at_world_quat(eye, target)` in `g1_dex3_cfg.py` (richtet Kamera per Look-at aus,
  Posen über `eye`/`target` statt Quaternionen tunebar). High-Kameras:
  `eye≈(0,±0.06,1.40)`, `target=(0.5,0,0.73)` → Pitch **53°** (Mathematik verifiziert).

**Ergebnis (Render verifiziert):** Tisch + 3 Würfel + Hände korrekt im Bild, Struktur wie Referenz.

**Sim-Treue — Stand:**

- ✅ **Roboter-Startpose**: `DATASET_INIT_STATE` = `observation.state` aus Frame 0 (28 Werte) übernommen.
- ✅ **Tisch weiß** (`diffuse_color (0.85,0.85,0.85)`).

> **Hinweis:** Die `eye`/`target`/Tisch-Werte oben sind die ERSTE Rekonstruktion. Sie wurden
> in Abschnitt 11 nach dem Tisch-Anheben aktualisiert (High-Target → `(0.40,0,0.86)`); alle 4
> Kameras danach erneut am Render verifiziert (gut).

**Dex3-Finger-Gelenklimits (KEIN Vorzeichen-Flip!):** ✅ **erledigt.** Über alle 281.196 Frames
geprüft: Dataset-Vorzeichen stimmen mit der URDF überein (links Beugen negativ, rechts positiv)
→ Greif-Richtung korrekt. Aber die echte Range überschreitet die URDF-Limits um ~0,2–0,33 rad
(z. B. left middle_1 bis −2,08 vs URDF −1,75) → volles Schließen würde geklemmt. **Fix:**
`_widen_finger_joint_limits()` in der Env hebt zur Laufzeit die Grenzen von 10 Finger-Gelenken
an die Dataset-Range an (`write_joint_position_limit_to_sim`). Init-Pose-Clamp (index_0/middle_0 → 0.0) korrekt.

**Wrist-Kameras:** ✅ **erledigt** — rekonstruiert + per Render verifiziert. Look-at im Link-Frame
(`convention="world"`, `eye=(-0.08,0,0.13)`). Behob „nur Grau" (Kamera steckte im Palm-Mesh) +
falsch-herum rechte Cam. Getrennte Targets wegen asymmetrischer Arm-Startpose: links
`(0.14,0,-0.18)` (55° runter), rechts `(0.16,0,-0.05)` (37°).

**Szenen-Übersichtskamera (`cam_scene`):** NUR fürs aufgenommene Video (NICHT Policy-Observation).
Weltfeste 5. Kamera, schräg vorne-seitlich-oben (`eye=(1.8,1.6,1.7)`, `target` auf Szenenmitte),
weiter FOV (focal 18) → zeigt Roboter + Tisch komplett. In `g1_dex3_cfg.py` (`cam_scene`),
in der Env als `TiledCameraCfg`, in `self.cameras` aufgenommen; `run_g1_dex3_sim_eval.py` nimmt
das Video aus `video.cam_scene` auf (Fallback `cam_left_high`). Die 4 Policy-Cams bleiben unberührt.

---

## 11. Diagnose: Open-Loop-Replay — Sim führt Aktionen korrekt aus

Frage: Liegt das Wegdriften der Hände im Closed-Loop am Modell oder an einem Sim-/Config-Bug?

**Werkzeug** (additiv, überschreibt nichts): `run_g1_dex3_replay.py` + `replay_episode0.npz`
(echte Dataset-Aktionen Episode 0, 1173 Frames) + `entrypoint_replay.sh`. Spielt die echten
Ground-Truth-Aktionen open-loop in dieselbe Env (kein Server, kein Modell), getrennte Ausgaben
(`/data/sim_videos_replay`, `/data/sim_results_replay`).

**Ergebnis (gemessen):** Tracking-Fehler Arm-Gelenke **mittel = 0,022 rad** (~1,3°),
worst-case 0,10–0,13 rad. → Der Roboter folgt den kommandierten Gelenkwinkeln **präzise**.

**Schlussfolgerung:**
- Aktuator-Gains, Joint-Mapping, Konvention, Aktions-Anwendung (absolut) sind **korrekt** —
  **kein Bug im Steuerungs-Pfad**. (Eine PD-Gain-Erhöhung wäre ein Fehler gewesen.)
- Replay stapelt nicht (`success=False`) → **erwartet**: Open-Loop kann sich nicht an die
  (zufälligen, nicht dataset-gematchten) Würfelpositionen anpassen; Objekt-Posen sind im Dataset
  nicht gespeichert. Kein Sim-Bug.
- **Das Wegdriften im Modell-Eval ist das Modell** (checkpoint-3000 untertrainiert), nicht der
  Steuerungs-Pfad. Die Sim fährt Trajektorien treu nach.

### KRITISCHER FUND (durch Greif-Diagnose): Tisch zu tief → Würfel unerreichbar

Die Greif-Diagnose im Replay (min Hand→Würfel-Distanz, Würfel-Anhebung, tiefster Greifpunkt)
deckte einen echten Geometrie-Bug auf, den kein Training ausgleichen könnte:

- Tiefster Handpunkt der echten Greif-Trajektorie: **z ≈ 0,92 m**. Alter Tisch: Oberseite 0,74,
  Würfel-Mitte 0,77 → die Würfel lagen **~15 cm UNTER dem erreichbaren Arbeitsraum**.
- min Hand→Würfel-Distanz war **21,6 cm** — die Hände kamen nie auch nur nah an die Würfel.
- ⇒ Selbst ein perfekt trainiertes Modell hätte nie greifen können.

**Fix (Geometrie):**
- Tisch-Oberseite **0,74 → 0,87** (Würfel sitzen jetzt bei z≈0,895, im Greifraum).
- Würfel-Startpos + Sampling in den erreichbaren Bereich: `block_x_range (0.30,0.40)`,
  `block_y_range (-0.20,0.20)`, `block_z_surface 0.895`.
- High-Cam-Ziel `(0.5,0,0.73)→(0.40,0,0.86)` + Szenen-Cam-Ziel angehoben (Tisch ist höher).
- Roboter NICHT abgesenkt (die weltfesten Kameras sind auf Becken z=0.85 kalibriert).

**Nach dem Fix (Greif-Test `--grasp-test`, Würfel exakt an die Greifpunkte gesetzt):**
min Hand→Würfel-Distanz **5,9 cm** (war 21,6), Würfel-**Anhebung 1,0 cm**, Würfel sichtbar
verschoben → **Kontakt- und Greif-Physik funktionieren** (kein kaputtes Reibungs-/Kollisionsmodell).
Kein fester Griff im Open-Loop, weil die Hand sich ohne visuelle Rückkopplung nicht exakt auf den
Würfel ausrichtet (systembedingt) — ein trainiertes Closed-Loop-Modell korrigiert das.

**Fazit:** Der einzige verbliebene Sim-Bug (Tischhöhe) ist behoben; die Sim kann greifen.
Ab hier ist **Training der Hebel**. Optionale Feinschritte: Würfel-Reibungsmaterial erhöhen
(festerer Griff), Greifpunkt-Platzierung verfeinern.

---

## 12. Aktueller Stand (2026-06-04)

§12 (Momentaufnahme 2026-06-04) → ausgelagert nach [`../historie.md`](../historie.md).

---

## 13. Physics Calibration Session (2026-06-04)

Diese Session hat die Greif-Physik der Sim systematisch kalibriert, ausgehend von
`max_cube_lift = 1,0 cm` (kein Greifen). Endergebnis: **2,8 cm — Greifen validiert**.

> ⚠️ **Einschränkung (2026-08-08) — inzwischen aufgelöst:** Das Ergebnis „2,8 cm — Greifen
> validiert" wurde unter Isaac Sim 4.x gemessen. Nach der Isaac-Sim-6.0-Migration hob derselbe
> Replay-Testtyp zunächst **keinen Würfel mehr an** (0,0 cm, trotz vollständigem Fingerschluss).
>
> **Aufgelöst mit Lauf 29 (2026-08-12):** Ursache war der Referenzpunkt der Greif-Diagnostik —
> gemessen wurde am distalen Gelenk statt an den Fingerspitzen. Mit korrigiertem Referenzpunkt
> hebt die Hand einen Würfel **7,9 cm**; die Greif-Physik war nie defekt. Details:
> [../ergebnisse/diagnose-chronik.md](../ergebnisse/diagnose-chronik.md) (Läufe 25–29).

### 14.1 Finger-Aktuatoren (`g1_dex3_cfg.py`)

| Parameter | Vorher | Nachher | Begründung |
|---|---|---|---|
| `stiffness` (Hände) | 20,0 | **60,0** | Finger schlossen Distal-Joints nicht (max_error Joint 18/27 = 0,74/0,88 rad) |
| `effort_limit` (Hände) | 5,0 N·m | **20,0 N·m** | 50 g Würfel gegen Schwerkraft halten erfordert >5 N·m |
| `damping` (Hände) | 2,0 | **4,0** | Proportional zur neuen stiffness |
| `solver_position_iter` | 4 | **8** | Bessere Kontaktauflösung bei Mehrfach-Kontakt (Finger + Würfel) |
| `solver_velocity_iter` | 0 | **1** | Stabilerere Kontaktdynamik |

### 14.2 Der „Sign-Convention-Fix" war selbst der Fehler (zurückgenommen 2026-08-24)

> Bis 2026-08-24 stand hier ein Fix, der `middle_0` und `index_0` **beider** Hände negierte
> (`_SIGN_FLIP_IDX = [17, 19, 24, 26]`), dazu negierte Startwerte in `DATASET_INIT_STATE` und
> die ausdrückliche Entscheidung, diese vier Gelenke **nicht** zu weiten. Die zugrunde liegende
> Annahme — „Datensatz: positiv = schließen" — ist falsch. Alles davon ist zurückgenommen.

**Was tatsächlich gilt:** Der Datensatz ist bereits **seitenweise in USD-Konvention**
aufgezeichnet: links negativ = schließen, rechts positiv = schließen. Die `_1`-Beugegelenke
belegen es — sie wurden nie gespiegelt und passen trotzdem beide exakt in ihre Grenzen
(`meta/stats.json`, ganzer Datensatz):

| Gelenk | Datensatz | Grenze |
|---|---|---|
| `left_hand_index_1` | −2,083 … −0,008 | −2,13 … 0,05 |
| `right_hand_index_1` | 0,010 … 2,085 | −0,05 … 2,14 |
| `left_hand_index_0` | −1,089 … 0,267 | −1,571 … 0,0 |
| `right_hand_index_0` | −0,199 … 1,646 | 0,0 … 1,571 |

**Was die Spiegelung anrichtete:** Sie drehte die `_0`-Werte aus ihrer Grenze heraus.
`set_joint_position_target` klemmt dort auf 0 — das Gelenk bewegte sich **überhaupt nicht**.
Nicht „in Öffnungsrichtung", wie die alte Notiz vermutete, sondern gar nicht. Gemessen mit
`server_rl_run.sh tipcheck` (Episode 0): auf den linken `_0`-Gelenken klemmten 550 bzw. 587
von 1173 Frames. Die Spiegelung wirkte in `_get_observations` **und** `_pre_physics_step`, also
in jedem Lauf — Eval, Grasp, RL, Replay, Co-Training.

**Symptom über die ganze Kette:** Hand schließt nie → 101/116 Griffen bleiben über 6 cm Öffnung
→ die v4-Greifanker sind unbrauchbar → der darauf gebaute `close_step`-Detektor misst Rauschen.

**Stand jetzt:**
- `_SIGN_FLIP_IDX = []` — keine Spiegelung, auf keiner Seite.
- `DATASET_INIT_STATE`: alle 28 Werte roh aus dem Datensatz, keine Umrechnung.
- `_widen_finger_joint_limits`: die vier `_0`-Gelenke sind wieder drin. Beide Hände fahren
  real ~0,2 rad über die Nulllinie in die Gegenrichtung, rechts `index_0` zusätzlich über
  1,571 hinaus. Grenzen = Union(USD, Datensatz-Min/Max) + ~0,05 rad, wie bei den `_1`-Gelenken.

Maßgeblich ist dabei `observation.state` (was das Gelenk erreicht **hat**), nicht `action`:
kommandiert wurde stellenweise deutlich mehr (`left_index_0` bis −1,762 gegen −1,089 erreicht),
das hat auch die reale Hand nicht ausgefahren.

### 14.3 Würfel-Reibung (`g1_dex3_blockstack_env.py`)

Standard-PhysX-Reibung (~0,5) ließ den Würfel trotz korrektem Griff herausgleiten.

| Parameter | Vorher | Nachher |
|---|---|---|
| `static_friction` | 0,5 (default) → 1,5 | **3,0** |
| `dynamic_friction` | 0,5 (default) → 1,2 | **2,5** |
| `restitution` | default | **0,0** |

Auf allen 3 Würfeln gesetzt. Wert 3,0 entspricht gummierter Greiffläche — für 50 g Würfel
notwendig, um Haltekraft während der Arm-Bewegung (Trägheitskräfte) zu gewährleisten.

### 14.4 Tischhöhe und Würfelposition

**Diagnose aus Replay-Logs:**
```
tiefster left-Hand-Punkt:  x=0.305  y=0.200  z=0.937
tiefster right-Hand-Punkt: x=0.350  y=-0.169 z=0.944
Würfel-Oberseite (Zentrum): z≈0.915  →  tatsächliche Oberkante z=0.940
```

Die Handflächen erreichen z ≈ 0,937–0,944. Würfel-Oberkante muss auf dieser Höhe liegen.

| Konfiguration | Vorher | Nachher |
|---|---|---|
| Tischhöhe | 0,87 m | **0,87 m** (unverändert — höherer Tisch blockiert Arme im Closed-Loop) |
| `block_z_surface` | 0,895 | **0,915** (Oberkante z=0,940 ≈ tiefster Handpunkt) |
| Würfel-Initialpositionen (z) | 0,895 | **0,915** |
| Grasp-Test grüner Würfel | (0,36, −0,18) | **(0,37, −0,16)** (rechte Handposition gemessen) |

> **Wichtig:** Tisch über 0,87 m anheben blockiert die Arme im Closed-Loop — Modell versucht
> zu z ≈ 0,915 zu greifen, Tischkollision stoppt die Arme früher → Hände liegen auf dem Tisch,
> Roboter zappelt. Cube-z-surface erhöhen ohne Tischhöhe zu ändern ist die korrekte Lösung.
> Die Würfel „schweben" dabei 2,5 cm über der Tischoberfläche — für die Sim-Eval akzeptabel.

### 14.5 Replay-Ergebnisse vor/nach der Kalibrierung

| Metrik | Vor Session | Nach Session |
|---|---|---|
| `max_cube_lift_cm` | 1,0 cm | **2,8 cm** ✅ |
| `min_hand_cube_dist_cm` | 5,9 cm | 7,3 cm (Würfel durch Kontakt verschoben) |
| Arm-Tracking (mittel) | 0,021 rad | 0,020 rad (unverändert gut) |
| Diagnose | Greifen scheitert | **Greifen bestätigt** (>2 cm Schwelle) |

Die erhöhte Endposition der Würfel (`(0,38, 0,15)` statt `(0,35, 0,20)` Startposition) bestätigt
echten Kontakt: Würfel wurden tatsächlich bewegt, nicht nur gestreift.

### 14.6 Closed-Loop-Eval (checkpoint-175000) — Diagnose

**Beobachtetes Verhalten:** Hände liegen auf dem Tisch, Roboter führt kleine, ungerichtete
Bewegungen aus, keine Greifaktion.

**Ursache (bestätigt):** Visueller Domain Gap — der eingefrorene Vision-Encoder produziert
unbrauchbare Features für synthetische Isaac-Sim-Renderings. Das Training ist korrekt
(loss 1,37 → 0,10, W&B-Diagnose: „converged"); die Sim-Config ist korrekt (Replay bestätigt).
Der einzige Unterschied zwischen Replay (funktioniert) und Closed-Loop (versagt) sind
die **Kamerabilder**, die das Modell als Input bekommt.

Detaillierte Analyse: [`lauf1-auswertung.md §8`](../ergebnisse/lauf1-auswertung.md).

---

---

## 14. Domain-Gap-Diagnose-Werkzeuge

Drei Skripte bilden zusammen die Domain-Gap-Diagnose (real vs. Isaac-Sim-Kamerabilder) — die
Kernursache, warum der eingefrorene Vision-Encoder im Closed-Loop versagt (§13.6):

| Schritt | Werkzeug | Zweck |
|---|---|---|
| 1 — Sim-Frames erzeugen | [`g1_dex3_sim/dump_policy_cams.py`](../../Simulation/g1_dex3_sim/dump_policy_cams.py) | Rendert je Policy-Kamera ein `sim_cam_*.png` (im Sim-Container, RT-Core-GPU). |
| 2 — qualitativ vergleichen | [`compare_domain_gap.sh`](../../Simulation/compare_domain_gap.sh) | Montiert pro Kamera das Dataset-Referenzbild (`Simulation/camera_reference/dataset_cam_*.png`) links neben das Sim-Renderbild rechts → ein beschriftetes Vergleichs-PNG. Lokal lauffähig (nur ImageMagick). |
| 3 — quantitativ messen | [`scripts/measure_domain_gap.py`](../../Simulation/scripts/measure_domain_gap.py) | Schickt Real- und Sim-Frames durch SigLIP (identische Gewichte zum frozen GR00T-ViT) und misst die Cosine-Distanz der Embeddings → eine Zahl pro Kamera. |

**Beispiel (compare, lokal):**
```bash
# Sim-Frames aus dem Sim-Container herunterladen, dann:
Simulation/compare_domain_gap.sh ./sim_cam_frames ./domain_gap_compare.png
```

Erstmessung Juni 2026 → [`../historie.md`](../historie.md); gültig ist die Neumessung vom
2026-08-08, siehe [`domain-gap-analyse.md`](../ergebnisse/domain-gap-analyse.md).

---

## 15. Kamera-Kalibrierung: Overlay-Verfahren (2026-06-05)

### Hintergrund

Die Kamera-Posen in [`g1_dex3_cfg.py`](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py) wurden
ursprünglich geschätzt und iterativ gegen Referenz-Frames angepasst. Präzise Hardware-Metadaten
(Mounting-Position, Intrinsics) sind im Dataset **nicht** hinterlegt — die HuggingFace-`info.json`
enthält nur Auflösung (640×480) und FPS (30). Unitree empfiehlt selbst: *"adjust the scene to
closely match the first frame of the dataset."*

### Methodik: Visuelles Overlay-Verfahren

Das **Overlay-Verfahren** ist der zuverlässigste Weg ohne Hardware-Specs:

1. **Referenzbild** aus dem Datensatz (Episode 0, Frame 0 = `Simulation/camera_reference/dataset_cam_*.png`)
2. **Sim-Frame** aus Isaac Sim (Episode 1, Step 0 → `_debug_obs_cam_*.png` in `--video-dir`)
3. **Alpha-Blend**: Real (grün getönt) 50 % über Sim (rot getönt)
   - Perfekte Überlagerung → **gelb** (rot + grün)
   - Misalignment → grüne oder rote Geister
4. **Tuning** von `wrist_eye`, `left_wrist_target`, `right_wrist_target`, `high_eye`, `hfov_deg`
   bis Finger/Tisch übereinander liegen

**Werkzeug:**
```bash
# Lokal ausführen (nur Pillow + NumPy, kein Isaac Sim nötig):
python Simulation/scripts/overlay_camera_check.py \
    --real-dir Simulation/camera_reference/ \
    --sim-dir  Simulation/runs/<run>/           \
    --out-dir  Simulation/runs/<run>/overlay/

# Iterationsloop:
# 1. overlay_camera_check.py → Misalignment sehen
# 2. g1_dex3_cfg.py anpassen
# 3. scp g1_dex3_cfg.py root@<vastai>:/workspace/g1_dex3_sim/
# 4. Replay-Script starten (kein Modell nötig, endet nach ~2 min):
#      GRASP_TEST=1 bash /scripts/entrypoint_replay.sh
#    (oder Modell-Eval; beide schreiben _debug_obs_*.png nach Step 0 ins video-dir)
# 5. Frames herunterladen → zurück zu 1.
```

### Befunde und durchgeführte Korrekturen

#### High-Kameras (`cam_left_high`, `cam_right_high`)

Overlay-Befund: Sim erschien stark herein-gezoomt verglichen mit Real — Real zeigte Tischkante
und Wand-Hintergrund, Sim zeigte Blöcke formatfüllend.

| Parameter | Alt | Neu | Grund |
|---|---|---|---|
| `hfov_deg` | 69° | **90°** | Sim zu eng/nah |
| Eye Z | 1.40 m | **1.60 m** | Kamera höher → mehr Hintergrund sichtbar |
| Eye X | 0.0 m | **−0.10 m** | Weiter vom Tisch zurück |

#### Wrist-Kameras (`cam_left_wrist`, `cam_right_wrist`)

**Befund 1 (vor der Korrektur):** `cam_left_wrist` zeigte den weißen Wrist-Connector
formatfüllend — kein Finger, kein Aufgabenraum sichtbar. Ursache: `target=(0.14,0,−0.18)` war zu
steil nach unten und traf in der asymmetrischen Arm-Startpose den Gelenk-Körper statt vorwärts
zu schauen. Fix: Target auf `(0.18,0,−0.06)` verschoben.

**Befund 2 (nach erstem Fix, Overlay-Analyse):** Beide Wrist-Cams zeigten von **oben nach unten**
auf die Hand. Die Real-Bilder zeigen die Kamera dagegen von **unten-vorne** auf den Wrist-Mechanismus
(Wrist-Joint oben im Bild, Hand zeigt weg → Kamera schaut aufwärts-vorwärts). Kompletter
Perspektiv-Flip.

| Parameter | Alt | Neu | Grund |
|---|---|---|---|
| `wrist_eye Z` | +0.13 m | **−0.06 m** | Unter statt über dem Wrist |
| `wrist_eye X` | −0.08 m | **−0.06 m** | Leicht angepasst |
| `left_wrist_target Z` | −0.06 | **+0.12** | Aufwärts-vorwärts statt abwärts |
| `right_wrist_target Z` | −0.05 | **+0.12** | analog |

### Domain-Gap-Auswirkung

Die Kamera-Fehlposition war (neben echtem Sim-vs-Real-Appearance-Gap) ein wesentlicher Treiber
des `cam_left_wrist`-Gaps von 0.427 (gemessen mit SigLIP, s. [`domain-gap-analyse.md`](../ergebnisse/domain-gap-analyse.md)).
Nach dem Kamerafix zeigte `measure_domain_gap.py` kaum Verbesserung in der Embedding-Distanz
(0.427 → 0.424), weil das SigLIP-Embedding den Bildinhalt global bewertet. Für die **Policy-Qualität**
ist die korrekte Perspektive aber entscheidend — eine Policy, die nur den Wrist-Connector sieht,
kann keine Greifentscheidungen treffen.

---

## 16. Outdated-Hinweise zu anderen Dokumenten

Die folgenden Dokumente sind historisch/überholt und liegen daher im Unterordner
[`archiv/`](archiv/). Sie bleiben als Planungs-/Entscheidungs-Kontext erhalten, sind aber
**nicht** die Quelle der Wahrheit für den aktuellen Stand (das sind §10/§11 dieses Dokuments).

| Dokument | Problem |
|---|---|
| [`archiv/gpu-kompatibilitaet.md`](archiv/gpu-kompatibilitaet.md) | RTX 5000 als "ja" (Isaac-Sim-Rendering) gelistet — ist faktisch **nein** für Isaac Lab 2.3.2 (Turing < Ampere-Mindestanforderung) |
| [`archiv/sim-docker-build.md`](archiv/sim-docker-build.md) | Beschreibt Zwei-Container-Plan; `Dockerfile.vastai` (kombiniert) jetzt primäre Impl. für vast.ai; KISSKI-Zwei-Container bleibt gültig |
| [`archiv/kisski-desktop.md`](archiv/kisski-desktop.md) | Setzt RTX-5000-Kompatibilität voraus — vor Nutzung prüfen ob ältere Isaac-Lab-Version kompatibel ist |
| [`archiv/isaac-lab-plan.md`](archiv/isaac-lab-plan.md) | Ursprünglicher Implementierungs-Plan (historisch). Konkrete Werte (Kamera-Posen, Tischhöhe 0.74, Würfelpositionen, Aktions-Annahmen) sind durch die Umsetzung überholt — **§10/§11 dieses Dokuments sind die Quelle der Wahrheit** für den aktuellen Stand. |

> **Geprüft (2026-06-01) und aktuell:** [`vastai-anleitung.md`](vastai-anleitung.md) (Env-Vars inkl.
> `ASSET_PATH`, `NO_FLASH_ATTN`-Hinweis, `-p 22`, Open-Loop-Replay-Abschnitt), `CLAUDE.md`
> (Architekturbaum mit Replay-Tool/`camera_reference`, Sim-Env-Tabelle). Das Trainings-README und
> die Trainings-Anleitung (`docs/training/`) betreffen nur das Training — von den Sim-Eval-
> Änderungen unberührt.

**Seit 2026-08-18** werden veraltete Inhalte aus diesem und anderen Sim-Dokumenten zentral in
[`../historie.md`](../historie.md) gesammelt, statt sie einzeln in `archiv/` abzulegen.
