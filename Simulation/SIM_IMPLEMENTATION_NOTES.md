# Sim-Eval Implementation Notes — Lessons Learned & aktueller Stand

Dieses Dokument fasst Erkenntnisse zusammen, die beim Aufbau des Sim-Eval-Workflows
gewonnen wurden und in keinem anderen Dokument stehen. Stand: 2026-05-31.

---

## 1. GPU-Kompatibilität — Korrekturen zu SIM_GPU_COMPATIBILITY.md

### Quadro RTX 5000 (KISSKI `jupyter`-Partition) — **nicht kompatibel**

`SIM_GPU_COMPATIBILITY.md` klassifiziert die RTX 5000 als prinzipiell geeignet.
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

`SIM_DOCKER_BUILD.md` beschreibt die ursprünglich geplante **Zwei-Container-Architektur**
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

**Fix:** Pfad ohne `#` übergeben. Bereits in `VASTAI_SIM_ANLEITUNG.md` korrigiert.

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
Warnung ignoriert. `NO_FLASH_ATTN`-Zeile + Troubleshooting-Tipp aus `VASTAI_SIM_ANLEITUNG.md`
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

**Noch offen für volle Dataset-Treue:**
| Prio | Lücke |
|---|---|
| 1 | **Dex3-Finger-Gelenklimits zu eng** (KEIN Vorzeichen-Flip!). Über alle 281.196 Frames geprüft: Dataset-Vorzeichen **stimmen** mit der URDF überein (links Beugen negativ, rechts positiv) → Greif-*Richtung* korrekt. Aber die echte Range überschreitet die URDF-Limits um ~0,2–0,33 rad (z. B. left middle_1 bis −2,08 vs URDF −1,75; right index_1 bis +2,09 vs +1,75) → volles Schließen/Öffnen wird leicht geklemmt. Optional: USD/Sim-Gelenklimits an Dataset-Range weiten. Init-Pose-Clamp (index_0/middle_0 → 0.0) ist korrekt. |
| 2 | ✅ **erledigt** — **Wrist-Kameras** rekonstruiert + per Render verifiziert. Look-at im Link-Frame (`convention="world"`, `eye=(-0.08,0,0.13)`). Behob „nur Grau" (Kamera steckte im Palm-Mesh) + falsch-herum rechte Cam. Beide zeigen jetzt Hand + Würfel auf dem Tisch. Wegen asymmetrischer Arm-Startpose getrennte Targets: links `(0.14,0,-0.18)` (55° runter), rechts `(0.16,0,-0.05)` (37°). |

---

## 11. Aktueller Stand (2026-06-01)

Pipeline läuft **vollständig end-to-end** (20 Episoden, `results.json` + Videos). High-Kameras
zeigen jetzt den Arbeitsbereich. Gemessen: **0/20 Erfolge** mit `checkpoint-3000` — erwartet
(zu früh trainiert + Rest-OOD durch offene Sim-Treue-Punkte).

| Komponente | Status |
|---|---|
| Pipeline (download → server → sim → eval) | ✅ end-to-end verifiziert |
| High-Kameras | ✅ Look-at auf Tisch (Render verifiziert) |
| Roboter-Startpose (Dataset Frame 0) + weißer Tisch | ✅ umgesetzt |
| Wrist-Kameras + Dex3-Finger-Vorzeichenkonvention | ⏳ offen (Sim-Treue, s. Abschnitt 10) |
| `Dockerfile.vastai` | python3.10 + deepspeed-uninstall + pyzmq + 3 Smoke-Tests + openssh |
| Repo-Fixes noch nicht im gepushten Image | `client.py`, `run_g1_dex3_sim_eval.py`, `g1_dex3_cfg.py`, `entrypoint_sim.sh` → `update_sim_image.ps1 -VastAI` |
| Modell | `checkpoint-3000` (3000 Steps); für echte Erfolge 30k+ nötig |
| KISSKI Sim-Eval | Blockiert (RTX 5000 Turing < Ampere) |

**Iteration auf warmer Instanz** (ohne Rebuild): geänderte Sim-Dateien per `scp` nach
`/workspace/g1_dex3_sim/`, dann `NUM_EPISODES=2 PYTHONUNBUFFERED=1 bash /scripts/entrypoint_sim.sh`.
Für reproduzierbaren Stand: Image neu bauen + pushen.

---

## 12. Outdated-Hinweise zu anderen Dokumenten

| Dokument | Problem |
|---|---|
| `SIM_GPU_COMPATIBILITY.md` | RTX 5000 als "ja" (Isaac-Sim-Rendering) gelistet — ist faktisch **nein** für Isaac Lab 2.3.2 (Turing < Ampere-Mindestanforderung) |
| `SIM_DOCKER_BUILD.md` | Beschreibt Zwei-Container-Plan; `Dockerfile.vastai` (kombiniert) jetzt primäre Impl. für vast.ai; KISSKI-Zwei-Container bleibt gültig |
| `KISSKI_SIM_DESKTOP_ANLEITUNG.md` | Setzt RTX-5000-Kompatibilität voraus — vor Nutzung prüfen ob ältere Isaac-Lab-Version kompatibel ist |
