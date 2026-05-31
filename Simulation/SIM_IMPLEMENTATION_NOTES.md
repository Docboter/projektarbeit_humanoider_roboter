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

## 5. Fix: Triton `gcc -lcuda` schlägt fehl — `libcuda.so`-Stub fehlt

**Problem:** Beim Start des GR00T-Servers importiert `transformers` → Triton → versucht
`cuda_utils.cpython-*.so` mit `gcc -lcuda` zu kompilieren. Im NVIDIA-Docker-Container liegt
nur `libcuda.so.1` (vom nvidia-container-toolkit injiziert), nicht der Compile-Stub `libcuda.so`.
Der gcc-Befehl schlägt mit Exit-Status 1 fehl:
```
RuntimeError: Failed to import transformers.modeling_utils
```

**Fix:** `LIBRARY_PATH` (gcc Compile-Zeit, nicht `LD_LIBRARY_PATH` welches nur Runtime betrifft)
als `ENV` im Dockerfile setzen, damit der gcc-Subprozess von Triton `libcuda.so` findet:
```dockerfile
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
```
Zusätzlich als Fallback im `entrypoint_sim.sh` exportiert. Bereits in beiden Dateien eingebaut.

---

## 6. Fix: SSH-Server im Entrypoint

**Problem:** Docker-ENTRYPOINT-Modus auf vast.ai startet keinen SSH-Server → `vastai ssh-url`
schlägt fehl mit "ssh port not found".

**Fix:** `service ssh start` am Anfang von `entrypoint_sim.sh` ergänzt (nach `mkdir`-Block).
Der isaac-lab-Basis-Image hat OpenSSH vorinstalliert.

```bash
service ssh start 2>/dev/null || true
```

SSH-Verbindung danach via:
```bash
vastai set api-key <key>
vastai ssh <instance-id>
```

---

## 6. Aktueller Stand (2026-05-31)

| Komponente | Status |
|---|---|
| `Dockerfile.vastai` | Gebaut, lokal vorhanden; **noch nicht gepusht** |
| `entrypoint_sim.sh` | `unset VIRTUAL_ENV` + SSH-Start + Triton `LD_LIBRARY_PATH`-Fix eingebaut |
| USD-Asset (`g1_dex3.usd` + `configuration/`) | Erzeugt, lokal unter `data/`, auf HF hochgeladen |
| Checkpoint `checkpoint-3000` | Auf HF (`luca-mue/groot-g1dex3-checkpoint`) |
| vast.ai Eval-Lauf | Instanz 38833601 gescheitert (Triton-Fehler) — Image neu bauen + pushen nötig |
| KISSKI Sim-Eval | Blockiert durch RTX-5000-Inkompatibilität (Turing + Isaac Sim 4.x) |

**Nächster Schritt:** `.\Simulation\update_sim_image.ps1 -VastAI` (baut + pusht Image mit SSH-Fix),
dann bei nächster Instanz `vastai ssh <id>` für SSH-Zugang nutzen.

---

## 6. Outdated-Hinweise zu anderen Dokumenten

| Dokument | Problem |
|---|---|
| `SIM_GPU_COMPATIBILITY.md` | RTX 5000 als "ja" (Isaac-Sim-Rendering) gelistet — ist faktisch **nein** für Isaac Lab 2.3.2 (Turing < Ampere-Mindestanforderung) |
| `SIM_DOCKER_BUILD.md` | Beschreibt Zwei-Container-Plan; `Dockerfile.vastai` (kombiniert) jetzt primäre Impl. für vast.ai; KISSKI-Zwei-Container bleibt gültig |
| `KISSKI_SIM_DESKTOP_ANLEITUNG.md` | Setzt RTX-5000-Kompatibilität voraus — vor Nutzung prüfen ob ältere Isaac-Lab-Version kompatibel ist |
