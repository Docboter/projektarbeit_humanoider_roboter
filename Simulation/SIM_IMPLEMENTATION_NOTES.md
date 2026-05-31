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

Am lokal gebauten Image (`docker run --entrypoint bash …`) **definitiv verifiziert** — zwei
unabhängige Ursachen, beide am gcc-Befehl ablesbar (`-I/usr/include/python3.12 … -lcuda`):

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

> **Nicht GPU-frei verifizierbar:** Der finale Triton-Compile braucht das zur Laufzeit
> injizierte `libcuda.so.1` (kein GPU beim Build). Die beiden konkreten Blocker sind aber
> ausgeräumt; verbleibendes Restrisiko liegt erst in der Isaac-Lab-Sim-Schleife selbst.

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

## 7. Fix: SSH-Server im Entrypoint

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

## 8. Aktueller Stand (2026-06-01)

| Komponente | Status |
|---|---|
| `Dockerfile.vastai` | Python-3.10-Fix + Build-Smoke-Test + openssh-server ergänzt — **muss neu gebaut + gepusht werden** |
| `entrypoint_sim.sh` | `unset VIRTUAL_ENV` + SSH (sshd + PUBLIC_KEY) + libcuda.so-Symlink + Flash-Attn-Flag entfernt |
| USD-Asset (`g1_dex3.usd` + `configuration/`) | Erzeugt, lokal unter `data/`, auf HF hochgeladen |
| Checkpoint `checkpoint-3000` | Auf HF (`luca-mue/groot-g1dex3-checkpoint`) |
| vast.ai Eval-Lauf | Mehrere Instanzen am Triton-gcc-Fehler gescheitert (Python 3.12/3.10-Mismatch). Fix liegt vor, Image noch nicht neu gebaut. |
| KISSKI Sim-Eval | Blockiert durch RTX-5000-Inkompatibilität (Turing + Isaac Sim 4.x) |

**Nächster Schritt:** `.\Simulation\update_sim_image.ps1 -VastAI` (baut + pusht). Der Build
bricht jetzt lokal ab, falls die venv-Python-Header nicht passen — d. h. der Triton-Fehler
kann nicht mehr unbemerkt erst auf vast.ai auftreten. Danach neue Instanz mit `-p 22` starten.

---

## 9. Outdated-Hinweise zu anderen Dokumenten

| Dokument | Problem |
|---|---|
| `SIM_GPU_COMPATIBILITY.md` | RTX 5000 als "ja" (Isaac-Sim-Rendering) gelistet — ist faktisch **nein** für Isaac Lab 2.3.2 (Turing < Ampere-Mindestanforderung) |
| `SIM_DOCKER_BUILD.md` | Beschreibt Zwei-Container-Plan; `Dockerfile.vastai` (kombiniert) jetzt primäre Impl. für vast.ai; KISSKI-Zwei-Container bleibt gültig |
| `KISSKI_SIM_DESKTOP_ANLEITUNG.md` | Setzt RTX-5000-Kompatibilität voraus — vor Nutzung prüfen ob ältere Isaac-Lab-Version kompatibel ist |
