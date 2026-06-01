# GPU-Eignung für die Isaac-Sim-Closed-Loop-Sim auf GWDG

> **TL;DR:** Die Closed-Loop-Sim aus [isaac-lab-plan.md](isaac-lab-plan.md)
> **läuft NICHT auf den KISSKI-Partitionen** (A100/H100), weil Isaac Sim zum Rendern
> der Kameras **RT-Cores** braucht und A100/H100 keine haben. Der einzige Weg auf GWDG
> führt über die **`jupyter`-Partition** (`jupyter.hpc.gwdg.de`) mit ihrer
> **Quadro RTX 5000** — die hat RT-Cores. Mit Vorbehalten (16 GB VRAM, Turing-Gen,
> geteilte Interaktiv-Partition).

---

## 1. Das Problem: Isaac Sim braucht RT-Cores

Isaac Sim rendert über die **RTX-Pipeline (Hardware-Raytracing)** und benötigt dafür
zwingend **RT-Cores** (und für Streaming NVENC). NVIDIA listet **A100 und H100
ausdrücklich als nicht unterstützt** — sie sind reine Compute-GPUs ohne RT-Cores.

**Das gilt auch headless.** „Headless" heißt nur „kein Fenster"; die Kamera-Sensoren
laufen trotzdem über dieselbe RTX-Render-Pipeline. Konkret dokumentiert:

- `./isaaclab.sh -p scripts/demos/sensors/cameras.py --headless --enable_cameras`
  scheitert auf A100 mit
  `[Error] [rtx.postprocessing.plugin] createDLSSContext error: unable to initialize context`
- Warnung: `DLSS-RR is not supported … NVIDIA A100 PCIe … RTX Real-Time raytracing
  mode will not be able to denoise the output correctly`

**Folge für unser Projekt:** Die GR00T-Policy *braucht* die 4 RGB-Kamerabilder als
Observation. Ohne funktionierendes Kamera-Rendering ist die Closed-Loop-Sim unmöglich
— egal wie gut der Container ist.

| Auf A100 / H100 | Status |
|---|---|
| Physik-Sim (Gelenke, Kollision, Control-Loop) | läuft |
| **Kamera-Rendering (4× RGB)** | **scheitert / fehlerhaft** |
| GR00T-Policy in der Schleife | **unmöglich** |

---

## 2. GPU-Landschaft auf GWDG

| Partition | GPU | VRAM | RT-Cores? | Isaac-Sim-Rendering |
|---|---|---|---|---|
| `kisski` | A100 | 80 GB | ❌ | nein |
| `kisski-h100` | H100 | 94 GB | ❌ | nein |
| `grete`, `grete:shared` | A100 | 40/80 GB | ❌ | nein |
| `grete-h100` | H100 | 94 GB | ❌ | nein |
| `scc-gpu` | A100 / H100 | — | ❌ | nein |
| `jupyter` (Phase 1) | V100 | 32 GB | ❌ | nein |
| **`jupyter`** | **Quadro RTX 5000** | **16 GB** | ✅ | **ja** |

Die **`jupyter`-Partition ist die einzige RTX-Klasse-Hardware im gesamten
GWDG-Cluster** (laut GWDG-Doku). Erreichbar über `jupyter.hpc.gwdg.de`
(JupyterHPC-Service): Bei „GPU" im HPC-Device-Tray bekommt man einen Knoten mit
**Quadro RTX 5000**. Anforderung im Batch alternativ über `-G RTX5000`.

---

## 3. Die Lösung: `jupyter.hpc.gwdg.de` (RTX 5000)

Die Quadro RTX 5000 ist eine professionelle Visualisierungskarte **mit RT-Cores und
NVENC** → Isaac-Sim-Rendering funktioniert prinzipiell. Aber mit Vorbehalten:

### Caveats

1. **Nur 16 GB VRAM.** Der eigentliche Engpass. Eine **einzelne** Block-Stacking-Env
   mit 4 Kameras passt wahrscheinlich rein, aber knapp. Keine massiv-parallelen Envs.
2. **Turing-Generation (älter)**, liegt **unter** NVIDIAs Mindestempfehlung für
   aktuelle Isaac-Sim-Versionen (RTX 3070+). RT-Cores sind da → rendert, aber mit
   bescheidener Performance. **Ggf. eine ältere Isaac-Sim/Lab-Version** verwenden, die
   Turing offiziell unterstützt (statt `nvcr.io/nvidia/isaac-lab:2.3.2` z. B. eine
   frühe 2.x oder 1.x).
3. **`jupyter` ist eine geteilte, oversubscribte Interaktiv-Partition** — für Notebooks
   gedacht, nicht für lange Batch-/Dauerläufe. Für die Sim-Eval als *qualitatives
   Werkzeug* (Plan, Abschnitt 0) ok, aber mit möglichen Scheduling-Wartezeiten.

---

## 4. Konsequenz für die Zwei-Container-Architektur

Die 16 GB VRAM erzwingen eine Entscheidung über die beiden Container
(GR00T-Server + Isaac-Sim-Client, vgl. [sim-docker-build.md](sim-docker-build.md)):

- **Variante A — alles auf dem RTX-5000-Knoten:** GR00T-Server **und** Isaac-Sim-Client
  auf derselben Karte. GR00T-N1.6-Gewichte + Isaac Sim + 4 Kamera-Buffer müssen
  zusammen in 16 GB passen → **vermutlich zu eng**. Zuerst messen.
- **Variante B — Server auf A100 (`kisski`), Client auf RTX 5000 (`jupyter`):**
  sauberer für VRAM, aber zwei **getrennte SLURM-Jobs auf verschiedenen Partitionen**
  → ZMQ läuft dann **nicht über `localhost`**, sondern über Node-Hostname/Netz.
  Bricht die Plan-Annahme „beide im selben Job auf demselben Node" → etwas
  Netzwerk-Aufwand, aber machbar.

---

## 5. Empfohlenes Vorgehen

1. **Erst klein testen, bevor der große Container gebaut wird.** In einer
   JupyterHPC-GPU-Session die Minimal-Render-Probe (Plan-Phase A) fahren:
   ```bash
   ./isaaclab.sh -p scripts/demos/sensors/cameras.py --headless --enable_cameras
   ```
   Liefert das auf der RTX 5000 ein RGB-Bild (kein `createDLSSContext`-Fehler), ist der
   Weg frei.
2. **VRAM-Budget prüfen** (Env + ggf. GR00T-Server) → entscheidet A vs. B.
3. **Ältere Isaac-Sim/Lab-Version** in Betracht ziehen (Turing-Kompatibilität).
4. **Parallel auf KISSKI die Open-Loop-Eval** (Plan, Abschnitt 10) fahren — kein
   Rendering nötig, läuft problemlos auf A100 und ist ohnehin die Voraussetzung, bevor
   sich der ganze Sim-Aufwand lohnt.

---

## Quellen

- [Isaac Sim Requirements (5.1.0)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [IsaacLab Issue #3421 — Camera-headless auf A100 schlägt fehl](https://github.com/isaac-sim/IsaacLab/issues/3421)
- [IsaacLab Issue #1519 — schlechte Renderqualität auf A100 (DLSS-RR)](https://github.com/isaac-sim/IsaacLab/issues/1519)
- [NVIDIA-Forum — Isaac Sim A100](https://forums.developer.nvidia.com/t/isaac-sim-a100/291492)
- [GWDG HPC — GPU Partitions](https://docs.hpc.gwdg.de/how_to_use/compute_partitions/gpu_partitions/index.html)
- [GWDG HPC — JupyterHPC](https://docs.hpc.gwdg.de/services/jupyterhpc/index.html)
- [GWDG Systems-Übersicht (Grete/SCC)](https://gwdg.de/en/hpc/systems/)
