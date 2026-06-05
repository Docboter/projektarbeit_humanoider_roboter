# Troubleshooting

Gebündelte Fehlerlösungen für Training und Simulation. Jeder Eintrag: **Symptom → Ursache →
Lösung**, mit Verweis auf die jeweilige Detail-Doku.

---

## Training

### `CUDA out of memory` (OOM)

**Symptom:** Training bricht mit `torch.cuda.OutOfMemoryError` ab, oft im ersten Forward/Backward.

**Ursache:** `GLOBAL_BATCH_SIZE` zu groß für die verfügbare VRAM.

**Lösung:** `GLOBAL_BATCH_SIZE` zuerst halbieren. Richtwerte siehe
[training/env-vars.md → VRAM-Richtwerte](training/env-vars.md). Full Fine-tuning braucht laut
NVIDIA ≥ 40 GB; unter 24 GB ist OOM praktisch garantiert.

### W&B-Pflicht / Training bricht wegen fehlendem Key ab

**Symptom:** `USE_WANDB=1, aber WANDB_API_KEY ist nicht gesetzt.` — Abbruch vor dem Training.

**Ursache:** Der Entrypoint aktiviert W&B automatisch, sobald ein `WANDB_API_KEY` erwartet wird;
ohne gültigen Key bricht `run_finetuning.sh` ab.

**Lösung:** Entweder `WANDB_API_KEY=...` setzen **oder** explizit `USE_WANDB=0` übergeben, um
ohne W&B zu trainieren. Siehe [training/env-vars.md](training/env-vars.md#L19).

### W&B-Charts bleiben leer / Daten fehlen online

**Symptom:** Auf wandb.ai erscheinen keine Metriken, obwohl Training läuft.

**Ursache:** Der Entrypoint setzt `WANDB_MODE=offline` — Metriken werden lokal gepuffert, nicht
live hochgeladen (relevant v. a. auf KISSKI ohne Internet auf den Compute-Nodes).

**Lösung:** Nach dem Lauf manuell synchronisieren — Schritt für Schritt in
[training/wandb-offline-sync.md](training/wandb-offline-sync.md).

### Training startet nach `docker start` wieder bei Step 0

**Symptom:** Nach Container-Neustart beginnt das Training erneut bei Schritt 0.

**Ursache:** `docker start` / `--resume` ist **Container-Resume** (Daten bleiben), **nicht**
Checkpoint-Resume. Letzteres (`--resume_from_checkpoint`) ist im Entrypoint nicht exponiert.

**Lösung:** Erklärung + manueller Weg in
[training/anleitung.md → Fortsetzen nach Abbruch](training/anleitung.md).

---

## KISSKI / HPC

### Job hängt lange in der Queue (`PENDING`)

**Symptom:** `squeue -u $USER` zeigt den Job dauerhaft als `PENDING`.

**Ursache:** Die GPU-Partitionen (`kisski`, `kisski-h100`) sind geteilt; Wartezeit ist normal,
besonders für 4-GPU-Jobs. Max. Walltime 48 h.

**Lösung:** Geduld bzw. kleinere Ressourcenanforderung (weniger GPUs). Monitoring und Details:
[training/kisski-hpc.md](training/kisski-hpc.md).

### Daten / Checkpoints nach Job-Ende verschwunden

**Symptom:** Nach Job-Ende sind `/data`-Inhalte weg.

**Ursache:** Falscher Pfad. KISSKI bind-mountet **`/mnt/vast-kisski/projects/kisski-humrob/data`**
nach `/data`. Der alte SCRATCH-SCC-Speicher (`/scratch/`) wurde **am 2026-03-31 abgeschaltet**.

**Lösung:** Checkpoints vom VAST-Projekt-Storage holen — siehe
[training/kisski-hpc.md → Checkpoint-Export](training/kisski-hpc.md).

---

## Simulation (Isaac Lab / Isaac Sim)

### Kamera-Rendering scheitert: `createDLSSContext error`

**Symptom:** `[Error] [rtx.postprocessing.plugin] createDLSSContext error` beim Start mit
`--enable_cameras` (auch headless).

**Ursache:** Die GPU erfüllt nicht **beide** Isaac-Sim-Anforderungen — **RT-Cores** *und*
**Ampere+ (Compute Capability ≥ 8.0)**:
- **A100 / H100:** Ampere/Hopper, aber **keine RT-Cores** → Kamera-Rendering unmöglich.
- **Quadro RTX 5000:** hat RT-Cores, ist aber **Turing (CC 7.5)** → eine Generation zu alt.

**Lösung:** Ampere-or-newer-GPU **mit** RT-Cores nutzen — auf vast.ai **L40 / RTX 4090 / A6000**.
Hintergrund: [simulation/umsetzungsnotizen.md §1](simulation/umsetzungsnotizen.md),
[simulation/archiv/gpu-kompatibilitaet.md](simulation/archiv/gpu-kompatibilitaet.md).

### Closed-Loop: Roboter bewegt sich kaum, greift nicht

**Symptom:** Im Closed-Loop liegen die Hände auf dem Tisch, der Roboter macht nur kleine,
ungerichtete Bewegungen — obwohl das Open-Loop-Replay sauber greift.

**Ursache:** **Visueller Domain Gap.** Der eingefrorene Vision-Encoder produziert für synthetische
Isaac-Sim-Bilder unbrauchbare Features (gemessen: mittlere Cosine-Distanz 0.260, `cam_left_wrist`
kritisch bei 0.427). Training und Sim-Config sind korrekt — der Unterschied liegt allein in den
Kamerabildern.

**Lösung:** Domain Gap angehen statt Sim-Pipeline weiter zu tunen — Vision-Encoder fine-tunen
(`tune_visual=true`) oder Domain Randomization in der Sim. Diagnose-Werkzeuge:
[simulation/umsetzungsnotizen.md §15](simulation/umsetzungsnotizen.md). Vollanalyse:
[umgebungsanalyse.md](umgebungsanalyse.md) und
[ergebnisse/lauf1-auswertung.md §8](ergebnisse/lauf1-auswertung.md).

### `flash-attn` / `--no-flash-attn`-Fehler auf älterer GPU

**Symptom:** `AssertionError` aus Flash-Attention, bzw. `--no-flash-attn` wird als unbekanntes
Flag abgewiesen.

**Ursache:** Flash Attention setzt **Ampere+** voraus (Volta/Turing brechen ab). Das Flag
`--no-flash-attn` ist **nicht implementiert**.

**Lösung:** Ampere-or-newer-GPU verwenden (dieselbe Anforderung wie für das Rendering).
Details: [simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md).

---

> Fehlt hier ein Fall? Tiefergehende, fortlaufend gepflegte Lessons Learned stehen in
> [simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md) (Sim) und der
> [Abschluss-Auswertung des 1. Laufs](ergebnisse/lauf1-auswertung.md) (Training).
