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

**Ursache:** Der Entrypoint setzt `WANDB_MODE=offline` als **Default** — Metriken werden lokal
gepuffert, nicht live hochgeladen (relevant v. a. auf KISSKI ohne Internet auf den Compute-Nodes).

**Lösung:** Nach dem Lauf manuell synchronisieren — Schritt für Schritt in
[training/wandb-offline-sync.md](training/wandb-offline-sync.md). Wo Internet verfügbar ist
(vast.ai, eigener Server), lässt sich der Default per `WANDB_MODE=online` überschreiben
([entrypoint.sh:175](../Training/scripts/entrypoint.sh#L175)).

### Training startet nach `docker start` wieder bei Step 0

**Symptom:** Nach Container-Neustart beginnt das Training erneut bei Schritt 0 — obwohl
Checkpoints vorhanden sein sollten.

**Ursache:** Der Trainer setzt normalerweise **automatisch** am letzten Checkpoint fort
(`trainer.train(resume_from_checkpoint=True)` in
[`experiment.py:288`](../app/Groot-1.6/gr00t/experiment/experiment.py), aufgelöst über
`get_last_checkpoint(output_dir)`). Startet er trotzdem bei 0, hat er im `output_dir` **keinen**
`checkpoint-*`-Ordner gefunden. Typische Gründe: ein geändertes `EXPERIMENT_NAME`/`OUTPUT_DIR`,
oder der erste Checkpoint war zum Abbruchzeitpunkt noch nicht geschrieben (`SAVE_STEPS=2000`).

**Lösung:** Prüfen, ob `/data/g1_dex3_finetune/blockstacking/g1_dex3_blockstacking_v1/checkpoint-*`
existiert, und ob `EXPERIMENT_NAME` unverändert ist. Details:
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

**Ursache:** Ursprünglich allein dem **visuellen Domain Gap** zugeschrieben: Der eingefrorene
Vision-Encoder produziert für synthetische Isaac-Sim-Bilder schlechtere Features. Aktueller
Messstand nach Kamera-Neukalibrierung (2026-08-08): mittlere Cosine-Distanz **0,2229**,
`cam_left_wrist` **0,3556** — die Juni-Erstmessung (0.260 / 0.427) ist überholt.

> **Stand 2026-08-14 (Läufe 29–34):** Die Greif-Diagnose der Läufe 29–31 zeigte: die Politik
> kommandiert nur ~19 % der demonstrierten Fingerspannweite; eine fehlerhafte De-Normalisierung
> wurde ausgeschlossen. **Lauf 32** (`span`-Gate) entschied die Ursache: auf **echten** Bildern
> erreicht dieselbe Politik 100 % der Demo-Spanne → der visuelle Domain Gap ist bestätigt.
> **Lauf 34** maß den daraufhin trainierten Vision-Checkpoint (`TUNE_VISUAL=1` + Jitter + Split)
> im Closed Loop: Fingerspanne **27,6 % statt 20,5 %** (p = 3,3 · 10⁻⁴) — die Maßnahme wirkt,
> `lifted` bleibt aber 0/10. Details: [ergebnisse/diagnose-chronik.md](ergebnisse/diagnose-chronik.md).

**Lösung:** Der wirksame Hebel ist der Vision-Encoder — mit Realdaten allein aber ausgereizt.
Nächster Schritt ist **Co-Training auf echten + gerenderten Bildern**
([training/co-training.md](training/co-training.md), `USE_COTRAIN=1`).
Diagnose-Werkzeuge: [simulation/umsetzungsnotizen.md §14–15](simulation/umsetzungsnotizen.md).
Aktuelle Messung: [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md).
Vollanalyse: [ergebnisse/lauf1-auswertung.md](ergebnisse/lauf1-auswertung.md) und
[ergebnisse/lauf3-vision-split-auswertung.md](ergebnisse/lauf3-vision-split-auswertung.md).

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
