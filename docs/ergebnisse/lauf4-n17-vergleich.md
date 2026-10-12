# Auswertung — vierter Trainingsdurchlauf, GR00T **N1.7** gegen N1.6 (Lauf 3)

> **TL;DR:** _offen — wird nach Training und Checkpoint-Sweep ergänzt._ Erster vollständiger
> N1.7-Lauf auf KISSKI mit derselben Konfiguration wie Lauf 3 (Vision-Encoder, Color-Jitter,
> 80/20-Split). Frage: Ist N1.7 auf den zurückgehaltenen Episoden besser als N1.6 (Lauf 3,
> checkpoint-30000)?

**Erstellt:** 2026-10-12 (Gerüst) · **Job:** 16634862 (KISSKI; ersetzt den nicht gestarteten 30k-Job 16633316) · **Run-ID:** _offen_ ·
**Modell:** GR00T N1.7 (3,14 Mrd. Parameter, Backbone Cosmos-Reason2-2B / Qwen3-VL),
`tune_visual = true`, Color-Jitter, **80/20-Split**, Task „stack the blocks".

> Bezug: [Migrationsplan Phase 4](../weiterfuehrend/groot-n17-migration.md#phase-4--kisski),
> [KISSKI-Smoke-Test](../weiterfuehrend/groot-n17-migration.md#kisski-smoke-test-2026-10-12),
> Vergleichslauf [Lauf 3](lauf3-vision-split-auswertung.md).

---

## 1. Kurzfazit

_offen_

## 2. Lauf-Eckdaten und Abweichungen von Lauf 3

| Feld | Lauf 3 (N1.6) | Lauf 4 (N1.7) |
|---|---|---|
| Run-ID / Experiment | `tp1nc699` / `g1_dex3_blockstacking_vision_v2` | _offen_ |
| Hardware | 4 × A100, DeepSpeed ZeRO-2, bf16 | 4 × A100, DeepSpeed ZeRO-2, bf16 |
| Steps / Batch | 44.000 × global 32 | **44.000** × global 32 (wie Lauf 3; ursprünglich 30.000 nach Lauf 3 §8.2, nach der Nachmessung §5 verlängert) |
| Durchsatz | 0,98 Steps/s | ~1,4 Steps/s (Smoke-Test) · _Lauf: offen_ |
| Laufzeit | 12 h 27 min | _offen_ |
| LR / Schedule | 1e-4, cosine, `warmup_ratio = 0.1` | 1e-4, cosine, **`warmup_ratio = 0.05`** (heutiger Default von `run_finetuning_vision.sh`) |
| State-Dropout | 0,0 | **0,2** (N1.7-Modell-Default, `STATE_DROPOUT_PROB` leer) |
| Augmentierung | Color-Jitter (hue 0,08 · contrast 0,4 · brightness 0,3 · saturation 0,5) | _prüfen: gleiche Werte?_ |
| Split | train `0:240`, test `240:301` | _prüfen: `split.json`_ |
| `action_horizon` | 16 | 16 (`g1_dex3_config.py`, unverändert) |
| Normierung | Mean/Std bzw. Min/Max (N1.6) | _prüfen: `use_percentiles` / `override_pretraining_statistics`_ |
| Checkpoints | alle 5.000 Steps + final | alle 5.000 Steps |

> **Konfundierung (bewusst in Kauf genommen):** Warmup und State-Dropout weichen von Lauf 3 ab.
> Entscheidung 2026-10-12: Lauf 4 läuft mit den **aktuell empfohlenen** Werten
> (`warmup_ratio = 0.05`, N1.7-Default `state_dropout_prob = 0.2`) statt mit dem Rezept von
> Lauf 3. Verglichen wird damit „empfohlenes N1.7-Rezept gegen Lauf 3", nicht der Modellwechsel
> allein. Ein Unterschied in der Validierung ist entsprechend vorsichtig zuzuordnen.

## 3. Trainingsdynamik

_offen — W&B-Kurven (`train/loss`, `grad_norm`, `learning_rate`) nach `wandb sync`._

| Step | 0 | 5k | 10k | 15k | 20k | 25k | 30k |
|---|---|---|---|---|---|---|---|
| `train/loss` Lauf 4 | | | | | | | |

`train/loss` ist zwischen N1.6 und N1.7 **nicht direkt vergleichbar** (anderer Action-Head,
16 statt 32 DiT-Layer, State-Dropout). Vergleichbar ist nur die Sweep-MSE (§4).

## 4. Checkpoint-Sweep auf dem Test-Split

Gleiches Protokoll wie Lauf 3 §4 (6 Episoden, je 300 Steps, Action-Horizon 16), damit die
Zahlen direkt neben Lauf 3 stehen können:

```bash
cd ~
GROOT_VERSION=1.7 RUN_DIR=/data/g1_dex3_finetune/blockstacking_vision_n17 \
sbatch --export=ALL /mnt/vast-kisski/projects/kisski-humrob/repo/Training/kisski_open_loop_eval.sh
```

| Step | Lauf 3 mean MSE | **Lauf 4 mean MSE** | Lauf 3 MAE | **Lauf 4 MAE** |
|---|---|---|---|---|
| 5.000 | 0,01335 | | 0,0694 | |
| 10.000 | 0,00915 | | 0,0572 | |
| 15.000 | 0,00911 | | 0,0518 | |
| 20.000 | 0,00780 | | 0,0478 | |
| 25.000 | 0,00752 | | 0,0447 | |
| 30.000 | **0,00716** ★ | | 0,0418 | |
| 35.000 | 0,00763 | | 0,0417 | |
| 40.000 | 0,00855 | | 0,0419 | |
| 44.000 | 0,00898 | | 0,0418 | |

Lauf-3-Werte aus [lauf3-vision-split-auswertung.md](lauf3-vision-split-auswertung.md) §4.

> **Vorbehalt aus Lauf 3 §5:** 6 Episoden à 300 Steps bewerten nur die Anfahrphase und trennen
> Checkpoints im Fenster 20.000–35.000 nicht signifikant. Belastbarer ist das Protokoll aus
> Lauf 3 §8.1 (`EVAL_NUM_TRAJ=20 EVAL_STEPS=750`), das für Lauf 3 **noch nie gefahren** wurde
> (der Sweep-Job 16182706 vom 2026-09-23 galt Lauf 1). Für einen fairen Vergleich müsste es
> für **beide** Läufe laufen (§5).

## 5. Nachgeschärfter Vergleich (20 Episoden, volle Länge)

Lauf 3 nachgemessen am 2026-10-12 mit Job 16633477 (`EVAL_NUM_TRAJ=20 EVAL_STEPS=750`,
Test-Episoden 240–300 jede dritte, 45 min auf 1 × A100) →
`…/g1_dex3_blockstacking_vision_v2/checkpoint_sweep_20x750.json`. Das ursprüngliche
6×300-Ergebnis liegt unverändert daneben (zusätzlich gesichert als
`checkpoint_sweep_6x300_20260814.json`). Lauf 4 wird mit denselben Parametern gemessen.

| Step | Lauf 3 mean MSE | Lauf 3 MAE | Lauf 3 vs 44.000 (gepaart, n = 20) | Lauf 4 mean MSE | Lauf 4 MAE |
|---|---|---|---|---|---|
| 5.000 | 0,01016 | 0,0614 | Δ +0,00533, t = 22,4 | | |
| 10.000 | 0,00830 | 0,0521 | Δ +0,00348, t = 13,7 | | |
| 15.000 | 0,00734 | 0,0465 | Δ +0,00252, t = 9,9 | | |
| 20.000 | 0,00628 | 0,0412 | Δ +0,00145, t = 6,7 | | |
| 25.000 | 0,00564 | 0,0375 | Δ +0,00082, t = 3,5 | | |
| 30.000 | 0,00519 | 0,0340 | Δ +0,00037, t = 2,15 (p ≈ 0,045; 15/20 Episoden schlechter) | | |
| 44.000 | **0,00482** ★ | **0,0313** | — | — | — |

(35.000 und 40.000 nicht gemessen.)

> **Das korrigiert Lauf 3 §4.1 und §8.2.** Über die volle Episodenlänge und 20 Episoden gibt es
> **kein Overfitting**: Die Validierungs-MSE fällt bis 44.000 monoton, und 44.000 ist
> signifikant besser als 30.000. Die U-Kurve aus dem 6×300-Sweep war ein Effekt der kurzen
> Auswertung (nur Anfahrphase) und weniger Episoden (ep300 trug dort den Anstieg). Die
> Empfehlung „`MAX_STEPS=30000` genügt“ ist damit **nicht** gedeckt.
>
> **Folge für Lauf 4:** Der Lauf wurde deshalb vor dem Start von 30.000 auf **44.000 Steps**
> verlängert (Job 16634862), damit beide Kurven bis zum besten Lauf-3-Checkpoint vergleichbar sind.

## 6. Einordnung

_offen — Was zeigt der Vergleich, was nicht (Open-Loop ≠ Closed-Loop; Sim-Eval mit dem
besten N1.7-Checkpoint steht aus, Migrationsplan Phase 5)._

## 7. Verwendete Artefakte

- Training-Log: `~/logs/slurm-16634862.out` (KISSKI)
- Checkpoints: `/data/g1_dex3_finetune/blockstacking_vision_n17/<experiment>/checkpoint-*`
- Sweep: _offen_
