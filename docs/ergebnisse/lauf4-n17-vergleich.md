# Auswertung — vierter Trainingsdurchlauf, GR00T **N1.7** gegen N1.6 (Lauf 3)

> **TL;DR:** Erster vollständiger N1.7-Lauf (KISSKI, 44k Steps, Vision-Encoder, Color-Jitter,
> 80/20-Split) gegen Lauf 3 (N1.6, gleiche Konfiguration). **Open-Loop auf den zurückgehaltenen
> Episoden ist N1.7 am Ende gleichauf mit N1.6, nicht besser:** bester N1.7-Checkpoint 0,00512
> (Step 35.000) gegen 0,00482 (N1.6, Step 44.000), Unterschied n.s. N1.7 lernt **früher**
> (signifikant besser bei 10k und 20k), erreicht sein Plateau schon bei ~30k und trainiert
> **1,6× schneller** (1,56 statt 0,98 Steps/s). Kurzfazit in §1.

**Erstellt:** 2026-10-12, Ergebnisse 2026-10-13 · **Job:** 16634862 (KISSKI; ersetzt den nicht
gestarteten 30k-Job 16633316) · **W&B (offline):** `offline-run-20261012_192301-2v02ye81` ·
**Modell:** GR00T N1.7 (3,14 Mrd. Parameter, Backbone Cosmos-Reason2-2B / Qwen3-VL),
`tune_visual = true`, Color-Jitter, **80/20-Split**, Task „stack the blocks".

> Bezug: [Migrationsplan Phase 4](../weiterfuehrend/groot-n17-migration.md#phase-4--kisski),
> [KISSKI-Smoke-Test](../weiterfuehrend/groot-n17-migration.md#kisski-smoke-test-2026-10-12),
> Vergleichslauf [Lauf 3](lauf3-vision-split-auswertung.md).

---

## 1. Kurzfazit

1. **Kein Gewinn am Ende.** Im belastbaren 20×750-Protokoll (§5) liegen beide Modelle ab Step
   25.000 gleichauf (|t| ≤ 0,26). Bester gegen besten Checkpoint: N1.7 @ 35.000 = 0,00512,
   N1.6 @ 44.000 = 0,00482, also N1.7 +6 %, gepaart t = 1,20 (n = 20) — **nicht signifikant**.
2. **N1.7 lernt früher.** Bei 10.000 (t = −2,95) und 20.000 (t = −2,48) ist N1.7 signifikant besser,
   bei 15.000 knapp nicht (t = −2,07; Schwelle 2,09). Nur bei 5.000 ist es schlechter (t = +2,41).
3. **Plateau ab ~30.000.** Innerhalb von Lauf 4 sind 30.000, 35.000, 40.000 und 44.000 nicht
   unterscheidbar (|t| ≤ 0,69). N1.6 verbessert sich dagegen bis 44.000 weiter.
4. **Kein Overfitting**, auch nicht bei N1.7. Im 6×300-Protokoll (§4) zeigen beide Läufe wieder
   die scheinbare U-Kurve mit Optimum bei 30.000 — dasselbe Messartefakt wie bei Lauf 3.
5. **Schneller.** 7 h 53 min statt 12 h 27 min für dieselben 44.000 Steps.

Für die Praxis: Jeder N1.7-Checkpoint ab 30.000 ist gleichwertig; 44.000 ist eine gute Wahl.
Im **Closed Loop** ändert N1.7 nichts: ckpt44000 erreicht 0/10 `lifted` bei 26 % Fingerspanne,
das Niveau von N1.6 aus Lauf 34 (§6).

## 2. Lauf-Eckdaten und Abweichungen von Lauf 3

| Feld | Lauf 3 (N1.6) | Lauf 4 (N1.7) |
|---|---|---|
| Run-ID / Experiment | `tp1nc699` / `g1_dex3_blockstacking_vision_v2` | `2v02ye81` (offline) / `g1_dex3_blockstacking_vision_v1` (Namespace `blockstacking_vision_n17`) |
| Hardware | 4 × A100, DeepSpeed ZeRO-2, bf16 | 4 × A100 (`ggpu186`), DeepSpeed ZeRO-2, bf16 |
| Steps / Batch | 44.000 × global 32 | 44.000 × global 32 (ursprünglich 30.000 nach Lauf 3 §8.2, nach der Nachmessung §5 verlängert) |
| Durchsatz | 0,98 Steps/s | **1,56 Steps/s** (49,8 Samples/s) |
| Laufzeit | 12 h 27 min | **7 h 53 min** (2026-10-12 19:22 → 2026-10-13 03:15) |
| LR / Schedule | 1e-4, cosine, `warmup_ratio = 0.1` | 1e-4, cosine, `warmup_ratio = 0.1` (Peak bei Step 4.410) |
| State-Dropout | 0,0 | **0,2** (N1.7-Modell-Default, `STATE_DROPOUT_PROB` leer) |
| Augmentierung | Color-Jitter (hue 0,08 · contrast 0,4 · brightness 0,3 · saturation 0,5) | identisch, keine Rotation |
| Split | train `0:240`, test `240:301` | identisch (`split.json`) |
| Action-Horizon | 16 | Modell-Maximum 40, genutzt 16 (`delta_indices` in `g1_dex3_config.py`) |
| Normierung | N1.6-Default | `use_percentiles = true`, `override_pretraining_statistics = true` (N1.7-Defaults) |
| Checkpoints | alle 5.000 Steps + final | alle 5.000 Steps + final (44.000) |

> **Konfundierung:** Nur der State-Dropout weicht vom Rezept von Lauf 3 ab (0,2 statt 0), dazu
> die modellbedingten N1.7-Defaults (Normierung). Entscheidung 2026-10-12: bewusst die
> empfohlenen N1.7-Werte. Ursprünglich war hier auch ein Warmup von 0,05 vermerkt; tatsächlich
> lief der Job mit 0,1 (Default von `kisski_submit.sh`, im SLURM-Log und an der Lernrate
> belegt) — also wie Lauf 3.

## 3. Trainingsdynamik — gesund ✅

Mittelwerte je 2.000-Step-Fenster aus `trainer_state.json` (4.400 Logpunkte):

| Step | 0 | 2k | 6k | 10k | 14k | 20k | 26k | 30k | 34k | 38k | 42k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `train/loss` | 0,416 | 0,100 | 0,075 | 0,047 | 0,040 | 0,030 | 0,023 | 0,020 | 0,015 | 0,015 | 0,012 |
| `grad_norm` | 1,85 | 0,90 | 0,49 | 0,36 | 0,32 | 0,26 | 0,24 | 0,21 | 0,19 | 0,16 | 0,14 |

Loss 1,36 (Step 10) → 0,0118 (Ende), Mittel 0,053; grad_norm final 0,065, Maximum 5,58 im
Warmup; keine NaN/Inf; LR läuft auf 1,6e-13 aus.

`train/loss` ist zwischen N1.6 und N1.7 **nicht direkt vergleichbar** (anderer Action-Head,
16 statt 32 DiT-Layer, State-Dropout). Vergleichbar ist nur die Sweep-MSE (§4, §5).

## 4. Checkpoint-Sweep, 6 Episoden × 300 Steps (Protokoll wie Lauf 3 §4)

Job 16643458 (15 min, 1 × A100), dieselben Test-Episoden wie Lauf 3 (240, 252, 264, 276, 288, 300).

| Step | Lauf 3 MSE | **Lauf 4 MSE** | Lauf 4 MAE | Δ Lauf 4 − Lauf 3 (gepaart, n = 6) |
|---|---|---|---|---|
| 5.000 | 0,01335 | 0,01393 | 0,0711 | +0,00058, t = +0,75 |
| 10.000 | 0,00915 | 0,00922 | 0,0532 | +0,00007, t = +0,25 |
| 15.000 | 0,00911 | 0,00873 | 0,0497 | −0,00038, t = −0,76 |
| 20.000 | 0,00780 | 0,00908 | 0,0481 | +0,00128, t = +1,42 |
| 25.000 | 0,00752 | 0,00807 | 0,0440 | +0,00055, t = +1,48 |
| 30.000 | **0,00716** ★ | **0,00690** ★ | 0,0402 | −0,00026, t = −0,37 |
| 35.000 | 0,00763 | 0,00819 | 0,0412 | +0,00057, t = +1,78 |
| 40.000 | 0,00855 | 0,00846 | 0,0414 | −0,00010, t = −0,16 |
| 44.000 | 0,00898 | 0,00816 | 0,0402 | −0,00082, t = −1,18 |

Kein Unterschied signifikant (Schwelle bei n = 6: |t| = 2,57). Beide zeigen die scheinbare
U-Kurve mit Optimum bei 30.000, die das 20×750-Protokoll (§5) für Lauf 3 als Artefakt
entlarvt hat — und für Lauf 4 ebenso: dort ist 30.000 gleichauf mit 44.000.
Lauf-3-Werte aus [lauf3-vision-split-auswertung.md](lauf3-vision-split-auswertung.md) §4.

## 5. Nachgeschärfter Vergleich, 20 Episoden × 750 Steps (volle Länge)

Lauf 3 nachgemessen am 2026-10-12 (Job 16633477, 45 min), Lauf 4 am 2026-10-13 (Job 16643464,
1 h 31 min), beide auf denselben 20 Test-Episoden (240–300, etwa jede dritte), Action-Horizon 16.
Das ursprüngliche 6×300-Ergebnis von Lauf 3 liegt unverändert neben der Nachmessung
(zusätzlich gesichert als `checkpoint_sweep_6x300_20260814.json`).

| Step | Lauf 3 MSE | Lauf 3 MAE | **Lauf 4 MSE** | **Lauf 4 MAE** | Δ Lauf 4 − Lauf 3 (gepaart, n = 20) |
|---|---|---|---|---|---|
| 5.000 | 0,01016 | 0,0614 | 0,01128 | 0,0629 | +0,00113, t = +2,41 (N1.6 besser) |
| 10.000 | 0,00830 | 0,0521 | 0,00763 | 0,0476 | −0,00068, t = −2,95 (**N1.7 besser**, 15/20) |
| 15.000 | 0,00734 | 0,0465 | 0,00687 | 0,0422 | −0,00047, t = −2,07 (Trend, 14/20) |
| 20.000 | 0,00628 | 0,0412 | 0,00575 | 0,0380 | −0,00053, t = −2,48 (**N1.7 besser**, 16/20) |
| 25.000 | 0,00564 | 0,0375 | 0,00569 | 0,0354 | +0,00005, t = +0,21 |
| 30.000 | 0,00519 | 0,0340 | 0,00515 | 0,0336 | −0,00004, t = −0,26 |
| 35.000 | — | — | **0,00512** ★ | 0,0323 | — |
| 40.000 | — | — | 0,00521 | 0,0320 | — |
| 44.000 | **0,00482** ★ | **0,0313** | 0,00513 | 0,0316 | +0,00031, t = +1,62 (8/20) |

Signifikanzschwelle bei n = 20: |t| = 2,09 (p < 0,05, zweiseitig). Bei Lauf 3 wurden 35.000 und
40.000 nicht gemessen.

**Innerhalb von Lauf 4** (gepaart gegen den besten, 35.000): 5.000 bis 25.000 signifikant
schlechter (t = 11,5 / 7,8 / 5,9 / 2,4 / 3,2); 30.000, 40.000 und 44.000 nicht unterscheidbar
(t = 0,19 / 0,69 / 0,05).

**Bester gegen besten:** Lauf 4 @ 35.000 − Lauf 3 @ 44.000 = +0,00030 (+6 %), t = 1,20,
N1.7 in 8/20 Episoden besser — kein belastbarer Unterschied.

> **Lauf 3 (Nachtrag 2026-10-12):** Diese Nachmessung korrigiert Lauf 3 §4.1 und §8.2. Über die
> volle Episodenlänge und 20 Episoden gibt es **kein Overfitting**: Die Validierungs-MSE fällt bis
> 44.000 monoton, und 44.000 ist signifikant besser als 30.000 (Lauf 3 gegen 44.000: t = 22,4 /
> 13,7 / 9,9 / 6,7 / 3,5 / 2,15 für 5k–30k). Die U-Kurve aus dem 6×300-Sweep war ein Effekt der
> kurzen Auswertung (nur Anfahrphase) und weniger Episoden. Deshalb wurde Lauf 4 vor dem Start
> von 30.000 auf 44.000 Steps verlängert.

## 6. Einordnung

- **Was der Vergleich zeigt:** N1.7 trifft die Aktionen der echten Demonstrationen am Ende nicht
  besser als N1.6, kommt aber schneller dorthin — in Steps und in Rechenzeit. Für weitere
  N1.7-Läufe in dieser Konfiguration genügen ~30.000 Steps (≈ 5,3 h auf 4 × A100); das ist jetzt
  im 20×750-Protokoll belegt, anders als die alte 30k-Empfehlung aus Lauf 3.
- **Was er nicht zeigt:** Open-Loop-MSE auf echten Bildern sagt nichts über den Sim-Closed-Loop.
  Dort liegt der eigentliche Engpass des Projekts. Die Sim-Läufe mit Checkpoints dieses Laufs
  ([diagnose-chronik.md, Läufe 54/55](diagnose-chronik.md#läufe-5455-die-ersten-n17-checkpoints-in-der-sim)):

  | Checkpoint | Closed Loop (10 × 40 s) | Open Loop `span`, Fingerspanne Train / Test | Open Loop MSE-Median Train / Test |
  |---|---|---|---|
  | 15.000 | 0/10 `lifted`, max. 0,83 cm | 83 % / 90 % | 0,0087 / 0,0064 |
  | 44.000 | 0/10 `lifted`, Fingerspanne 26 % der Demo; Hand fährt an den Würfel, schließt nicht | 77 % / 94 % | 0,0035 / 0,0043 |

  Das auf echten Bildern deutlich bessere Modell kommt im Closed Loop nicht an; die Sim steht auf
  dem Niveau von N1.6 mit `TUNE_VISUAL` (Lauf 34: 27,6 %, 0/10). Der Modellwechsel allein schließt
  den Domain-Gap also nicht.
- **Konfundierung:** State-Dropout 0,2 (N1.7) gegen 0 (N1.6) und die N1.7-Normierungs-Defaults —
  ein Teil des frühen Vorsprungs oder des späteren Gleichstands kann daran liegen.

## 7. Verwendete Artefakte

- Training-Log: `~/logs/slurm-16634862.out` (KISSKI); Container-Log
  `/data/logs/finetune-vision-20261012-192231.log`
- Checkpoints: `/data/g1_dex3_finetune/blockstacking_vision_n17/g1_dex3_blockstacking_vision_v1/checkpoint-*`
- Sweeps Lauf 4: `…/g1_dex3_blockstacking_vision_v1/checkpoint_sweep_6x300.json`,
  `…/checkpoint_sweep_20x750.json` (Plots in `open_loop_plots_6x300/`, `open_loop_plots_20x750/`);
  Logs `~/logs/slurm-ckpt-sweep-16643458.out`, `…-16643464.out`
- Sweeps Lauf 3: `/data/g1_dex3_finetune/blockstacking_vision/g1_dex3_blockstacking_vision_v2/checkpoint_sweep_6x300_20260814.json`,
  `…/checkpoint_sweep_20x750.json`
- HF (privat, nur Inferenzdateien): `luca-mue/groot-g1dex3-n17-vision-v1-ckpt{5000,10000,15000,44000}`
