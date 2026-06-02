# W&B-Auswertung — Run `g1_dex3_blockstacking_v1`

**Erstellt:** 2026-06-02 · **Datenquelle:** W&B-Projekt `gr00t-g1-dex3`
(Entity `projektarbeit_humanoider_roboter`) · **Run-ID:** `i6n1t613`

> Momentaufnahme eines noch **laufenden** Trainings. Die unten genannten
> Schrittzahlen sind der Stand zum Erstellungszeitpunkt; der Lauf läuft weiter.

> 📈 **Interaktive Kurven:** [`wandb-run-charts.html`](wandb-run-charts.html) — Loss /
> LR / grad_norm als eigenständige HTML (im Browser öffnen). Snapshot Stand
> 2026-06-02, ~19:38 UTC (Step ~11.820).

---

## 1. Run-Status (Stand 2026-06-02, ~19:26 UTC)

| Feld | Wert |
|---|---|
| Run-ID / Name | `i6n1t613` / `g1_dex3_blockstacking_v1` |
| Node | `ggpu177` (KISSKI / GWDG), `num_gpus = 1` |
| Start | 2026-06-02 18:37 UTC |
| Letzter Heartbeat | 2026-06-02 19:26 UTC (aktiv, Schritte wachsen weiter) |
| Fortschritt | **global_step ≈ 9.820 / 175.000 (~5,6 %)** |
| Laufzeit bisher | ~3.831 s (~64 min) |
| Durchsatz | ~2,56 Steps/s → **~19 h Gesamtlaufzeit** (passt in 48 h KISSKI-Walltime) |

> Hinweis: W&B zeigt den `state` zeitweise als `finished` an, obwohl die
> Schrittzahl zwischen Abfragen weiter steigt — ein Sync-Artefakt. Der Lauf ist
> anhand des fortlaufenden Heartbeats und wachsender `global_step` **aktiv**.

---

## 2. Trainingsdynamik — gesund ✅

| Metrik | Verlauf | Bewertung |
|---|---|---|
| `train/loss` | 1,37 → ~0,06 | sauberer Abfall, kein NaN |
| `train/grad_norm` | ~2,8 (früh) → ~0,42 | stabil sinkend; Clipping (`max_grad_norm=1`) nur anfangs aktiv |
| Loss-Plateau | seit ~Step 2.600 bei 0,05–0,09 | **erwartbar** für Flow-Matching/Diffusion-Loss |

**Wichtig:** Niedriger Train-Loss ≠ gute Policy. Der Flow-Matching-Loss von VLA-
Modellen plateaut praktisch immer früh bei kleinen Werten. Aussagekräftig ist die
**Erfolgsrate in der Sim-Eval**, nicht diese Kurve. Das Plateau ist also kein Defekt.

---

## 3. Over-/Underfitting — derzeit nicht messbar

**Ein belastbarer Befund ist aus diesem Run nicht möglich** — nicht wegen des
Zeitpunkts, sondern wegen der fehlenden Messgröße.

Over-/Underfitting misst man am Auseinanderlaufen von **Train- und Val-Kurve**. In
diesem Run gibt es keine Val-Kurve: `eval_strategy = "no"`,
`enable_open_loop_eval = false` → geloggt sind nur `train/loss`,
`train/learning_rate`, `train/grad_norm`. Die klassische Overfitting-Signatur
(Train-Loss ↓, Val-Loss ↑) ist damit **prinzipiell unsichtbar**, egal wie lange der
Lauf läuft. Zusätzlich: Stand ~5,6 % (Step 9.820) wäre es auch *mit* Eval zu früh.

Die Train-Loss-Kurve sagt nur: Das Modell passt die **Trainingsverteilung** gut an
das Flow-Matching-Ziel an. Das ist **kein** Generalisierungssignal (s. Abschnitt 2).

**Qualitative Einschätzung (mit Vorbehalt):**

- **Underfitting: unwahrscheinlich.** Loss niedrig und ohne Stocken gefallen,
  grad_norm gesund/sinkend — die Optimierung läuft problemlos.
- **Overfitting: nicht sichtbar, strukturelles Risiko eher moderat-niedrig:**
  1. nur **~3 Epochen** (175k Steps ≈ 3 Durchläufe, `num_train_epochs=3`),
  2. **Partial-Finetune** (nur Projector + Diffusion + 4 LLM-Layer + VLLN; Backbone &
     Visual eingefroren) → wenige trainierbare Parameter,
  3. **kräftige Augmentation** (color jitter, albumentations).
- Das eigentliche *unbeobachtete* Risiko ist „stilles Overfitting" in späteren
  Epochen bzw. **abnehmender Ertrag** (Train-Loss flach seit ~Step 2.600, aber noch
  ~165k Steps offen) — nur eine Eval würde das zeigen.

**Wie man es tatsächlich beantwortet:**

1. **Open-Loop-Eval einschalten** (`enable_open_loop_eval=true`, ggf.
   `eval_strategy="steps"`; `eval_set_split_ratio=0.1` ist schon gesetzt) → Val-MSE
   über die Zeit. Sauberer Weg ab dem nächsten Lauf.
2. **Pragmatisch ohne Neustart:** einen Zwischen-Checkpoint (z. B. Step 5.000 vs.
   einen späteren) in der **Sim-Eval** auf ungesehenen Episoden vergleichen
   (siehe [train-test-split.md](train-test-split.md)). Erfolgsrate steigt weiter →
   kein Overfitting; fällt bei flachem Train-Loss → Overfitting-Indiz. Die
   Sim-Erfolgsrate ist hier ohnehin die relevantere Metrik als jeder Loss.

---

## 4. LR-Schedule — korrekt für 175k Steps ✅

Verifizierte Config-Werte (`i6n1t613`): `max_steps = 175000`, `warmup_ratio = 0.05`,
`lr_scheduler_type = cosine`, `learning_rate = 1e-4`.

- Warmup = 0,05 × 175.000 = **8.750 Steps** → deckt sich exakt mit dem beobachteten
  Peak-LR-Erreichen (~Step 8.930, LR = 9,9999e-5).
- Ab dort Cosine-Decay über die vollen 175k, klingt am Ende sauber auf ~0 aus.

> Eine frühere Analyse-Version meldete fälschlich „kein LR-Annealing". Sie basierte
> auf einer **Stale-Config** (`max_steps=30000`) aus gesampelten *Failed-Runs* des
> Probe-Tools. Für den realen Lauf mit 175k Steps ist der Schedule **korrekt** —
> hier ist nichts zu ändern.

---

## 5. Handlungsempfehlungen (für den **nächsten** Lauf, nicht den laufenden)

### 5.1 ⚠️ Batch-Size erhöht GPU-Auslastung (größter Hebel)
Aktuell: `global_batch_size = 8`, `per_device_train_batch_size = 8`,
`gradient_accumulation_steps = 1`, `gradient_checkpointing = false`.

Das ist der CLAUDE.md-**Minimal**-Default (für 24-GB-Karten). Auf einer KISSKI-A100
(80 GB) / H100 (94 GB) belegt bs=8 nur ~31 GB — der Rest liegt brach. Empfehlung
laut CLAUDE.md für 80 GB: **64–128**.
- → `GLOBAL_BATCH_SIZE` deutlich erhöhen (z. B. 32–64), ggf. `gradient_checkpointing=true`
  für noch größere Batches. Vorteil: bessere Auslastung + stabilere Gradienten.
- ⚠️ Caveat: Bei festem `max_steps` bedeutet größere Batch = mehr gesehene Samples =
  faktisch mehr Epochen. Für gleiche Datenmenge `max_steps` proportional senken.
- Vorab prüfen: `nvidia-smi` auf dem Node — liegt die VRAM-Nutzung bei ~30/80 GB,
  ist die Reserve real.

### 5.2 ⚠️ Eval ist konfiguriert, aber AUS — über 19 h kein Generalisierungssignal
Infrastruktur vorhanden, aber inaktiv:
- `eval_strategy = "no"`, `do_eval = false`
- `eval_set_split_ratio = 0.1` (Split *wäre* definiert), `eval_steps = 500`
- `enable_open_loop_eval = false` (GR00T-Open-Loop-Eval *vorhanden*, aber aus)

Folge: kein Val-Signal → kein principled Checkpoint-Auswahlkriterium (am Ende blind
Step 175000). Empfehlung: `enable_open_loop_eval=true` (+ ggf. `eval_strategy="steps"`),
um Val-MSE über die Zeit zu sehen und Overfitting/besten Checkpoint zu erkennen.
Siehe [train-test-split.md](train-test-split.md).

---

## 6. Sonstiges (unkritisch)

| Punkt | Wert | Anmerkung |
|---|---|---|
| `save_steps` / `save_total_limit` | 5000 / 40 | ~35 Checkpoints × ~6 GB ≈ **~200 GB** auf `/scratch` — beobachten |
| `deepspeed_stage` | 2 (single GPU) | ZeRO-2 spart Optimizer-Memory, ok |
| `color_jitter` (Modell) | hue .08 / contrast .4 / bright .3 / sat .5 | kräftige Augmentation → gut gegen Overfitting |
| `dataloader_num_workers` | 8 | passt |
| Trainable Teile | projector + diffusion + 4 LLM-Layer + VLLN; LLM/Visual frozen | Standard-GR00T-Partial-Finetune |
| `optim` / Precision | adamw_torch / bf16 + tf32 | sinnvoll |
| `last-error.md` (Repo-Root) | Sim-Joint-Limits | separater **Sim-Eval**-Bug (DEX3-Finger-Default-Pose außerhalb der USD-Limits), **kein** Trainingsproblem |

---

## 7. Kurzfazit

Der laufende Lauf ist mit 175.000 Steps **sauber konfiguriert** (LR-Schedule korrekt,
Trainingsdynamik gesund, keine NaN). **Für den laufenden Job ist nichts zu ändern.**
Für den nächsten Lauf lohnen sich zwei Dinge: **(1) Batch-Size hochziehen**, um die
KISSKI-GPU auszunutzen, und **(2) Eval/Open-Loop-Eval einschalten**, um Checkpoints
bewerten zu können.
