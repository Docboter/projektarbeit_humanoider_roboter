# W&B-Auswertung — Run `g1_dex3_blockstacking_v1`

**Erstellt:** 2026-06-02 · **Datenquelle:** W&B-Projekt `gr00t-g1-dex3`
(Entity `projektarbeit_humanoider_roboter`) · **Run-ID:** `i6n1t613`

> Momentaufnahme eines noch **laufenden** Trainings. Die unten genannten
> Schrittzahlen sind der Stand zum Erstellungszeitpunkt; der Lauf läuft weiter.

> 📈 **Interaktive Kurven:** [`wandb-run-charts.html`](wandb-run-charts.html) — Loss /
> LR / grad_norm als eigenständige HTML (im Browser öffnen). Snapshot Stand
> 2026-06-02, ~20:47 UTC (Step ~23.060).

---

## 1. Run-Status (Stand 2026-06-02, ~20:47 UTC)

| Feld | Wert |
|---|---|
| Run-ID / Name | `i6n1t613` / `g1_dex3_blockstacking_v1` |
| Node | `ggpu177` (KISSKI / GWDG), `num_gpus = 1` |
| Start | 2026-06-02 18:37 UTC |
| Letzter Heartbeat | 2026-06-02 20:47 UTC (aktiv, Schritte wachsen weiter) |
| Fortschritt | **global_step ≈ 23.060 / 175.000 (~13,2 %)** |
| Laufzeit bisher | ~8.760 s (~2,4 h) |
| Durchsatz | ~2,6 Steps/s → **~19 h Gesamtlaufzeit** (passt in 48 h KISSKI-Walltime) |

> Hinweis: W&B zeigt den `state` zeitweise als `finished` an, obwohl die
> Schrittzahl zwischen Abfragen weiter steigt — ein Sync-Artefakt. Auch die
> `summaryMetrics` der GraphQL-API können kurzzeitig einen **stale/inkonsistenten**
> Snapshot liefern (z. B. niedrigere Laufzeit/Step als die History) — maßgeblich
> sind die fortlaufende History und der Heartbeat. Der Lauf ist durchgehend **aktiv**
> (kein Neustart/Resume).

---

## 2. Trainingsdynamik — gesund ✅

| Metrik | Verlauf | Bewertung |
|---|---|---|
| `train/loss` | 1,37 → ~0,036 (geglättet ~0,04) | sauberer Abfall, kein NaN |
| `train/grad_norm` | ~2,8 (früh) → ~0,27 | stabil sinkend; Clipping (`max_grad_norm=1`) nur anfangs aktiv |
| Loss-Verlauf | schneller Abfall bis ~Step 2.600 (~0,08), danach **weiter langsam sinkend**: ~0,06 (Step 11k) → ~0,04 (Step 23k) | gesunde, anhaltende Verbesserung — **kein** echtes Plateau |

**Wichtig:** Niedriger Train-Loss ≠ gute Policy. Der Flow-Matching-Loss von VLA-
Modellen liegt grundsätzlich auf kleinen Werten mit hoher Punkt-zu-Punkt-Streuung
(zufälliger Diffusions-Timestep pro Schritt). Auf die **geglättete** Linie achten —
die sinkt weiter. Aussagekräftig für die Policy bleibt aber die **Erfolgsrate in der
Sim-Eval**, nicht diese Kurve.

---

## 3. Over-/Underfitting — derzeit nicht messbar

**Ein belastbarer Befund ist aus diesem Run nicht möglich** — nicht wegen des
Zeitpunkts, sondern wegen der fehlenden Messgröße.

Over-/Underfitting misst man am Auseinanderlaufen von **Train- und Val-Kurve**. In
diesem Run gibt es keine Val-Kurve: `eval_strategy = "no"`,
`enable_open_loop_eval = false` → geloggt sind nur `train/loss`,
`train/learning_rate`, `train/grad_norm`. Die klassische Overfitting-Signatur
(Train-Loss ↓, Val-Loss ↑) ist damit **prinzipiell unsichtbar**, egal wie lange der
Lauf läuft. Zusätzlich: Stand ~13 % (Step 23.060) wäre es auch *mit* Eval noch früh.

Die Train-Loss-Kurve sagt nur: Das Modell passt die **Trainingsverteilung** gut an
das Flow-Matching-Ziel an. Das ist **kein** Generalisierungssignal (s. Abschnitt 2).

**Qualitative Einschätzung (mit Vorbehalt):**

- **Underfitting: unwahrscheinlich.** Loss niedrig und ohne Stocken gefallen,
  grad_norm gesund/sinkend — und der geglättete Loss **sinkt bis Step 23k weiter**
  (~0,06 → ~0,04). Die Optimierung läuft problemlos und ist nicht stehengeblieben.
- **Overfitting: nicht sichtbar, strukturelles Risiko eher moderat-niedrig:**
  1. nur **~3 Epochen** (175k Steps ≈ 3 Durchläufe, `num_train_epochs=3`),
  2. **Partial-Finetune** (nur Projector + Diffusion + 4 LLM-Layer + VLLN; Backbone &
     Visual eingefroren) → wenige trainierbare Parameter,
  3. **kräftige Augmentation** (color jitter, albumentations).
- Das eigentliche *unbeobachtete* Risiko ist „stilles Overfitting" in späteren
  Epochen — der weiter sinkende Train-Loss (bis Step 23k) kann sowohl echte
  Verbesserung als auch beginnende Anpassung an die Trainingsdaten sein; ohne
  Val-Kurve nicht unterscheidbar. Bei noch ~152k offenen Steps nur per Eval klärbar.

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
