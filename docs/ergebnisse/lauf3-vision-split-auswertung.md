# Auswertung — dritter Trainingsdurchlauf, Vision-Encoder **mit** Split & Augmentierung (`g1_dex3_blockstacking_vision_v2`)

**Erstellt:** 2026-08-14 · **Run-ID:** `tp1nc699` (W&B-Projekt `gr00t-g1-dex3`, Entity
`projektarbeit_humanoider_roboter`) · **Modell:** GR00T N1.6 (3,29 Mrd. Parameter),
Finetune mit aufgetautem Vision-Encoder (`tune_visual = true`), Color-Jitter aktiv,
**80/20-Train-Test-Split**, Task „stack the blocks".

> Dies ist der **dritte** vollständige Lauf und der erste im ganzen Projekt, für den eine
> **echte Validierungszahl auf zurückgehaltenen Episoden** existiert. Er löst Schritt 2 aus
> [`next-steps.md`](../../next-steps.md) ein und ist die in
> [`lauf2-vision-auswertung.md`](lauf2-vision-auswertung.md) §6.2 geforderte Gegenprobe:
> `TUNE_VISUAL=1`, aber diesmal **mit** Augmentierung — Lauf 2 hatte gar keinen Jitter.
>
> Datengrundlage: die W&B-Kurven von `tp1nc699` und der Checkpoint-Sweep
> [`Simulation/runs/20260814/01/checkpoint_sweep.json`](../../Simulation/runs/20260814/01/checkpoint_sweep.json).

---

## 1. Kurzfazit (TL;DR)

- **Erstmals ein Validierungssignal — und es zeigt eine klare U-Kurve.** Die Validierungs-MSE
  fällt von 0,01335 (Step 5.000) auf **0,00716 bei Step 30.000** und steigt danach wieder auf
  0,00898 bei Step 44.000. **Der letzte Checkpoint ist um 25 % schlechter als der beste.**
- **Damit ist Overfitting im Projekt zum ersten Mal belegt statt vermutet.** Zwischen Step
  30.000 und 44.000 fällt der Trainingsloss um 44 %, während die Validierungs-MSE um 25 %
  steigt — die klassische Schere.
- **Rückwirkend heißt das: Lauf 1 und Lauf 2 haben blind den letzten Checkpoint genommen**
  und damit sehr wahrscheinlich einen schlechteren als den verfügbar besten in die Sim-Eval
  geschickt.
- **Statistischer Vorbehalt, der die Aussage begrenzt:** Der Sweep lief mit den Defaults —
  **6 Episoden, je 300 Steps**. Gegen Step 20.000/25.000/35.000 ist Step 30.000 **nicht
  signifikant** besser (Abschnitt 5). Belastbar ist nur: *ab ~20.000 auskonvergiert, der
  letzte Checkpoint ist nicht der beste.*
- **Das Training selbst war sauber:** echter Neustart (`resume_from_checkpoint = None`, der
  Resume-Guard hat gegriffen), Cosine-LR vollständig ausgelaufen, keine NaN, grad_norm
  gesund.
- **Nachtrag vom Abend des 2026-08-14 — die Verhaltens-Evaluation ist gefahren, und sie fällt
  positiv aus.** checkpoint-30000 kommandiert im Closed Loop **27,6 % Fingerspanne** statt der
  20,5 % des Vorgänger-Checkpoints (10 Episoden à 40 s, vollständige Trennung gegen die
  Vergleichsmessung, p = 3,3 · 10⁻⁴). Damit ist dieser Lauf der **erste, dessen Wirkung auf das
  Verhalten belegt ist** — und zugleich der Beleg, dass er allein nicht reicht: `lifted` bleibt
  0/10. Details in Abschnitt 8.3.

---

## 2. Lauf-Eckdaten

| Feld | Wert |
|---|---|
| Run-ID / Name | `tp1nc699` / `g1_dex3_blockstacking_vision_v2` |
| State | `finished` |
| Start / Ende | 2026-08-13 21:12 UTC → 2026-08-14 09:39 UTC |
| Laufzeit | 44.843 s (**12 h 27 min**) |
| Hardware | **4 GPUs**, Knoten `ggpu183` (KISSKI), DeepSpeed Stage 2, bf16 |
| Steps / Batch | **44.000** × global 32 (per-device 8 × 4) = **1,41 Mio. Samples** |
| Durchsatz | 0,98 Steps/s, **31,4 Samples/s** |
| Epochen | ≈ **7,8** (1,41 Mio. Samples / ~180.000 Trainings-Steps bei 240 Episoden) |
| Trainierbare Teile | Vision-Encoder + Projector + Diffusion-Head + VLLN (`tune_llm = false`) |
| LR / Schedule | 1e-4, cosine, `warmup_ratio = 0.1` → Peak bei Step 4.400, Ende 1,6e-13 |
| Augmentierung | **Color-Jitter aktiv** — hue 0,08 · contrast 0,4 · brightness 0,3 · saturation 0,5. Kein State-Dropout (`state_dropout_prob = 0`), keine Rotation |
| Split | **`TRAIN_TEST_SPLIT=1`** → train `0:240`, test `240:301` |
| Action-Repräsentation | Arme `RELATIVE`, Hände (`*_dex3`) `ABSOLUTE` — wie Lauf 1 und 2 |
| `action_horizon` | 16 (deckt sich mit dem `--action-horizon 16` des Sweeps) |
| `eval_strategy` | `"no"` — wie erwartet, In-Training-Eval ist im Fork strukturell nicht vorhanden |
| `resume_from_checkpoint` | **`None`** → echter Neustart |
| Checkpoints | alle 5.000 Steps (`save_steps`), zusätzlich der finale bei 44.000 |
| Output | `/data/g1_dex3_finetune/blockstacking_vision/g1_dex3_blockstacking_vision_v2` |

> **Zum Namen:** Das Experiment heißt `…_vision_v2`, ist aber der **dritte** Lauf. Der
> `v1`-Namespace gehört Lauf 2 (`ajgoskon`). Dazwischen liegt der abgebrochene Versuch
> `m5vcsjg9` vom 2026-08-13 (~27 min), der unbemerkt einen alten Checkpoint fortgesetzt hat —
> das war der Anlass für den Resume-Guard
> ([`lib_resume_guard.sh`](../../Training/scripts/lib_resume_guard.sh)). Er zählt nicht als
> eigener Lauf.

---

## 3. Trainingsdynamik — gesund ✅

Mittelwerte je 2.000-Step-Fenster aus der W&B-History:

| Step | 0 | 2k | 6k | 10k | 14k | 20k | 26k | **30k** | 34k | 38k | 42k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `train/loss` | 0,514 | 0,077 | 0,057 | 0,036 | 0,026 | 0,020 | 0,016 | **0,013** | 0,008 | 0,008 | 0,007 |

| Metrik | Verlauf | Bewertung |
|---|---|---|
| `train/loss` | 1,267 (Step 80) → 0,0073 (Ende) | sauberer, monotoner Abfall, kein NaN |
| `train/grad_norm` | ~1,3–2,1 früh → **0,073** final | gesund sinkend; Clipping (`max_grad_norm = 1`) nur in den ersten ~2.000 Steps aktiv |
| `train/learning_rate` | Cosine, Peak 1e-4 bei Step ~4.400 → 1,6e-13 | korrekt für 44.000 Steps, vollständig ausgelaufen |

Der finale Loss (0,0073) liegt unter Lauf 1 (0,0081) und Lauf 2 (0,0093) — was **nichts
Gutes bedeutet**, sondern genau der Punkt ist: siehe Abschnitt 4.2.

---

## 4. Der Checkpoint-Sweep — das eigentliche Ergebnis

Gemessen mit [`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py) auf dem
**zurückgehaltenen** Test-Split (`240:301`), 6 gleichmäßig verteilte Episoden, je 300 Steps,
Action-Horizon 16.

| Step | mean MSE | mean MAE | ep240 | ep252 | ep264 | ep276 | ep288 | ep300 |
|---|---|---|---|---|---|---|---|---|
| 5.000 | 0,01335 | 0,0694 | 0,0109 | 0,0112 | 0,0143 | 0,0103 | 0,0169 | 0,0166 |
| 10.000 | 0,00915 | 0,0572 | 0,0096 | 0,0095 | 0,0092 | 0,0056 | 0,0132 | 0,0079 |
| 15.000 | 0,00911 | 0,0518 | 0,0079 | 0,0083 | 0,0106 | 0,0068 | 0,0129 | 0,0080 |
| 20.000 | 0,00780 | 0,0478 | 0,0077 | 0,0063 | 0,0055 | 0,0047 | 0,0142 | 0,0085 |
| 25.000 | 0,00752 | 0,0447 | 0,0062 | 0,0056 | 0,0078 | 0,0036 | 0,0133 | 0,0087 |
| **30.000** | **0,00716** ★ | 0,0418 | 0,0073 | 0,0049 | 0,0080 | 0,0037 | 0,0129 | 0,0062 |
| 35.000 | 0,00763 | **0,0417** | 0,0045 | 0,0048 | 0,0065 | 0,0033 | 0,0135 | 0,0132 |
| 40.000 | 0,00855 | 0,0419 | 0,0071 | 0,0052 | 0,0114 | 0,0031 | 0,0136 | 0,0108 |
| 44.000 | 0,00898 | 0,0418 | 0,0068 | 0,0049 | 0,0118 | 0,0032 | 0,0138 | 0,0134 |

RMSE bei Step 30.000: **0,085** in normierten Aktionseinheiten.

### 4.1 Overfitting ist erstmals sichtbar

Von Step 30.000 bis 44.000:

| | Step 30.000 | Step 44.000 | Δ |
|---|---|---|---|
| `train/loss` (Fenstermittel) | 0,0131 | 0,0073 | **−44 %** |
| Validierungs-MSE | 0,00716 | 0,00898 | **+25 %** |

Die Richtungen sind gegenläufig — das ist die Definition von Overfitting.

> **Methodischer Vorbehalt:** Die beiden Größen sind **nicht dieselbe Metrik**. `train/loss`
> ist der Flow-Matching-Velocity-MSE auf verrauschten Aktionen bei zufälligem Noise-Level;
> die Sweep-MSE ist der Aktions-MSE nach 4 Denoising-Schritten. Die **Absolutwerte** sind
> deshalb nicht vergleichbar — die **Trends** sehr wohl, und die genügen für die Aussage.

### 4.2 Was das rückwirkend über Lauf 1 und 2 sagt

Beide früheren Läufe haben den letzten Checkpoint in die Sim-Eval geschickt, weil es keine
Alternative gab. Dass ein niedrigerer finaler Trainingsloss **kein** besseres Modell bedeutet,
ist jetzt gemessen und nicht mehr nur ein Lehrbuchargument. Die Loss-Vergleiche zwischen den
Läufen in [`lauf1-auswertung.md`](lauf1-auswertung.md) und
[`lauf2-vision-auswertung.md`](lauf2-vision-auswertung.md) sind entsprechend zu lesen: sie
belegen gesunde Optimierung, keine Modellqualität.

### 4.3 Die Verschlechterung sind Ausreißer, keine Breitendegradation

Zwischen Step 30.000 und 44.000 steigt die **MSE um 25 %**, während die **MAE bei 0,0418
exakt flach** bleibt. Der Fehler verlagert sich also in **wenige große Abweichungen**, statt
breit zuzunehmen. Die Einzelepisoden bestätigen das:

| Episode | 30.000 → 44.000 | |
|---|---|---|
| ep264 | 0,0080 → 0,0118 | **+47 %** |
| ep300 | 0,0062 → 0,0134 | **+116 %** |
| ep240 | 0,0073 → 0,0068 | −7 % |
| ep252 | 0,0049 → 0,0049 | ±0 % |
| ep276 | 0,0037 → 0,0032 | −14 % |
| ep288 | 0,0129 → 0,0138 | +7 % |

Der gesamte Anstieg stammt aus **zwei** Episoden; vier von sechs verbessern sich in diesem
Fenster sogar leicht. Praktisch: das Modell wird nicht global schlechter, sondern auf
einzelnen Trajektorien **instabil**. Für den Closed-Loop ist das die ungünstigere Sorte
Fehler — ein einzelner grober Ausreißer reicht, um einen Griff scheitern zu lassen.

### 4.4 Episode 288 lernt ab Step 10.000 nichts mehr

0,0169 (5k) → 0,0132 (10k) → danach 0,0129–0,0142 **ohne jede Tendenz** über 34.000 weitere
Steps. Sie ist in 8 von 9 Checkpoints die schlechteste Episode. Das Muster passt zu einer
Episode außerhalb der Trainingsverteilung (andere Blockposition, anderer Ablauf). Der Plot
`open_loop_plots/ckpt30000_ep288.jpeg` im Lauf-Verzeichnis wäre der nächste Blick — bislang
ungeprüft.

---

## 5. Belastbarkeit — was die 6 Episoden **nicht** hergeben

Gepaarter t-Test über die Einzelepisoden, jeweils gegen checkpoint-30000
(positiv = 30.000 besser):

| Vergleich | Δ mean MSE | t | Bewertung |
|---|---|---|---|
| vs 5.000 | +0,00619 | 6,27 | signifikant |
| vs 10.000 | +0,00199 | 3,43 | signifikant |
| vs 15.000 | +0,00194 | 3,44 | signifikant |
| vs 20.000 | +0,00064 | 0,96 | n.s. |
| vs 25.000 | +0,00036 | 0,74 | n.s. |
| vs 35.000 | +0,00046 | 0,33 | n.s. |
| vs 40.000 | +0,00139 | 1,60 | Trend |
| vs 44.000 | +0,00182 | 1,45 | n.s. |

**Belastbar ist:** ab ~20.000 ist der Lauf auskonvergiert, und der letzte Checkpoint ist
nicht der beste. **Nicht belastbar ist:** welcher aus dem Fenster 20.000–35.000 gewinnt. Selbst
30.000 vs. 44.000 ist mit t = 1,45 nur ein Trend, der fast vollständig von ep300 getragen wird.

**Zweiter Vorbehalt — nur die ersten 40 % jeder Episode.** Der Sweep lief mit dem Default
`--steps 300`. Bei ~750 Steps je Episode (~226.000 Steps / 301 Episoden — hergeleitet, nicht
direkt am Datensatz geprüft) bewertet die Checkpoint-Auswahl damit im Wesentlichen die
**Anfahrphase**. Greifen und Ablegen — genau das, woran alle bisherigen Sim-Evals gescheitert
sind — geht in die Auswahl **gar nicht ein**.

**Nicht vergleichbar mit Lauf 1.** Die dort nachgemessenen 0,001817 (checkpoint-175000) sind
auf **Trainingsdaten** entstanden, weil es damals keinen Split gab. Lauf 3 ist nicht „4×
schlechter", sondern zum ersten Mal ehrlich gemessen.

---

## 6. Noch zu verifizieren: war der Split beim Training aktiv?

Der Sweep protokolliert `"split": "test"` und `"split_range": "240:301"` — zum
**Sweep**-Zeitpunkt stand der Split also in `meta/info.json`. Ob er auch **während des
Trainings** aktiv war, geht aus der JSON nicht hervor: die Guards in
[`checkpoint_sweep.py:236–280`](../../Training/scripts/checkpoint_sweep.py#L236-L280) warnen
ausschließlich auf stdout und hinterlassen keine Spur im Ergebnis.

Auf dem Cluster zu prüfen:

```bash
# 1. Hat der Trainingslauf das Split-Protokoll geschrieben?
cat /data/g1_dex3_finetune/blockstacking_vision/g1_dex3_blockstacking_vision_v2/split.json

# 2. Hat der Launcher den Split angewandt?
grep -i "TRAIN_TEST_SPLIT\|\[split\]" logs/slurm-<training-jobid>.out

# 3. Hat der Sweep gewarnt?
grep -i "WARNUNG" logs/slurm-ckpt-sweep-<jobid>.out
```

Erwartet: `split.json` mit `"test": "240:301"`, keine Warnung im Sweep-Log. Trifft das zu,
gelten die Zahlen aus Abschnitt 4 unverändert. Trifft es **nicht** zu, wären die
Test-Episoden mittrainiert worden und der gesamte Sweep wertlos.

---

## 7. Einordnung: was dieser Lauf **nicht** zeigt

`TUNE_VISUAL=1` hat den Vision-Encoder ausschließlich auf **Realbildern** trainiert — der
Color-Jitter variiert Helligkeit, Kontrast, Sättigung und Farbton, aber **nicht** Geometrie,
Textur oder Rendering-Stil. Der Vorbehalt aus [`next-steps.md`](../../next-steps.md) §2 gilt
unverändert: dass Farb-Jitter allein den Gap zur Sim schließt, ist nicht gesagt, und der
Ausreißer der [Domain-Gap-Messung](domain-gap-analyse.md) ist mit `cam_left_wrist` (0,36)
ausgerechnet eine Nahbereichskamera.

Die MSE-Zahlen oben messen die Nachahmung auf **echten Datensatz-Bildern**. Sie sind das
Kriterium für die **Checkpoint-Auswahl** — nicht für den Sim-Transfer. Die Frage, ob Lauf 3
den Domain-Gap bewegt, entscheidet weiterhin der `span`-Test.

---

## 8. Handlungsempfehlungen

### 8.1 Sweep nachschärfen, bevor der Checkpoint ins Gate geht
Die Entscheidung „30.000 statt 44.000" ruht auf 6 Episoden à 300 Steps. Das nachzuschärfen
kostet einen Bruchteil der 12,5 h Training:

```bash
export EVAL_NUM_TRAJ=20          # statt 6 → SEM etwa −45 %
export EVAL_STEPS=750            # volle Episodenlänge, inkl. Greif-/Ablegephase
export EVAL_CHECKPOINTS=20000,25000,30000,35000,44000
export RUN_DIR=/data/g1_dex3_finetune/blockstacking_vision/g1_dex3_blockstacking_vision_v2
sbatch Training/kisski_open_loop_eval.sh
```

> `RUN_DIR` zeigt bewusst auf das **Experiment**-Unterverzeichnis, nicht auf den
> `blockstacking_vision`-Namespace — sonst mischt der Sweep die Checkpoints von Lauf 2 und
> Lauf 3 mit kollidierenden Step-Nummern.

### 8.2 Für den nächsten Lauf: `MAX_STEPS=30000` genügt
Die letzten 14.000 Steps (≈ 4 h auf 4 GPUs) haben die Validierung verschlechtert. Bei
gleicher Konfiguration ist alles jenseits von ~30.000 Steps verschenkte Rechenzeit — vorbehalten
der Nachmessung aus 8.1, die den Umschlagpunkt genauer eingrenzt.

### 8.3 ✅ erledigt — das `span`-Gate ist gefahren (2026-08-14, abends)

Schritt 3 aus [`next-steps.md`](../../next-steps.md), gefahren mit checkpoint-30000 statt dem
letzten. Ergebnis:

| Messung | alter Checkpoint | **checkpoint-30000** |
|---|---|---|
| Fingerspanne auf **echten** Bildern (Median-Verhältnis) | 1,00 | **1,00** |
| Fingerspanne im **Closed Loop** (Median) | 0,43 rad = 20,5 % | **0,577 rad = 27,6 %** |
| Episoden ≥ 60 % der Demonstration | 0/5 | **3/10** |
| `lifted` | 0/5 | **0/10** |

Beide Läufe im identischen Zeitfenster (40 s je Episode). Die Verteilungen überlappen **gar
nicht** — exakter einseitiger Rangtest p = 3,3 · 10⁻⁴. Und die Würfel-Anhebung läuft mit der
Fingerspanne mit (Spearman +0,62), die bloße Verschiebung nicht (+0,05): der Greifbefehl wirkt
mechanisch, statt nur Nebenprodukt von mehr Kontakt zu sein.

**Einordnung für diesen Lauf:** damit ist `TUNE_VISUAL=1` **mit** Augmentierung als Maßnahme
belegt — die Gegenprobe zu [`lauf2-vision-auswertung.md`](lauf2-vision-auswertung.md) §6.2 ist
bestanden, wo derselbe Schalter *ohne* Jitter zum Politik-Kollaps führte. Der Gap ist damit
allerdings nicht geschlossen: 27,6 % in der Sim gegen 100 % auf echten Bildern. Die vollständige
Auswertung steht in
[diagnose-chronik.md, Läufe 33/34](diagnose-chronik.md#läufe-3334-runs2026081403-runs2026081404-der-tune_visual-checkpoint-im-closed-loop).

### 8.4 Episode 288 anschauen
Billig und potenziell aufschlussreich: warum stagniert eine einzelne Testepisode ab Step
10.000 vollständig? Wenn sich das als systematische Lücke im Datensatz entpuppt, ist das ein
Befund für die Ausarbeitung.

---

## 9. Offene Punkte

- **Split-Verifikation** (Abschnitt 6) — bis dahin stehen alle Zahlen unter Vorbehalt.
- **Nachgeschärfter Sweep** mit 20 Episoden über die volle Episodenlänge (8.1).
- ~~**Verhaltens-Evaluation** von checkpoint-30000~~ — erledigt am 2026-08-14 (8.3).
- **Erneute Domain-Gap-Messung** mit dem getunten Encoder aus Lauf 3 — ist der Gap durch das
  Jitter-Training kleiner geworden als die 0,22 / 0,36 aus der letzten Messung? Nach 8.3 die
  interessanteste offene Zahl: die Fingerspanne ist um ein Drittel gestiegen, also sollte sich
  auch die Kosinus-Distanz je Kamera bewegt haben — `./Simulation/server_rl_run.sh gap`.
- **Plot-Sichtung** `open_loop_plots/ckpt30000_ep288.jpeg` (Abschnitt 4.4).

---

## 10. Verwendete Artefakte / Quellen

| Zweck | Quelle |
|---|---|
| W&B-Trainingsmetriken Lauf 3 | Run `tp1nc699`, Projekt `gr00t-g1-dex3` |
| Checkpoint-Sweep (Rohdaten) | [`Simulation/runs/20260814/01/checkpoint_sweep.json`](../../Simulation/runs/20260814/01/checkpoint_sweep.json) |
| Sweep-Werkzeug | [`Training/scripts/checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py) |
| Split-Mechanismus | [`Training/scripts/lib_split.sh`](../../Training/scripts/lib_split.sh) · [train-test-split.md](../training/train-test-split.md) |
| Resume-Guard (Vorgeschichte `m5vcsjg9`) | [`Training/scripts/lib_resume_guard.sh`](../../Training/scripts/lib_resume_guard.sh) |
| Vorlauf 1 (frozen Encoder) | [`lauf1-auswertung.md`](lauf1-auswertung.md) (`i6n1t613`) |
| Vorlauf 2 (Vision, ohne Jitter) | [`lauf2-vision-auswertung.md`](lauf2-vision-auswertung.md) (`ajgoskon`) |
| Domain-Gap-Messung | [`domain-gap-analyse.md`](domain-gap-analyse.md) |
| Methodik-Review Sim-Eval | [`sim-bewertung.md`](sim-bewertung.md) |
| Priorisierung / Gate-Definition | [`next-steps.md`](../../next-steps.md) |
| Closed-Loop-Eval + `span` von checkpoint-30000 (8.3) | `Simulation/runs/20260814/03` und `04` (gitignored) · Auswertung in [diagnose-chronik.md](diagnose-chronik.md#läufe-3334-runs2026081403-runs2026081404-der-tune_visual-checkpoint-im-closed-loop) |
