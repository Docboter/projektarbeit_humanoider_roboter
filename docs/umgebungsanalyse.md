# Umgebungsanalyse: Training & Simulation

> Erstellt: 2026-06-04 — Automatische Analyse der Entwicklungsumgebung
>
> ⚠️ **Zeitdokument (Stand 2026-06-04).** Die Analyse als solche ist gültig geblieben, mehrere
> Einzelbefunde und alle Empfehlungen sind es nicht mehr. Was sich seither geändert hat, ist
> unten jeweils direkt am Befund vermerkt. Die drei wichtigsten Punkte vorweg:
> - Die **Domain-Gap-Zahlen dieses Dokuments (Mittel 0,260 / `cam_left_wrist` 0,427) sind
>   überholt.** Sie wurden vor der Kamera-Neukalibrierung gemessen; die Neumessung vom
>   2026-08-08 liegt bei **0,2229 / 0,3556** → [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md).
> - Die **Empfehlungen sind abgearbeitet**: Vision-Encoder-Fine-tuning (`TUNE_VISUAL=1`),
>   Domain Randomization (`USE_AUGMENTATION=1`, Default) und der 80/20-Split
>   (`TRAIN_TEST_SPLIT=1`) sind implementiert.
> - **RL ist gebaut** — allerdings mit FPO statt der hier empfohlenen SAC/PPO
>   → [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md).

---

## Was konzeptuell gut ist

- **Obs/Action-Alignment**: Training-Config und Sim sind perfekt synchron — 4 Kameras, 28-dim Joints, gleiche Modality-Keys (`video.cam_left_high` usw.), gleiche Chunk-Size (16). Das ist solide.
- **Containerisierung**: Die dreistufige Pipeline (Download → Convert → Train) ist sauber und reproduzierbar. Env-Var-Only-Config funktioniert auf vast.ai, KISSKI und lokal ohne Änderungen.
- **Diagnose**: Der Root-Cause (Domain Gap + frozen ViT) wurde durch Replay-Vergleich methodisch korrekt isoliert.

---

## Fundamentale Denkfehler

### 1. Frozen Vision Encoder + Sim-Eval — strukturell inkompatibel

Das Modell wurde auf realen Teleop-Bildern trainiert, der Vision Encoder dabei eingefroren (`tune_visual = false`). Evaluiert wird dann mit synthetischen Isaac-Sim-Bildern. Diese Kombination war **von Anfang an zum Scheitern verurteilt** — nicht als unvorhergesehene Entdeckung, sondern als vorhersehbare Konsequenz des Setups.

Der eigentliche Denkfehler: Es wurde erheblicher Aufwand in die Sim-Pipeline investiert, bevor die fundamentale Frage beantwortet war: *Kann das Modell überhaupt auf Sim-Bilder generalisieren?* Ein kurzer Offline-Test (Sim-Frame durch den ViT, Embedding-Distanz zu Real-Frame messen) hätte das in unter einer Stunde gezeigt.

#### Domain-Gap-Messung (2026-06-04) — ⚠️ überholt

> **Diese Erstmessung ist nicht mehr gültig.** Sie entstand vor den Kamera-Fixes vom August 2026
> (falsch geschriebene Kamera-Prim-Orientierungen, unwirksame Head-Cam-FOV, falsch montiertes
> Stereo-Paar, DLSS-Leerframes). Die Neumessung vom 2026-08-08 auf korrekt kalibrierten Kameras
> ergibt **Mittelwert 0,2229** und **`cam_left_wrist` 0,3556** — der Gap ist kleiner als hier
> angenommen, die Rangfolge der Kameras bleibt aber bestehen. Aktuelle Zahlen und Interpretation:
> [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md). Die folgenden Werte
> bleiben nur als Vergleichspunkt stehen.

Durchgeführt mit `google/siglip-so400m-patch14-224` (identische Gewichte zum frozen GR00T-ViT).
Frames: `Simulation/camera_reference/` (Real) vs. `Simulation/runs/202606002/02/` (Isaac Sim).
Script: [`Simulation/scripts/measure_domain_gap.py`](../Simulation/scripts/measure_domain_gap.py)

| Kamera | Cosine-Distanz Real→Sim | Bewertung |
|---|---|---|
| `cam_left_high` | 0.148 | moderat |
| `cam_right_high` | 0.214 | groß |
| `cam_right_wrist` | 0.249 | groß |
| `cam_left_wrist` | **0.427** | **kritisch** |
| **Mittelwert** | **0.260** | groß |

Baseline-Vergleich:
- Real-vs-Real (verschiedene Kameras): **0.273** — der mittlere Real→Sim-Gap liegt leicht darunter
- Sim-vs-Sim (verschiedene Kameras): **0.211**

**Interpretation:** Die Overhead-Kameras allein wären noch vertretbar (0.15–0.21). Der kritische Befund ist `cam_left_wrist` mit 0.427 — diese Kamera zeigt die Greifaktion aus nächster Nähe. Synthethische Hände, Tisch-Texturen und Beleuchtung weichen so stark von realen Bildern ab, dass der frozen ViT das Wrist-Signal nicht übertragen kann. Weil die Policy alle 4 Kameras gleichzeitig verarbeitet, blockiert dieser eine Eingang die Ausführung.

### 2. Training ohne Validation = Blindflug

`eval_strategy = "no"` bedeutet: 175 000 Trainingsschritte, kein Checkpoint-Vergleich, keine Metriken auf einem Hold-Out-Set. Der verwendete Checkpoint (Step 175k) wurde arbiträr gewählt. Es ist unbekannt, ob 50k oder 100k besser gewesen wären. Für ein reproduzierbares Forschungsprojekt ist das eine echte Lücke.

> **Nachtrag:** Die Infrastruktur für den Hold-Out existiert inzwischen —
> `TRAIN_TEST_SPLIT=1` hält 20 % der Episoden zurück
> ([run_finetuning.sh:102](../Training/scripts/run_finetuning.sh#L102),
> [training/train-test-split.md](training/train-test-split.md)). Der Befund gilt damit für den
> 1. Lauf, nicht mehr für das Setup.

### 3. DEX3-Fingerqualität als bekanntes, ungelöstes Problem

Die Open-Loop-Auswertung zeigt, dass die DEX3-Dimensionen noisig sind — selbst auf realen Bildern. Das ist dokumentiert, wurde aber nicht adressiert. Wenn die Fingersteuerung im einfachsten Fall (real → real) bereits schlecht ist, ist Sim-Eval der falsche nächste Schritt; der Fehler liegt früher.

---

## Konkrete technische Bugs

### Episode-Length-Diskrepanz (behoben)

> **Status 2026-06-04: erledigt.** Die ursprüngliche Diskrepanz (Config 180 s, aber nur
> 1 200 Steps = 40 s ausgeführt) besteht nicht mehr. Der Eval-Loop berechnet
> `max_steps = episode_length_s * policy_hz`
> ([run_g1_dex3_sim_eval.py:207](../Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py#L207)),
> die Konfiguration wird also tatsächlich ausgeführt.
>
> **Nachtrag 2026-08:** Der Default steht inzwischen auf `episode_length_s = 300.0`
> ([g1_dex3_blockstack_env.py:411](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L411))
> = 9 000 Steps = 5 Minuten; überschreibbar per `EPISODE_LENGTH_S`. Der Docstring-Kommentar am
> Dateikopf ([g1_dex3_blockstack_env.py:9](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L9))
> nennt weiterhin „5400 Steps @ 30 Hz = 180 Sekunden" und ist damit erneut veraltet — rein
> kosmetisch, der Wert wird aus der Config gelesen.

### WANDB_MODE=offline als Default

`entrypoint.sh` setzt `WANDB_MODE=offline` — auch auf vast.ai, wo Internet verfügbar ist.
W&B-Daten müssen dann manuell nachgesyncet werden.

> **Korrektur zum ursprünglichen Befund:** „hardcoded" trifft nicht (mehr) zu. Der Entrypoint
> setzt `export WANDB_MODE="${WANDB_MODE:-offline}"`
> ([entrypoint.sh:175](../Training/scripts/entrypoint.sh#L175)) — offline ist nur der Default und
> lässt sich per `WANDB_MODE=online` überschreiben. Der Sim-/RL-Entrypoint verhält sich genauso
> ([entrypoint_rl.sh:126](../Simulation/scripts/entrypoint_rl.sh#L126)).

Relevante Datei: [`Training/scripts/entrypoint.sh`](../Training/scripts/entrypoint.sh)

### Joint Sign-Flip — Convention-Leak-Risiko

Die Indices 17, 19, 24, 26 (DEX3-Proximalgelenke) werden sowohl in Observations als auch Actions negiert, weil USD-Achsen und Dataset-Convention gegensätzlich sind. Es wird nie verifiziert, dass der GR00T-Server dieselbe Convention annimmt. Ein Mismatch würde sich als stilles Fehlverhalten äußern.

Relevante Datei: [`Simulation/g1_dex3_sim/client.py`](../Simulation/g1_dex3_sim/client.py)

### Success-Kriterium möglicherweise unerreichbar

Alle drei Bedingungen müssen gleichzeitig erfüllt sein:
- xy-Distanz < 3 cm zwischen Blöcken
- Height > 8 cm (Stack bestätigt)
- Velocity < 5 cm/s (stabil)

Bei Friction = 3.0 und physikalischer Blockdrift im Sim kann "stabil" möglicherweise nie zuverlässig getriggert werden.

Relevante Datei: [`Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py`](../Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py)

---

## Empfehlungen

### Sofort (ohne GPU)

1. ~~**Episode-Length-Bug fixen**~~ ✅ **Erledigt** — Config steht auf 180 s, Eval rechnet korrekt 5 400 Steps (→ oben); nur ein kosmetischer Docstring-Kommentar ist noch veraltet.
2. ~~**Domain-Gap quantifizieren**~~ ✅ **Erledigt** — Messergebnis: mittlere Cosine-Distanz 0.260, `cam_left_wrist` kritisch bei 0.427 (→ oben).

### Nächster sinnvoller Schritt (mit GPU) — ⚠️ inzwischen umgesetzt

> Alle drei Empfehlungen dieses Abschnitts sind implementiert. Sie stehen als Begründung der
> damaligen Priorisierung, nicht mehr als offene Aufgaben.

**Option A — Vision Encoder fine-tunen:** ✅ **umgesetzt** als `TUNE_VISUAL=1`
(→ [run_finetuning_vision.sh](../Training/scripts/run_finetuning_vision.sh), eigener
`blockstacking_vision`-Namespace). Auswertung:
[ergebnisse/lauf2-vision-auswertung.md](ergebnisse/lauf2-vision-auswertung.md).
`tune_visual = true` in der Trainings-Config, auf einem Mix aus realen und Sim-ähnlichen Bildern trainieren. Direktester Fix, aber teurer (mehr VRAM, längere Laufzeit).

**Option B — Domain Randomization in der Sim:** ✅ **umgesetzt** — Bild-Augmentierung läuft per
`USE_AUGMENTATION=1` standardmäßig mit ([run_finetuning.sh:52](../Training/scripts/run_finetuning.sh#L52));
in der Sim-Env kommen Albedo- und Dome-Light-Regler dazu.
Texturen, Beleuchtung, Hintergründe randomisieren, sodass Sim-Bilder realen Bildern ähnlicher werden. Kein Retraining nötig, aber erfordert Arbeit an der Sim-Env.

**Eval-Set aufbauen:** ✅ **umgesetzt** als `TRAIN_TEST_SPLIT=1` (Ratio über `TRAIN_SPLIT_RATIO`,
Default 0.8) → [training/train-test-split.md](training/train-test-split.md).
~20 % der Episoden zurückhalten und nach dem Lauf [`checkpoint_sweep.py`](../Training/scripts/checkpoint_sweep.py) über alle Checkpoints fahren — damit Checkpoint-Selektion beim nächsten Lauf nicht mehr blind ist. *(Korrektur 2026-08-13: hier stand `eval_strategy = "steps"` mit `eval_steps = 5000`. In-Training-Eval existiert im Fork nicht — `factory.py:26` bricht mit `assert eval_strategy == "no"` ab. Begründung: [training/train-test-split.md](training/train-test-split.md) Hinweis 1.)*

### Längerfristig — ⚠️ teilweise überholt

RL (SAC/PPO auf Sim) macht erst Sinn, nachdem entweder der Domain Gap gelöst ist oder das Reward-Signal rein propriozeptiv ist (Vision-unabhängig). Sonst lernt RL eine Policy, die nur im Sim funktioniert und genauso an Domain Gap scheitert.

> **Nachtrag 2026-08:** RL ist gebaut — mit **FPO** statt SAC/PPO und nur auf dem Action-Head
> ([rl_finetune.py](../Simulation/g1_dex3_sim/rl_finetune.py)). Die hier formulierte
> Reihenfolge-Warnung hat sich bestätigt: die Pipeline läuft end-to-end, der Lernerfolg ist
> unverifiziert, und die Diagnose der Läufe 29–31 verweist zurück auf Wahrnehmung/Politik statt
> auf RL → [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md).

---

## Zusammenfassung

| Bereich | Status | Priorität |
|---------|--------|-----------|
| Obs/Action-Alignment Training↔Sim | ✅ korrekt | — |
| Containerisierung & Reproduzierbarkeit | ✅ solide | — |
| Episode-Length-Bug | ✅ behoben | — |
| Validation während Training | ⚠️ fehlt → ✅ `TRAIN_TEST_SPLIT=1` verfügbar | hoch |
| Domain Gap (frozen ViT + Sim-Eval) | ❌ strukturell → ⚠️ verkleinert (0,2229 nach Neumessung) | hoch |
| DEX3-Fingerqualität | ⚠️ noisig | mittel |
| WANDB_MODE offline hardcoded | ⚠️ unkomfortabel → überschreibbar (`WANDB_MODE=online`) | niedrig |
| Joint Sign-Flip Convention | ⚠️ Risiko | niedrig |

> Die Spalte „Status" trägt jeweils zuerst die Bewertung vom 2026-06-04, nach dem Pfeil den
> Stand August 2026.

**Die Infrastruktur ist gut** — kein prinzipieller Fehler im Container-Setup oder im Obs/Action-Alignment. Der Kern des Problems liegt im Training-Eval-Loop: Frozen ViT + Real-only Training + Sim-Eval ist kein valides Setup — der gemessene Domain Gap bestätigt das quantitativ und ist der eigentliche Hebel.

> **Nachtrag 2026-08:** Die Neumessung nach der Kamera-Kalibrierung senkt den Gap auf
> **0,2229 im Mittel** und **0,3556** für `cam_left_wrist` — unter die real↔real-Baseline
> (0,2726). Der Domain Gap bleibt ein Faktor, ist aber nicht mehr die alleinige Erklärung für
> die 0-%-Erfolgsrate. Die Diagnose der Läufe 29–31 zeigt zusätzlich, dass die Politik nur
> ~19 % der demonstrierten Greifbewegung kommandiert — eine Fehlerquelle im Modell selbst,
> unabhängig vom Rendering.
