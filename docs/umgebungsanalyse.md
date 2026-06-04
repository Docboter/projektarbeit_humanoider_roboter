# Umgebungsanalyse: Training & Simulation

> Erstellt: 2026-06-04 — Automatische Analyse der aktuellen Entwicklungsumgebung

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

#### Domain-Gap-Messung (2026-06-04)

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

### 3. DEX3-Fingerqualität als bekanntes, ungelöstes Problem

Die Open-Loop-Auswertung zeigt, dass die DEX3-Dimensionen noisig sind — selbst auf realen Bildern. Das ist dokumentiert, wurde aber nicht adressiert. Wenn die Fingersteuerung im einfachsten Fall (real → real) bereits schlecht ist, ist Sim-Eval der falsche nächste Schritt; der Fehler liegt früher.

---

## Konkrete technische Bugs

### Episode-Length-Diskrepanz (behoben)

> **Status 2026-06-04: erledigt.** Die ursprüngliche Diskrepanz (Config 180 s, aber nur
> 1 200 Steps = 40 s ausgeführt) besteht nicht mehr. `episode_length_s = 180.0`
> ([g1_dex3_blockstack_env.py:259](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L259))
> und der Eval-Loop berechnet `max_steps = episode_length_s * policy_hz`
> ([run_g1_dex3_sim_eval.py:141](../Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py#L141)) =
> 5 400 Steps = 3 Minuten. Geblieben ist nur ein **veralteter Docstring-Kommentar**
> ([g1_dex3_blockstack_env.py:9](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L9)),
> der noch „1200 Steps @ 30 Hz = 40 Sekunden" sagt — rein kosmetisch.

### WANDB_MODE=offline hardcoded

`entrypoint.sh` setzt `WANDB_MODE=offline` bedingungslos — auch auf vast.ai, wo Internet verfügbar ist. W&B-Daten müssen manuell nachgesyncet werden, ohne dass das irgendwo gewarnt wird.

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

### Nächster sinnvoller Schritt (mit GPU)

**Option A — Vision Encoder fine-tunen:**
`tune_visual = true` in der Trainings-Config, auf einem Mix aus realen und Sim-ähnlichen Bildern trainieren. Direktester Fix, aber teurer (mehr VRAM, längere Laufzeit).

**Option B — Domain Randomization in der Sim:**
Texturen, Beleuchtung, Hintergründe randomisieren, sodass Sim-Bilder realen Bildern ähnlicher werden. Kein Retraining nötig, aber erfordert Arbeit an der Sim-Env.

**Eval-Set aufbauen:**
~20 % der Episoden zurückhalten, `eval_strategy = "steps"` mit `eval_steps = 5000` — damit Checkpoint-Selektion beim nächsten Lauf nicht mehr blind ist.

### Längerfristig

RL (SAC/PPO auf Sim) macht erst Sinn, nachdem entweder der Domain Gap gelöst ist oder das Reward-Signal rein propriozeptiv ist (Vision-unabhängig). Sonst lernt RL eine Policy, die nur im Sim funktioniert und genauso an Domain Gap scheitert.

---

## Zusammenfassung

| Bereich | Status | Priorität |
|---------|--------|-----------|
| Obs/Action-Alignment Training↔Sim | ✅ korrekt | — |
| Containerisierung & Reproduzierbarkeit | ✅ solide | — |
| Episode-Length-Bug | ✅ behoben | — |
| Validation während Training | ⚠️ fehlt | hoch |
| Domain Gap (frozen ViT + Sim-Eval) | ❌ strukturell | hoch |
| DEX3-Fingerqualität | ⚠️ noisig | mittel |
| WANDB_MODE offline hardcoded | ⚠️ unkomfortabel | niedrig |
| Joint Sign-Flip Convention | ⚠️ Risiko | niedrig |

**Die Infrastruktur ist gut** — kein prinzipieller Fehler im Container-Setup oder im Obs/Action-Alignment. Der Kern des Problems liegt im Training-Eval-Loop: Frozen ViT + Real-only Training + Sim-Eval ist kein valides Setup — der gemessene Domain Gap (mittlere Cosine-Distanz 0.260, `cam_left_wrist` 0.427) bestätigt das quantitativ und ist der eigentliche Hebel.
