# Domain-Gap-Analyse: Real → Isaac Sim

> Durchgeführt: 2026-06-04

> ⚠️ **Überholt — neu gemessen am 2026-08-08 (`runs/20260808/22`).** Die Zahlen im Hauptteil stammen
> von vor dem Isaac-Sim-6.0-Port und vor der Kamerakalibrierung (Iteration 13/14: Sichtfeld 47,2° →
> 75°, Montagepunkt auf `d435_link`, parallele statt konvergierender Stereobasis). Sie beschreiben
> ein Rendering, das es so nicht mehr gibt. Sie bleiben als Referenzpunkt stehen; **gültig ist die
> Neumessung unten.**

## Neumessung 2026-08-08 (`runs/20260808/22`)

Gemessen nach Kamerakalibrierung, mit schwarzen Händen (`BLACK_HANDS=1`) und aufgehelltem Boden
(`RL_GROUND_COLOR=0.35,0.35,0.36`), feste Beleuchtung (`DR_ENABLED=0`, Dome 2000).

| Kamera | Juni 2026 | **2026-08-08** | Delta |
|---|---|---|---|
| `cam_left_high` | 0,1477 | **0,1121** | −0,0356 |
| `cam_right_high` | 0,2136 | **0,1310** | −0,0826 |
| `cam_left_wrist` | 0,4275 | **0,3556** | −0,0719 |
| `cam_right_wrist` | 0,2491 | **0,2930** | +0,0439 |
| **Mittelwert** | 0,2595 | **0,2229** | −0,0366 |

| Grundlinie | Juni | 2026-08-08 |
|---|---|---|
| real → real (andere Kamera) | 0,2726 | 0,2726 |
| sim → sim (andere Kamera) | 0,2111 | 0,2163 |

**Was sich geändert hat.** Der Mittelwert liegt jetzt bei 0,2229 und damit unter der real↔real-
Grundlinie (0,2726) und praktisch auf der sim-internen Streuung (0,2163): real→sim ist das
0,8-fache des real↔real-Abstands. Ursache war zu einem großen Teil **Albedo**, nicht Beleuchtung —
die reale DEX3-Hand ist schwarz, das URDF-Asset weiß, und ein Beleuchtungsfaktor kann ein
Kontrastverhältnis nicht ändern. Mit schwarzen Händen trifft der Wrist-Kontrast 68,9 bei real 65,0
(vorher 36,1), der Kopfkamera-Kontrast 59,0 bei real 59,4.

**Was offen bleibt.** `cam_left_wrist` ist mit 0,3556 weiter der Ausreißer (1,3× real↔real) und
verfehlt das vorab festgelegte Kriterium von < 0,35. Konsequenz nach der in
[`rl-anleitung.md`](../weiterfuehrend/rl-anleitung.md) festgehaltenen Regel: kein weiterer
Albedo-Versuch, sondern Option 3 (`TUNE_VISUAL=1`) — vorher aber Schritt 3 (BC-Erfolgsrate in der
Sim), der die Zielgröße direkt misst statt über den Proxy. Zu beachten: die Kennzahl enthält auch
Szeneninhalt (Armpose, Würfel, **Tischabstand ~15 cm abweichend**), nicht nur Renderqualität — und
die Wrist-Kamera ist dafür die empfindlichste.

---

## Fragestellung

Kann der frozen Vision Encoder von GR00T N1.6 reale Teleop-Bilder und synthetische Isaac-Sim-Bilder gleich behandeln? Da `tune_visual = false` während des Fine-Tunings gesetzt war, sind die ViT-Gewichte identisch mit dem vortrainierten SigLIP-Checkpoint — eine Domänenanpassung hat nie stattgefunden.

---

## Methodik

**Modell:** `google/siglip-so400m-patch14-224`
Da `tune_visual = false`, sind die Gewichte identisch mit dem frozen GR00T-ViT — das 6 GB GR00T-Modell ist nicht nötig.

**Script:** [`Simulation/scripts/measure_domain_gap.py`](../../Simulation/scripts/measure_domain_gap.py)

**Ausführung:** im `lucam03/projekt-humanoider-roboter-sim-vastai`-Container (hat GR00T-venv mit torch + transformers)

```bash
docker run --rm \
  -v "$REPO/Simulation/camera_reference:/workspace/camera_reference:ro" \
  -v "$REPO/Simulation/runs/202606002/02:/workspace/sim_debug:ro" \
  -v "$REPO/Simulation/scripts:/workspace/scripts:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "/tmp/output:/output" \
  lucam03/projekt-humanoider-roboter-sim-vastai:latest \
  /app/Groot-1.6/.venv/bin/python /workspace/scripts/measure_domain_gap.py
```

**Frames:**
- Real: `Simulation/camera_reference/dataset_cam_*.png` (aus dem Teleop-Datensatz)
- Sim: `Simulation/runs/202606002/02/_debug_obs_cam_*.png` (Isaac Sim Debug-Frames)

**Metrik:** Cosine-Distanz zwischen CLS-/Pooled-Embeddings (0 = identisch, ~0.5–0.7 = zufälliges Bildpaar)

---

## Ergebnisse

### Per-Kamera Real → Sim

| Kamera | Cosine-Distanz | Bewertung |
|---|---|---|
| `cam_left_high` | 0.148 | moderat |
| `cam_right_high` | 0.214 | groß |
| `cam_right_wrist` | 0.249 | groß |
| `cam_left_wrist` | **0.427** | **kritisch** |
| **Mittelwert** | **0.260** | groß |

### Baseline-Vergleiche

| Vergleich | Cosine-Distanz |
|---|---|
| Real → Sim (Mittelwert) | **0.260** |
| Real → Real (verschiedene Kameras) | 0.273 |
| Sim → Sim (verschiedene Kameras) | 0.211 |

---

## Interpretation

### Gesamtbild

Der mittlere Real→Sim-Gap (0.260) liegt leicht **unterhalb** des Real→Real-Cross-View-Gaps (0.273). Das heißt: im Durchschnitt wirken Sim-Bilder auf den frozen ViT nicht wesentlich fremder als eine andere Kamera-Perspektive innerhalb des realen Datensatzes. Dieser Befund ist überraschend positiv — und würde für sich allein keine Katastrophe bedeuten.

### Der Blocker: cam_left_wrist

`cam_left_wrist` hat mit **0.427** einen kritischen Ausreißer, der rund **1.6× über dem Real→Real-Baseline** liegt. Diese Kamera zeigt die Greifaktion aus nächster Nähe: synthetische Roboterhand, Tischoberfläche und Nahbereichs-Beleuchtung weichen so stark vom realen Trainings-Material ab, dass der frozen ViT das Signal nicht überträgt.

Da GR00T alle 4 Kameras gleichzeitig als Input verarbeitet, genügt ein einziger kritisch-fremder Eingang, um die Politiksequenz zu destabilisieren — und das ist genau die handnahe Kamera, die für Greifentscheidungen entscheidend ist.

### Warum die Sim zu "sauber" ist

Die Sim-interne Varianz (0.211) ist **niedriger** als die Real-interne Varianz (0.273). Isaac Sim rendert konsistente Materialien, gleichförmige Beleuchtung und identische Hintergrundszenarien ohne die natürlichen Schwankungen realer Aufnahmen (Reflexionen, leichte Beleuchtungsänderungen, Kamerarauschen). Das macht die Sim-Verteilung im Embedding-Space eng und weit von der breiten Real-Verteilung entfernt.

---

## Handlungsoptionen

### Option 1: Domain Randomization (empfohlen als erster Schritt)

**Aufwand:** 1–2 Tage, kein GPU, kein Retraining
**Ziel:** Wrist-Gap von 0.427 auf ≤ 0.25 senken

In [`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) für die Wrist-Kameras aktivieren:
- Textur-Randomization (Tischoberfläche, Roboterhand-Material)
- Beleuchtungs-Randomization (Intensität, Farbe, Position)
- Hintergrund-Randomization (optional)

Erfolg messbar mit `measure_domain_gap.py` nach jeder Änderung — kein GPU-Run nötig zum Testen.

**Risiko:** Wie viel Gap-Closure realistisch erreichbar ist, hängt vom Renderer ab. PhysX-basierte Isaac-Lab-Umgebungen haben begrenzte Randomization-Tiefe verglichen mit Ray-Tracing-Renderern.

### Option 2: 2-Kamera-Konfiguration (schnellster Weg zum messbaren Ergebnis)

**Aufwand:** ~1 Tag Konfiguration + 1 KISSKI-Run (≈24 h)

Die zwei Overhead-Kameras (`cam_left_high`: 0.148, `cam_right_high`: 0.214) liegen im vertretbaren Bereich. Ein neues Training nur mit diesen zwei Kameras eliminiert den kritischen Wrist-Blocker vollständig.

Änderungen für den nächsten Trainingslauf:
- Modality-Config auf `g1_dex3_2cam_config` umstellen (existiert bereits in [`g1_dex3_config.py`](../../app/Groot-1.6/examples/G1_DEX3/g1_dex3_config.py))
- `eval_strategy = "steps"`, `eval_steps = 5000` setzen (behebt den Blind-Checkpoint-Fehler)
- Sim-Eval analog auf 2 Kameras umstellen

**Risiko:** Ohne Wrist-Information sieht die Policy nicht, was die Hand gerade macht — feine Fingersteuerung wird schwieriger. Der Robot kann Blöcke lokalisieren, aber keine propriozeptive Greif-Korrektur aus Kamerabildern ableiten.

### Option 3: tune_visual = true (strukturell korrekte Lösung)

**Aufwand:** Sim-Frames + KISSKI-Run (≈48 h, mehr VRAM)

Setzt `tune_visual = true` im Training-Config und trainiert auf einem Mix aus echten und Sim-ähnlichen Bildern. Der ViT lernt die Sim-Embedding-Verteilung direkt.

Empfohlene Reihenfolge:
1. Erst Domain Randomization einbauen (Option 1)
2. Dann Sim-Frames aus der Randomized-Sim als Trainings-Augmentation hinzufügen
3. `tune_visual = true` mit erhöhtem VRAM-Budget auf KISSKI H100

Diese Option ist die richtige Langzeitlösung, lohnt sich aber erst nachdem Option 1 oder 2 einen Baseline-Erfolg gezeigt hat.

---

## Empfohlene Reihenfolge

```
Jetzt (ohne GPU):
  → Option 2 konfigurieren + KISSKI-Job submittieren (läuft autonom)

Parallel:
  → Option 1 entwickeln, mit measure_domain_gap.py iterativ testen

Wenn Option 2 Erfolg zeigt:
  → Option 3 als Follow-up mit tune_visual=true

Wenn Option 1 den Wrist-Gap auf ≤ 0.25 senkt:
  → Erneute Sim-Eval mit dem bestehenden Checkpoint (kein Retraining nötig!)
```

---

## Rohdaten

JSON-Output aus dem Messungslauf:

```json
{
  "model": "google/siglip-so400m-patch14-224",
  "per_camera": {
    "cam_left_high": 0.1477,
    "cam_right_high": 0.2136,
    "cam_left_wrist": 0.4275,
    "cam_right_wrist": 0.2491
  },
  "mean_real_sim": 0.2595,
  "mean_real_real_crossview": 0.2726,
  "mean_sim_sim_crossview": 0.2111,
  "verdict": "Large domain gap",
  "action": "ViT fine-tuning strongly recommended"
}
```
