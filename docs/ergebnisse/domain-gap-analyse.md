# Domain-Gap-Analyse: Real → Isaac Sim

> **TL;DR:** Messung des visuellen Domain-Gaps zwischen realen Trainingsbildern und
> Isaac-Sim-Renderings per Cosine-Distanz (frozen SigLIP-ViT), inkl. Handlungsoptionen. Relevant, um
> den Real→Sim-Gap als Ursache für Closed-Loop-Fehlschläge zu belegen; Ergebnis in §Ergebnis.

> Aktuelle Messung: 2026-08-08 (`runs/20260808/22`). Die Erstmessung vom 2026-06-04 ist überholt
> (Kamera-Neukalibrierung + Albedo-Fix „schwarze Hände") und archiviert in
> [`historie.md`](../historie.md).

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
  -v "$REPO/Simulation/runs/20260602/02:/workspace/sim_debug:ro" \
  -v "$REPO/Simulation/scripts:/workspace/scripts:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "/tmp/output:/output" \
  lucam03/projekt-humanoider-roboter-sim-vastai:latest \
  /app/Groot-1.6/.venv/bin/python /workspace/scripts/measure_domain_gap.py
```

**Frames:**
- Real: `Simulation/camera_reference/dataset_cam_*.png` (aus dem Teleop-Datensatz)
- Sim: `Simulation/runs/20260602/02/_debug_obs_cam_*.png` (Isaac Sim Debug-Frames)

**Metrik:** Cosine-Distanz zwischen CLS-/Pooled-Embeddings (0 = identisch, ~0.5–0.7 = zufälliges Bildpaar)

---

## Ergebnis

Neumessung vom 2026-08-08 (`runs/20260808/22`), gemessen nach Kamerakalibrierung, mit schwarzen
Händen (`BLACK_HANDS=1`) und aufgehelltem Boden (`RL_GROUND_COLOR=0.35,0.35,0.36`), feste
Beleuchtung (`DR_ENABLED=0`, Dome 2000).

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

Zu beachten: die Kennzahl enthält auch Szeneninhalt (Armpose, Würfel, **Tischabstand ~15 cm
abweichend**), nicht nur Renderqualität — und die Wrist-Kamera ist dafür die empfindlichste.

---

## Interpretation & Nachträge

**Was sich geändert hat.** Der Mittelwert liegt jetzt bei 0,2229 und damit unter der real↔real-
Grundlinie (0,2726) und praktisch auf der sim-internen Streuung (0,2163): real→sim ist das
0,8-fache des real↔real-Abstands. Ursache war zu einem großen Teil **Albedo**, nicht Beleuchtung —
die reale DEX3-Hand ist schwarz, das URDF-Asset weiß, und ein Beleuchtungsfaktor kann ein
Kontrastverhältnis nicht ändern. Mit schwarzen Händen trifft der Wrist-Kontrast 68,9 bei real 65,0
(vorher 36,1), der Kopfkamera-Kontrast 59,0 bei real 59,4.

**Was offen bleibt.** `cam_left_wrist` ist mit 0,3556 weiter der Ausreißer (1,3× real↔real) und
verfehlt das vorab festgelegte Kriterium von < 0,35. Konsequenz nach der in
[`diagnose-chronik.md`](diagnose-chronik.md) festgehaltenen vorregistrierten Regel: kein weiterer
Albedo-Versuch, sondern Option 3 (`TUNE_VISUAL=1`) — vorher aber Schritt 3 (BC-Erfolgsrate in der
Sim), der die Zielgröße direkt misst statt über den Proxy.

> **Nachtrag: Schritt 3 ist gelaufen.** Lauf 30 (2026-08-12) misst die BC-Erfolgsrate in der Sim
> mit **0/20**. Nach der vorregistrierten Regel ist `TUNE_VISUAL=1` damit bestätigt. Lauf 30 zeigt
> zusätzlich, dass die Politik nur ~19 % der demonstrierten Fingerspannweite kommandiert — der
> Befund zeigt also auf Wahrnehmung/Politik, nicht nur auf das Rendering. Details:
> [`diagnose-chronik.md`](diagnose-chronik.md).

---

## Handlungsoptionen

Der ursprüngliche Optionenkatalog (Domain Randomization zuerst, 2-Kamera-Konfiguration als
schnellster Weg zum Ergebnis, `tune_visual = true` als strukturell korrekte Langzeitlösung) ist
durch die vorregistrierte Regel und Lauf 30 überholt: Option 2 (2-Kamera) wurde nie verfolgt,
Domain Randomization (Option 1) ebenfalls nicht. Vollständiger alter Stand inkl. Aufwandsschätzung
in [`historie.md`](../historie.md).

Stattdessen ist direkt **Option 3** (`TUNE_VISUAL=1`) angelaufen, bestätigt durch Lauf 30 — die
konkrete Umsetzung und ihre Ergebnisse stehen in [`diagnose-chronik.md`](diagnose-chronik.md)
sowie in [`co-training.md`](../training/co-training.md).
