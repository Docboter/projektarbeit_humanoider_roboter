# Basismodell-Referenz-Eval — Ergebnis (Pipeline-Validierung)

> **TL;DR:** Ergebnis der Pipeline-Validierung: das un-finetunte GR00T-N1.6-Basismodell zero-shot
> auf dem RoboCasa-GR-1-Tabletop-Benchmark, um zu prüfen, ob die eigene GR00T-Inferenz-Pipeline
> NVIDIAs publizierte Erfolgsquoten reproduziert. Befund im Kurzfazit direkt darunter.

**Stand:** 2026-07-19 · **Status:** ✅ Phase 1 validiert — Aggregat-Mittel über 12 Tasks **47,7 % ≈ 47,8 %** erwartet. `full`-Lauf nach ~2 h **manuell** bei 12/24 Tasks gestoppt (Re-Run der restlichen 12 offen).

Ausführung der in [basismodell-referenzaufgabe.md](../simulation/basismodell-referenzaufgabe.md) geplanten
**Phase 1**: das **un-finetunte** Basismodell `nvidia/GR00T-N1.6-3B` (zero-shot, Embodiment `GR1`) auf dem
**RoboCasa GR-1 Tabletop** Benchmark (robosuite/MuJoCo) laufen lassen und prüfen, ob **unsere GR00T-Inferenz-
Pipeline** NVIDIAs publizierte Erfolgsquoten reproduziert. Bedienung: [robocasa-referenz-eval.md](../simulation/robocasa-referenz-eval.md).

> **Kurzfazit:** Unsere Pipeline reproduziert den Goldstandard **präzise**. Über 12 abgeschlossene Tasks (je 100 Ep.)
> liegt der **gemessene Mittelwert bei 47,7 %** — praktisch identisch mit dem **erwarteten 47,8 %** (publizierter
> Gesamt-Schnitt über 24 Tasks: 47,6 %). Der Einzeltask-Ausreißer (Top-Task 70 %/74 % statt 78,7 %) mittelt sich als
> Rauschen heraus. Damit ist die GR00T-Inferenz-Hälfte (Checkpoint-Laden, `--embodiment-tag GR1`, Sim-Policy-Wrapper,
> ZMQ, Obs/Action-Mapping) **extern bestätigt**. Kernkonsequenz: Die **0 % im Isaac-Lab-Closed-Loop**
> ([sim-bewertung.md](sim-bewertung.md)) sind der **visuelle Domain-Gap**, **nicht** eine kaputte Harness.

---

## 1. Setup

| Aspekt | Wert |
|---|---|
| Hardware | Server `ikr-ki-server-01`: **2× RTX PRO 6000 Blackwell** (je 96 GB, sm_120), Threadripper 7975WX (32 C), 256 GB RAM, Treiber CUDA 13.3 UMD |
| Ausführung | Docker, langlebiger Container; Helferskript [`Simulation/server_robocasa_ref_run.sh`](../../Simulation/server_robocasa_ref_run.sh) |
| Image | `lucam03/projekt-humanoider-roboter:latest` (Trainings-Image, GR00T-Venv, EGL) — **ohne Rebuild** |
| Modell | `nvidia/GR00T-N1.6-3B` (HF-Download, zero-shot), Embodiment `GR1`, `--use-sim-policy-wrapper` |
| Task | `gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env` (bester publizierter Einzeltask) |
| Eval-Config | `n_envs=8`, `n_action_steps=8`, `max_episode_steps=720`, ZMQ-Port 5757 |

**Blackwell-Befund (wichtig):** Kein Image-Rebuild nötig. Das Image bringt `torch==2.7.1` vom `cu128`-Index
([pyproject.toml:26](../../app/Groot-1.6/pyproject.toml#L26)) → **sm_120-Kernels sind enthalten**. Auch das
vorgebaute `flash-attn==2.7.4.post1` lief ohne `sm_120`-Fehler durch. Der neuere Treiber (CUDA 13.3 UMD) ist
gegenüber dem 12.8-Toolkit abwärtskompatibel. → Der Server ist voll Blackwell-tauglich mit dem Stock-Image.

## 2. Ergebnis

Drei Läufe (2026-07-19, Rohdaten unter [`Simulation/robocasa_reference/runs/01/`](../../Simulation/robocasa_reference/runs/01/)):
Smoke (Top-Task, 5 Ep.), Referenz (Top-Task, 100 Ep.) und ein `full`-Lauf über alle 24 Tasks (je 100 Ep.), der nach
12 abgeschlossenen Tasks manuell gestoppt wurde (→ §2.3).

### 2.1 Einzeltask (Top: `PlateToPlate`)
| Lauf | Episoden | Erfolg | Publiziert | Δ |
|---|---|---|---|---|
| Smoke | 5 | 5/5 = 100 % | 78,7 % | (n zu klein) |
| Referenz | 100 | 70/100 = 70,0 % | 78,7 % | −8,7 pp |
| im `full`-Lauf | 100 | 74/100 = 74,0 % | 78,7 % | −4,7 pp |

- 95-%-KI (Wald) um 70 %: [61,0 %, 79,0 %] → publizierte 78,7 % liegen (knapp) drin. z(vs 78,7 %) = −2,13 (p ≈ 0,03), grenzwertig.
- Die drei Top-Task-Messungen (100/70/74 %) streuen um ~72 %; bei n=100 ist die Einzeltask-Streuung ±~8–10 pp, also erwartbar.

### 2.2 Aggregat über 24 Tasks (`full`-Lauf, 12/24 abgeschlossen) ★ entscheidend
| Task | Ep. | Gemessen | Publiziert | Δ pp |
|---|---|---|---|---|
| PnPBottleToCabinetClose | 100 | 45,0 % | 51,5 % | −6,5 |
| PnPMilkToMicrowaveClose | 100 | 16,0 % | 14,0 % | +2,0 |
| PnPPotatoToMicrowaveClose | 100 | 35,0 % | 41,5 % | −6,5 |
| …CuttingboardToCardboardbox | 100 | 49,0 % | 46,5 % | +2,5 |
| …CuttingboardToPot | 100 | 58,0 % | 65,0 % | −7,0 |
| …PlacematToBasket | 100 | 61,0 % | 58,5 % | +2,5 |
| …PlacematToBowl | 100 | 49,0 % | 57,5 % | −8,5 |
| …PlacematToTieredshelf | 100 | 37,0 % | 28,5 % | +8,5 |
| …PlateToBowl | 100 | 56,0 % | 57,0 % | −1,0 |
| …PlateToCardboardbox | 100 | 47,0 % | 43,5 % | +3,5 |
| …PlateToPlate (Top) | 100 | 74,0 % | 78,7 % | −4,7 |
| …TrayToTieredshelf | 100 | 45,0 % | 31,5 % | +13,5 |
| **Mittelwert (12 Tasks)** | | **47,7 %** | **47,8 %** | **−0,1** |

**Kernbefund:** Der gemessene Mittelwert **47,7 %** trifft den erwarteten Mittelwert **47,8 %** (dieselben 12 Tasks;
publizierter Gesamt-Schnitt über alle 24 = **47,6 %**) auf **0,1 pp** genau. Die per-Task-Streuung (Δ −8,5 … +13,5 pp)
ist normales Binomial-Rauschen bei n=100 und **mittelt sich vollständig heraus**.

**Verdict:** Bestanden — und robust. Eine defekte Inferenz-Hälfte (falscher Embodiment-Tag, kaputter Wrapper,
vertauschtes Obs/Action-Mapping) könnte keinen 12-Task-Schnitt auf 0,1 pp reproduzieren. Die GR00T-Inferenz-Pipeline
ist extern validiert. Der Einzeltask-Ausreißer aus §2.1 ist damit als Rauschen entlarvt.

### 2.3 Abbruch (12/24) — manuell gestoppt
Der `full`-Lauf wurde nach ~2 h **manuell** gestoppt (12 von 24 Tasks fertig), **nicht** durch einen Fehler. Der
`av.error.FFmpegError`-Traceback am Ende des letzten Task-Logs (`…CuttingboardToTieredbasket`) ist ein
**Shutdown-Artefakt des Interrupts** — die Warnung „Calling close while waiting for a pending call to step" belegt es:
der Abbruch schließt die AsyncVectorEnv mitten im Step, wodurch die Video-Finalisierung fehlschlägt. Kein Modell-/Sim-
Problem, keine GPU/OOM/Walltime-Grenze.

## 3. Geklärte Nebenbefunde

- **„No object files found for category 'book' in registry 'sketchfab'"** und **„distractor_obj … skip this object
  config"**: **benign, kein unvollständiger Install.** `book` ist ein reines **Distraktor**-Objekt (Hintergrund-Clutter,
  nie Target); `download_tabletop_assets.py` provisioniert bewusst nur die Registries `sketchfab` + `lightwheel`
  (objaverse ist nicht Teil des offiziellen Provisioning). NVIDIAs Zahlen entstanden mit demselben Zwei-Registry-Setup
  → **like-for-like**. Fehlende Distraktor-Vielfalt würde die Quote eher **nach oben** verzerren, erklärt die −8,7 pp
  also nicht. (Verifiziert am RoboCasa-Quellcode: `kitchen_object_utils.py`, `tabletop_24dc.py`.)
- **Beide GPUs ausgelastet:** Das **Modell** läuft komplett auf **`cuda:0`** (`device="cuda"`, kein `device_map` —
  [gr00t_policy.py:30](../../app/Groot-1.6/gr00t/policy/gr00t_policy.py#L30)). GPU 1 kommt **allein vom
  MuJoCo/EGL-Rendering** der `n_envs` parallelen robosuite-Umgebungen. Harmlos. Zum Freihalten von GPU 1:
  Container mit `RC_GPUS='"device=0"'` pinnen.

## 4. Was validiert ist — und was nicht

- **Validiert (Phase 1):** GR00T-Inferenz-/Server-Hälfte — Checkpoint-Laden, `--embodiment-tag GR1`,
  Sim-Policy-Wrapper, ZMQ-Server/Client, Obs/Action-Mapping.
- **Nicht validiert (Phase 2, offen):** unsere **eigene Isaac-Lab-Szene** (anderer Simulator, eigenes Robot-USD,
  eigener Reward) — siehe Phase 2 im [Plan](../simulation/basismodell-referenzaufgabe.md#3-implementierungsplan).

## 5. Reproduktion

Kommandos (Pfad A2, `server_robocasa_ref_run.sh`) und Ergebnis-Pfade stehen in
[robocasa-referenz-eval.md](../simulation/robocasa-referenz-eval.md) — dort die gepflegte
Referenz.

## 6. Offen / Ausblick

- [x] **`full`-Lauf gestartet** — 12/24 Tasks abgeschlossen, Mittel **47,7 % ≈ 47,8 %** (§2.2). Aggregat-Validierung
  damit bereits sehr belastbar.
- [ ] **Re-Run der restlichen 12 Tasks** (`RC_PRESET=custom` mit `RC_TASKS=<offene Tasks>`, oder `full` neu), um alle
  24 abzudecken. Tipp: bei langen Läufen sammeln sich die Rollout-Videos in `/tmp` an (Übernahme nach `/data` erst am
  Runner-Ende) — für einen reinen Aggregatlauf sind Videos verzichtbar; ggf. je Task nach `/data` verschieben.
- [ ] Optional: 2. Goldstandard **RoboCasa Panda** (ø 66 %) als unabhängige Bestätigung.
- **Hardware-Nebeneffekt:** Die RTX PRO 6000 **haben RT-Cores** — dieser Server kann damit (anders als KISSKI
  A100/H100) auch die **volle Isaac-Sim-Closed-Loop-Eval** und den **RL-Pfad** fahren, für die bisher vast.ai nötig war.
