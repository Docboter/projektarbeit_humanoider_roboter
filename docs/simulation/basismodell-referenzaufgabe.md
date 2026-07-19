# Basismodell-Fähigkeiten & Referenzaufgabe zur Sim-Validierung

**Stand:** 2026-06-16
**Frage:** Was kann das Basismodell **GR00T N1.6** (`nvidia/GR00T-N1.6-3B`), das wir als Basis für unser Finetuning verwenden, *out-of-the-box* (zero-shot, ohne Finetuning)? Kann es bereits eine Aufgabe im **Closed Loop** lösen? Und können wir eine solche Aufgabe rekonstruieren, um zu prüfen, ob **unsere Entwicklungsumgebung** sie ebenfalls korrekt umsetzt?

> **TL;DR**
> - **Ja**, das Basismodell löst Closed-Loop-Manipulationsaufgaben zero-shot — **aber nur für Embodiments, die im vortrainierten Set enthalten sind** (`gr1`, `robocasa_panda_omron`, `unitree_g1`, …), **nicht** für unser `NEW_EMBODIMENT` (G1+DEX3). Für `NEW_EMBODIMENT` existieren keine Basisgewichte; das Basismodell kann unseren DEX3-Block-Stack-Task also prinzipiell nicht zero-shot.
> - **„Perfekt"** löst das Basismodell *keine* Aufgabe. Die besten dokumentierten Zero-Shot-Zahlen sind **RoboCasa Panda ø 66,2 %** und **RoboCasa GR-1 Tabletop ø 47,6 %** (Einzeltask bis **78,7 %**). Der Validierungs-Maßstab kann daher nicht „100 % Erfolg" sein, sondern **„die publizierte Erfolgsquote innerhalb statistischer Toleranz reproduzieren"**.
> - **Wichtiger Architektur-Befund:** Diese lösbaren Tasks laufen in **robosuite/MuJoCo** über GR00Ts eigene Eval-Pipeline (`run_gr00t_server.py` + `rollout_policy.py`), **nicht** in unserem **Isaac Lab**. Das trennt die Validierung in zwei klar verschiedene Stufen (siehe Plan).

---

## 1. Was ist GR00T N1.6 und was kann es bereits?

### 1.1 Modell
- Open Vision-Language-Action-Modell, **3 B Parameter**. Aufbau: VLM-Backbone (**N1.6: Cosmos-Reason-2B-Variante**, vorher Eagle) + **Diffusion-Transformer-Action-Head mit Flow-Matching** (N1.6: **32 DiT-Layer**, vorher 16). Pro-Embodiment-MLPs für Proprio-State und Aktions-Decoding, indiziert über die Embodiment-ID. Release **~Dez 2025**, Nachfolger von N1.5/N1.
  - Quellen: [HF-Modelcard](https://huggingface.co/nvidia/GR00T-N1.6-3B), [NVIDIA GEAR](https://research.nvidia.com/labs/gear/gr00t-n1_6/), [Isaac-GR00T `n1.6-release`](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release)
- N1.6 sagt **state-relative Action-Chunks** vorher (nicht mehr absolute Gelenkwinkel) — relevant, weil unsere DEX3-Config Arm-Aktionen ebenfalls relativ, Hand-Aktionen absolut führt ([g1_dex3_config.py:57-81](../../app/Groot-1.6/examples/G1_DEX3/g1_dex3_config.py#L57)).

### 1.2 Zero-shot oder Finetuning nötig?
- Das Repo stellt laut README **„convenient scripts to validate zero-shot performance of the pretrained model"** bereit — **aber** ein **neues/eigenes Embodiment (`new_embodiment`) muss post-trainiert werden**, bevor es brauchbare Aktionen liefert. Die HF-Card framt das Modell explizit als *„adaptable through post-training for specific embodiments"*.
- **Konsequenz für uns:** Unser G1+DEX3 ist als `NEW_EMBODIMENT` registriert ([embodiment_tags.py:14-62](../../app/Groot-1.6/gr00t/data/embodiment_tags.py#L14), [g1_dex3_config.py:150](../../app/Groot-1.6/examples/G1_DEX3/g1_dex3_config.py#L150)). Dafür gibt es **im Basismodell keine Gewichte** → Zero-Shot auf unserem DEX3-Task ist nicht sinnvoll testbar.

### 1.3 Mitgelieferte Embodiments (`embodiment_id.json` der Release-Checkpoint)
| Tag | ID | Closed-Loop zero-shot belegt? |
|---|---|---|
| `oxe_google` | 0 | nur SimplerEnv (dort **finetuned**) |
| `oxe_widowx` | 1 | nur SimplerEnv (dort **finetuned**) |
| `libero_panda` | 2 | LIBERO (dort **finetuned**, ~94–98 %) |
| `unitree_g1` | 8 | vortrainiert, **keine publizierte Zero-Shot-Zahl** |
| `robocasa_panda_omron` | 13 | **ja — ø 66,2 %** (RoboCasa, MuJoCo) |
| `gr1` | 20 | **ja — ø 47,6 %, max 78,7 %** (RoboCasa GR-1 Tabletop, MuJoCo) |
| `behavior_r1_pro` | 24 | — |
| `new_embodiment` | — | **nein** (muss finetuned werden) ← *unser G1+DEX3* |

Quelle: [embodiment_id.json](https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/embodiment_id.json).

### 1.4 Closed-Loop-Benchmarks mit Erfolgsquoten
| Benchmark | Embodiment | Simulator | Modell | Erfolg | Zero-shot? |
|---|---|---|---|---|---|
| **RoboCasa Panda** (25 Kitchen-Tasks) | `robocasa_panda_omron` | robosuite/MuJoCo | `GR00T-N1.6-3B` direkt | **ø 66,2 %** (19–100 %) | pretrained-Embodiment, kein Finetune im Befehl |
| **RoboCasa GR-1 Tabletop** (24 PnP-Tasks, Fourier-Hände) | `gr1` | robosuite/MuJoCo | `GR00T-N1.6-3B` direkt | **ø 47,6 %**, Einzeltask bis **78,7 %** | ja (pretrained) |
| LIBERO | `libero_panda` | robosuite | finetuned | 94–98 % | nein |
| SimplerEnv (Bridge/Fractal) | widowx/google | SimplerEnv | **finetuned** checkpoints | 52–57 % | **nein** (Vorsicht: oft als „base" zitiert) |
| DROID | oxe_droid | — | base | nur Open-Loop-MSE | — |

Quellen: [robocasa/README.md](../../app/Groot-1.6/examples/robocasa/README.md) (lokal), [robocasa-gr1-tabletop-tasks/README.md](../../app/Groot-1.6/examples/robocasa-gr1-tabletop-tasks/README.md) (lokal, Tabelle ø 47,6 %), [Isaac-GR00T robocasa](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/examples/robocasa/README.md), [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).

**GR-1 Tabletop — beste Einzeltasks (N1.6, zero-shot):**
`PnPNovelFromPlateToPlateSplitA` **78,7 %**, `…FromTrayToPlate` 71,0 %, `…FromCuttingboardToPan` 68,5 %, `…FromCuttingboardToPot` 65,0 %, `…FromTrayToPot` 64,5 %, `…FromPlacematToPlate` 63,0 %.

### 1.5 Unitree G1 speziell
`unitree_g1` (ID 8) ist vortrainiert, NVIDIAs öffentliche G1-Demos sind aber **task-spezifisch post-trainiert** (Mug-Placement etc., „10K–30K steps"), **nicht** als zero-shot beworben. Es existiert **keine publizierte Zero-Shot-Erfolgsquote für den G1**. Die nächstliegende belegte Zero-Shot-Fähigkeit eines **Humanoiden mit geschickten Händen** ist daher **GR-1 + Fourier-Hände**.

---

## 2. Kritische Einordnung der Idee

Die Grundidee — eine vom Basismodell beherrschte Aufgabe rekonstruieren und prüfen, ob unsere Umgebung sie ebenfalls löst — ist **methodisch wertvoll** (sie validiert die Pipeline gegen einen externen Goldstandard). Vier Punkte müssen aber sauber adressiert werden:

1. **„Perfekt" gibt es nicht.** Selbst die besten Zero-Shot-Tasks liegen bei 66 %/79 %. Validierungskriterium = **Reproduktion der publizierten Quote ± Toleranz**, nicht 100 %.
2. **Simulator-Mismatch.** Die lösbaren Tasks laufen in **robosuite/MuJoCo**; unsere Entwicklungsumgebung ist **Isaac Lab**. „Unsere Umgebung" kann zwei Dinge meinen:
   - **(a) Die GR00T-Inferenz-/Server-Hälfte** (Checkpoint-Laden, Embodiment-Tag, `--use-sim-policy-wrapper`, ZMQ, Obs/Action-Mapping) — die teilen wir mit der offiziellen Pipeline. **Günstig validierbar.**
   - **(b) Die Isaac-Lab-Szenen-/Robot-/Reward-Hälfte** ([g1_dex3_blockstack_env.py](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py), [client.py](../../Simulation/g1_dex3_sim/client.py)) — die ist **eigenentwickelt** und in der offiziellen RoboCasa-Pipeline **nicht** enthalten. Vollvalidierung erfordert Nachbau eines base-fähigen Tasks **in Isaac Lab** (teuer, eigenes Robot-USD nötig).
3. **Kein Basis-Embodiment für DEX3.** Der vorhandene Baseline-Modus ([entrypoint_baseline.sh](../../Simulation/scripts/entrypoint_baseline.sh)) fährt zwar das Basismodell mit `--embodiment-tag UNITREE_G1` + Stock-G1-Greifer auf unserem **Block-Stacking-Task in Isaac Lab** — aber **Block-Stacking ist keine publizierte Basis-Fähigkeit**. Schlägt es fehl, wissen wir nicht, ob die Sim falsch ist oder das Basismodell den Task schlicht nicht kann. **Als sauberer Pipeline-Validator ungeeignet** (kein bekanntes Soll-Ergebnis).
4. **Embodiment-Distanz.** GR-1 (Fourier-Hände) ≠ G1+DEX3. Eine Isaac-Lab-Rekonstruktion eines GR-1-Tasks bräuchte ein **GR-1-USD-Asset** und eine eigene Szene — sie validiert dann die *Engine/Plumbing*, nicht 1:1 unsere DEX3-Assets.

**Empfohlene Referenzaufgabe:** **RoboCasa GR-1 Tabletop, Einzeltask `PnPNovelFromPlateToPlateSplitA` (zero-shot 78,7 %)** — höchste Zero-Shot-Quote, Humanoid + geschickte Hände (nächster Verwandter zu G1+DEX3), publizierte Zahl im Submodul vorhanden. **Fallback/einfacher:** **RoboCasa Panda** (ø 66 %, Parallelgreifer, aber sehr gut dokumentiert).

---

## 3. Implementierungsplan

Zweistufig: **Phase 1** validiert günstig die GR00T-Inferenz-Hälfte gegen einen Goldstandard. **Phase 2** (optional/Stretch) rekonstruiert einen base-fähigen Task in unserem Isaac Lab.

### Phase 0 — Vorbereitung (½ Tag)
- [ ] Submodul-Stand prüfen: `examples/robocasa/` und `examples/robocasa-gr1-tabletop-tasks/` sind bereits vorhanden ([bestätigt](../../app/Groot-1.6/examples/)).
- [ ] GPU mit RT-Cores ist für robosuite **nicht** zwingend (MuJoCo rendert ohne RTX) — Phase 1 kann auf **A100/H100 (KISSKI)** oder jeder vast.ai-GPU laufen, anders als unsere Isaac-Sim-Evals.
- [ ] Soll-Werte fixieren: Tabelle aus [robocasa-gr1-tabletop-tasks/README.md](../../app/Groot-1.6/examples/robocasa-gr1-tabletop-tasks/README.md) als Referenz speichern.

### Phase 1 — Goldstandard reproduzieren (1–2 Tage) ★ empfohlener Start
Ziel: Zeigen, dass **unser** Setup das **Basismodell** im Closed Loop fahren und die **publizierte ø 47,6 % / 78,7 %** reproduzieren kann. Validiert: Checkpoint-Laden, `--embodiment-tag GR1`, Sim-Policy-Wrapper, Obs/Action-Konvention, ZMQ-Rollout.

> **Erstimplementierung vorhanden** (Scaffold, Hardware-Lauf ausstehend): Orchestrierung
> [`Simulation/robocasa_reference/run_robocasa_ref_eval.sh`](../../Simulation/robocasa_reference/run_robocasa_ref_eval.sh),
> KISSKI-Job [`Simulation/kisski_robocasa_ref_submit.sh`](../../Simulation/kisski_robocasa_ref_submit.sh),
> Bedienung [`robocasa-referenz-eval.md`](robocasa-referenz-eval.md). Die manuellen Schritte unten sind dort gekapselt.

1. [ ] Eval-Umgebung einrichten (einmalig):
   ```bash
   bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
   ```
2. [ ] **Server** (Basismodell, GR1, kein Finetune):
   ```bash
   uv run python gr00t/eval/run_gr00t_server.py \
     --model-path nvidia/GR00T-N1.6-3B --embodiment-tag GR1 --use-sim-policy-wrapper
   ```
3. [ ] **Client/Rollout** auf dem Top-Task:
   ```bash
   gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
     --env-name gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
     --n-rollouts <N>   # genug für enge CIs, z. B. 100–200
   ```
4. [ ] **Akzeptanzkriterium:** gemessene Quote im **95-%-Konfidenzintervall** um 78,7 % (bzw. ø 47,6 % über alle 24 Tasks). Bei N=200 ist die Halbbreite ~±5–6 %.
5. [ ] Ergebnis dokumentieren in [docs/ergebnisse/](../ergebnisse/README.md) (neue `basismodell-referenz-eval.md`).

> **Optional 1b — RoboCasa Panda** als zweiter, unabhängiger Goldstandard (Parallelgreifer, ø 66 %): analog mit `--embodiment-tag` aus [robocasa/README.md](../../app/Groot-1.6/examples/robocasa/README.md). Höhere Quoten → klareres Signal, aber kein geschickter-Hand-Bezug.

**Aufwand:** gering, kein Eigencode, kein Asset-Bau. **Risiko:** robosuite-Setup-Skript / RoboCasa-Asset-Download kann zicken (Apt-Deps `libegl1-mesa-dev`, `libglu1-mesa`).

### Phase 2 — Referenztask in **unserem** Isaac Lab nachbauen (Stretch, 1–2 Wochen)
Nur falls explizit die **Isaac-Lab-Hälfte** validiert werden soll. Ziel: denselben GR-1-Pick-and-Place-Task (`PlateToPlate`) in Isaac Lab nachstellen und prüfen, ob die **Basismodell-Quote** dort ähnlich erreicht wird (Engine-Äquivalenz robosuite↔Isaac Lab).

1. [ ] **GR-1-USD-Asset** beschaffen/konvertieren (Fourier-GR1-URDF → USD via vorhandener [convert_urdf_to_usd.py](../../Simulation/g1_dex3_sim/convert_urdf_to_usd.py)-Logik).
2. [ ] Neue Isaac-Lab-Env analog [g1_dex3_blockstack_env.py](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py): Tisch + Plate-Quell/Ziel + Objekt, GR-1-Artikulation, Kameras passend zur GR1-Modality.
3. [ ] **GR1-Modality/Embodiment** im Server statt `NEW_EMBODIMENT` verdrahten; `build_obs` ([client.py:162](../../Simulation/g1_dex3_sim/client.py#L162)) auf GR1-State-Keys (Arme+Waist+Fourier-Hände) anpassen.
4. [ ] Erfolgskriterium dem RoboCasa-Task nachbilden (Objekt auf Ziel-Plate).
5. [ ] **Akzeptanzkriterium:** Quote ≈ robosuite-Referenz ± Sim-Gap-Toleranz. Differenz = Maß für den **Engine-/Asset-Domain-Gap** unserer Umgebung.

**Aufwand:** hoch (Asset + Szene + Modality-Verdrahtung). **Nutzen:** einziger Weg, die Isaac-Lab-Szene gegen einen externen Goldstandard zu prüfen. **Empfehlung:** erst angehen, wenn Phase 1 grün ist und der Bedarf an Isaac-Lab-Vollvalidierung bestätigt wurde.

### Was bewusst NICHT empfohlen wird
- **Basismodell auf unserem DEX3-Env (`NEW_EMBODIMENT`)** zero-shot testen → keine Basisgewichte, Aktions-Dim-Mismatch, kein Soll-Ergebnis.
- **Vorhandenen `entrypoint_baseline.sh` (UNITREE_G1 + Block-Stacking)** als Pipeline-Validator nehmen → Block-Stacking ist keine publizierte Basis-Fähigkeit; Fehlschlag ist nicht interpretierbar. (Als *qualitativer* Smoke-Test des Isaac-Lab-Plumbings weiterhin brauchbar, aber nicht als Goldstandard.)

---

## 4. Validierungskriterien (Zusammenfassung)
| Stufe | Was wird validiert | Soll-Wert | Aufwand |
|---|---|---|---|
| **Phase 1** | GR00T-Inferenz: Checkpoint, Embodiment-Tag, Wrapper, ZMQ-Rollout, Obs/Action | GR-1 PlateToPlate **78,7 %** / ø **47,6 %** (95-%-CI) | gering |
| 1b | dito, 2. Goldstandard | Panda ø **66,2 %** | gering |
| **Phase 2** | Isaac-Lab-Szene/Asset/Engine (Domain-Gap zu robosuite) | ≈ robosuite-Quote ± Toleranz | hoch |

## 5. Offene Punkte / Risiken
- **N1.6 vs. N1.5 für GR-1:** Die ø-47,6-%-Tabelle stammt aus dem Submodul-README und ist für `GR00T-N1.6-3B` angegeben; die ursprünglich publizierten 42 %/47 % betrafen N1.5. N1.6 sollte vergleichbar/besser sein — vor Akzeptanz die README-Quelle als verbindlich setzen.
- **RoboCasa-Setup-Reproduzierbarkeit:** Asset-Downloads/Apt-Deps können scheitern; Setup-Skript zuerst isoliert testen.
- **Stochastik:** Diffusion-Action-Head ist nicht-deterministisch → ausreichend Rollouts (N≥100) für enge CIs.
- **Phase-2-Asset:** Verfügbarkeit eines sauberen Fourier-GR1-URDF/USD ist noch unbestätigt.

## Quellen (extern)
- HF Modelcard: https://huggingface.co/nvidia/GR00T-N1.6-3B · `embodiment_id.json`: https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/embodiment_id.json
- Isaac-GR00T `n1.6-release`: https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release · RoboCasa-Beispiel: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/examples/robocasa/README.md
- RoboCasa GR-1 Tabletop: https://github.com/robocasa/robocasa-gr1-tabletop-tasks · Paper: https://arxiv.org/abs/2503.14734
- NVIDIA GEAR (GR00T N1.6): https://research.nvidia.com/labs/gear/gr00t-n1_6/
</content>
</invoke>
