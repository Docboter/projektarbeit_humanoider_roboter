# Bewertung der Sim-Umsetzung — Sinnvoll & korrekt? (2026-06-05)

> **TL;DR:** Methodik-Review: Ist die Closed-Loop-Sim-Eval sinnvoll und korrekt aufgebaut? Klärt per
> Code-Verifikation und Literaturrecherche, warum 0 % Erfolg der erwartete Real→Sim-Gap ist und
> Open-Loop-MSE die valide Metrik bleibt.

> **Nachtrag 2026-10-11:** Die Review-Aussage „Steuerungspfad stimmt nachweislich" ist überholt: am
> 2026-08-24 wurde ein Sim-Fehler gefunden und behoben (falsche Vorzeichenspiegelung ließ vier
> proximale Fingergelenke stillstehen, Beckenhöhe 0,85→0,764 m, Basis x −0,057 m;
> [Chronik, Läufe 38–51](diagnose-chronik.md#läufe-3846-runs2026082302-runs202608240512-die-fingergrundgelenke-standen-still)).
> Die 0-%-Ergebnisse stammen aus dieser fehlerhaften Geometrie; „0 % = erwarteter Real→Sim-Gap"
> ist plausibel, aber nicht mehr sauber von diesem Fehler getrennt. Das SIMPLER-Beispiel in §1 war
> außerdem falsch wiedergegeben und ist korrigiert. Auch die Aussage zu `TUNE_VISUAL` in §2 ist
> durch Lauf 34 relativiert (Fingerspanne 27,6 % statt 20,5 % mit Augmentierung, `lifted` weiter 0/10).

> **Nachtrag 2026-10-13 — Messgrößen.** Seit 2026-10-13 meldet die Eval zusätzlich **`lifted_clear`**: Würfel-Unterseite > 2 cm über dem
> Tisch **und** im selben Schritt ein Fingerkuppen-Abstand ≤ 4 cm zur Würfelmitte (Proxy, keine
> Kontaktsensoren), je Episode `max_clear_lift_cm`. Das alte `lifted` (Schwerpunkt-Δz > 2 cm) bleibt
> zur Vergleichbarkeit, zählt aber Kippen mit (Kante ≈ 1,0 cm, Ecke ≈ 1,8 cm Schwerpunkthub).
> Zusammenfassung und JSON tragen 95-%-Clopper-Pearson-Intervalle (`milestone_ci95`; 0/10 ⇒ 0–31 %).
> Läufe vor dem 2026-10-13 zählten Server-Fehler als misslungene Episoden.
> Details: [sim-eval-anleitung.md, Schritt 7](../simulation/sim-eval-anleitung.md#schritt-7--ergebnisse-sichern).

Tiefenreview der Closed-Loop-Sim-Eval: Ist der aktuelle Aufbau methodisch sinnvoll und
technisch korrekt? Grundlage sind (a) eine Code-Verifikation gegen die echte GR00T-Trainings-
konfiguration und (b) eine belegte Literaturrecherche (Quellen jeweils verlinkt).

> **Kurz:** Die Sim ist **technisch korrekt gebaut**, aber als **Bewertungsinstrument für dieses
> Modell methodisch am Ziel vorbei**. Der Steuerungspfad stimmt nachweislich; das Problem ist,
> dass Closed-Loop-Sim für ein auf *Realdaten* trainiertes VLA mit *eingefrorenem* Vision-Encoder
> die falsche Messgröße ist. Das 0-%-Ergebnis ist der erwartete, nicht-aussagekräftige Effekt des
> Real→Sim-Gaps — kein Modell-Urteil.

---

## 1. Die zentrale Erkenntnis

**0 % Closed-Loop-Erfolg in einer nicht-photorealistischen Sim mit eingefrorenem Real-Bild-Encoder
ist das erwartete Ergebnis des Real→Sim-Gaps, nicht ein Urteil über das Modell.**

Kanonische Referenz: **SIMPLER (CoRL 2024)**, gebaut für Real→Sim-Eval von VLAs.

- Zahlen aus dem SIMPLER-Paper (Li et al. 2024), Aufgabe „Pick Coke Can": Octo-Base real
  **29,3 %**, in SIMPLER mit Visual Matching **17,0 %**, mit Variant Aggregation **0,6 %**;
  RT-1 (konvergiert) 85,7 % (VM) bzw. 89,8 % (VA). Der Gap ist also je nach Modell und
  Erscheinungsbild-Angleichung groß, aber „0 % ohne / 29,3 % mit Visual Matching" stimmt nicht
  (frühere Fassung dieses Dokuments).
- SIMPLERs empfohlene Methode ist **„Visual Matching"**: echte Hintergründe einblenden, reale
  Texturen auf Sim-Objekte projizieren, Roboter umfärben, Licht angleichen.
- **PreviewSurface-Quader (unser Setup) sind der Worst Case** für einen Real-Bild-Encoder.

Unsere eigene Diagnose („Versagen = Domain Gap, nicht die Sim", siehe
[umsetzungsnotizen.md §13](../simulation/umsetzungsnotizen.md)) ist damit **korrekt und durch
Primärliteratur gestützt**.

Quellen:
- SIMPLER — <https://arxiv.org/html/2405.05941v1>
- <https://github.com/simpler-env/SimplerEnv>

---

## 2. Konsequenz für die aktuelle Strategie (wichtigster Punkt)

Die Trainings-Doku framt **`TUNE_VISUAL`** (Vision-Encoder fine-tunen) als Domain-Gap-Fix.
**So funktioniert das nicht:**

> Einen Encoder gegen den Sim-Gap robust zu machen gelingt **nur, wenn er Sim-Bilder (oder
> DR-Bilder) im *Training* sieht.** Wir trainieren ausschließlich auf *echten* Teleop-Bildern und
> sehen Sim erst zur Eval-Zeit. `TUNE_VISUAL` auf Realdaten macht den Encoder **nicht** robust
> gegen Isaac-Sim-Renderings. (Stand 2026-10-11: Lauf 3/34 zeigt mit Color-Jitter-Augmentierung einen
> messbar positiven, aber unzureichenden Effekt — siehe Nachtrag oben.)

Aus demselben Grund ist **Domain Randomization zur Eval-Zeit (`dr_enabled=True`) konzeptionell
rückwärts**: DR ist eine *Trainings*-Technik. Auf eingefrorene, nie variiert trainierte Gewichte
angewandt, fügt sie nur Rauschen zu ohnehin schlechten Features hinzu — kein Nutzen.

Quellen:
- „Bridging the Sim2Real Gap: Vision Encoder Pre-Training" — <https://arxiv.org/pdf/2501.16389>
  (Domain-Invarianz ist eine *gelernte* Eigenschaft, kein Eval-Schalter)

---

## 3. Die richtige Bewertungsmethode (bereits vorhanden)

NVIDIAs **offizielle** Checkpoint-Eval ist **Open-Loop-Action-MSE auf gehaltenen echten
Trajektorien**, nicht Closed-Loop-Sim. Genau das ist im Repo vorhanden:
[`open_loop_eval.py`](../../app/Groot-1.6/gr00t/eval/open_loop_eval.py) +
[`kisski_open_loop_eval.sh`](../../Training/kisski_open_loop_eval.sh).

Empfohlene Eval-Hierarchie für unser Modell:

1. **Open-Loop-MSE** auf Held-out-Realdaten → primäre Modellqualität (kein Gap, sofort verfügbar).
2. **Real-Roboter-Rollouts** → Goldstandard (falls Hardware zugänglich).
3. **Sim-Rollouts nur mit SIMPLER-Visual-Matching** → sonst dominiert der Gap.

> Hinweis: Die GR00T-Doku nennt **keinen** MSE-Schwellwert für „gut" — MSE ist eine relative
> Vergleichsmetrik (Checkpoint vs. Checkpoint), kein Pass/Fail-Gate.

Quellen:
- <https://github.com/NVIDIA/Isaac-GR00T> (open_loop_eval / server-client)
- <https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning> (`eval_policy.py --plot`, MSE)
- <https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html>

---

## 4. Was die Recherche *entlastet* hat

| Punkt | Urteil aus der Literatur |
|---|---|
| **Execution-Horizon 8 von 16** | ✅ Lehrbuch-Receding-Horizon (exakt die Diffusion-Policy-Konvention). Nicht die Fehlerursache. |
| **Frequenz 28,57 vs 30 Hz** | ✅ ~5 %, für absolute Positions-Targets unkritisch, **nicht** die Erfolgsursache. Seit 2026-10-13 behoben (`dt = 1/210`, echte 30 Hz). |
| **Erste-Frame-/Overlay-Kamerakalibrierung** | ✅ Richtige *Richtung* (SIMPLER nutzt reale Referenz-Frames). Adressiert aber nur Geometrie/Blickwinkel, **nicht** Material-Realismus. |
| **Action absolut/relativ** | ✅ Korrekt: Arme `RELATIVE`, Hände `ABSOLUTE` laut Modality-Config, aber `decode_action(..., batched_states)` rechnet serverseitig alles in absolute Gelenkpositionen um. Das Env wendet zu Recht alle 28 Dims absolut an. Replay bestätigt (0,02 rad Tracking). |

Quellen:
- GR00T N1 (H=16) — <https://arxiv.org/html/2503.14734v1>
- Real-Time Execution of Action Chunking Flow Policies — <https://arxiv.org/html/2506.07339>
- Isaac Lab Replay-Frequenz-Toleranz — <https://github.com/isaac-sim/IsaacLab/issues/1833>

---

## 5. Konkrete Code-Befunde (Handlungsbedarf)

| Prio | Datei | Befund | Maßnahme |
|---|---|---|---|
| ✅ | [`g1_dex3_cfg.py`](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py) (Aktuator-Kommentar zu Armen/Händen) | Kommentar behauptete „Arme = relative Deltas, im Loop addiert" — Code wendet **alle 28 Dims absolut** an. Widersprach der korrekten `_pre_physics_step`-Docstring. | **Erledigt 2026-08-21:** Kommentar auf ABSOLUTE Targets korrigiert, plus die gemessenen Tracking-Werte als Beleg gegen die wiederkehrende „PD-Gains zu weich"-These. |
| 🟠 | [`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) (`dr_enabled`, Z. ~452) | `dr_enabled=True` bei Eval bringt für eingefrorenen Encoder nichts (s. §2). Unverändert offen: Default ist weiter `True` (`DR_ENABLED=0` schaltet ab). | Für Eval auf `False` (oder als Trainings-Augmentierung verschieben). |
| 🟠 | [`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) (`policy_hz`, `dt`, `decimation`, Z. ~405–419) | ~~`policy_hz=30.0`, real aber 28,57 Hz (`dt=1/200`, `decimation=7`)~~ — **behoben 2026-10-13:** `dt=1/210` → echte 30 Hz (Review §3.3). | erledigt |
| 🟡 | [`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) (`_check_success`, Z. ~1303) | Erfolg verlangt vollständige 3er-Säule (alle Paare xy<3 cm + Höhe>8 cm + stabil). | Gegen die echte Dataset-Task abgleichen (stapelt sie 2 oder 3?). |

---

## 6. Empfehlung

Die Sim als **qualitatives Demo-/Debug-Werkzeug** behalten — dafür ist sie sauber gebaut und die
Replay-Diagnose ist wertvoll. Modellqualität aber **über die Open-Loop-Eval belegen** und
Closed-Loop-Erfolgsraten **nicht** als Modell-Benchmark lesen. Wer Closed-Loop ernsthaft als Metrik
will, kommt an **SIMPLER-Visual-Matching** (reale Texturen/Hintergründe statt Quader) **oder** an
Sim-Bildern im *Trainingssatz* nicht vorbei.

---

## Quellenübersicht

- SIMPLER (Real→Sim-Eval, Visual Matching): <https://arxiv.org/html/2405.05941v1> ·
  <https://github.com/simpler-env/SimplerEnv>
- NVIDIA Isaac-GR00T (Open-Loop-Eval, Server/Client): <https://github.com/NVIDIA/Isaac-GR00T>
- GR00T SO-101-Tuning (eval_policy.py, MSE): <https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning>
- GR00T N1 Paper (H=16, frozen VLM): <https://arxiv.org/html/2503.14734v1>
- Vision-Encoder Sim2Real Pre-Training (DR ist Trainings-Technik): <https://arxiv.org/pdf/2501.16389>
- Action-Chunking-Horizont: <https://arxiv.org/html/2506.07339> · <https://arxiv.org/html/2505.21851v1>
- Frequenz-Toleranz (Isaac Lab Replay): <https://github.com/isaac-sim/IsaacLab/issues/1833>
