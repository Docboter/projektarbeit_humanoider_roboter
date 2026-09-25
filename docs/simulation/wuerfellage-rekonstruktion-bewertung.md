# Würfellage-Rekonstruktion — Bewertung der Implementierung

> **TL;DR:** Code-Audit von `wuerfellage-rekonstruktion.md` (Pfad B, `pick_anchored_homography`
> v4) gegen die tatsächliche Implementierung: Die Doku stimmt mit dem Code überein, aber es
> existieren zwei parallele Pfade nebeneinander, und der Dataset-Modus von Pfad B ist tot.
> Enthält Bewertung, priorisierte Empfehlungen und eine verifizierte Behauptungs-Tabelle — keine
> eigenen Messungen (die liefert wuerfellage-rekonstruktion-lauf35-befund.md).

**Stand:** 2026-08-21 · Branch `training-luca-IKR-IS6.0-wuerfel-rekon-matthias` · HEAD `d73c6aa`
**Gegenstand:** Vergleich von [wuerfellage-rekonstruktion.md](wuerfellage-rekonstruktion.md) (Verfahren
`pick_anchored_homography` v4) mit dem tatsächlichen Code, dem älteren Co-Training-Pfad und den
Ideen aus der Doku-Historie. Das Feature ist ein Teilschritt für den Co-Training-Datensatz
(Schritt 4, [co-training.md](../training/co-training.md)).

Geprüft wurden: `collect_replay_anchors.py`, `replay_calibration.py`, `reconstruct_cube_poses.py`,
`replay_grasp_metrics.py`, `run_dataset_replay_videos.py`, `render_cotrain_dataset.py`,
`extract_block_layout.py`, `grasp_pose_support.py`, `estimate_grasp_pose_support.py`
(Umgesetzt 2026-08-26: entfernt), `camera_geometry.py`, die zugehörigen Tests (40/40 bestanden), `server_rl_run.sh`,
`launch_cotrain.py`/`lib_split.sh` sowie alle Revisionen der Doku (`641ffc4` → `d73c6aa`).
Zeilenangaben beziehen sich auf HEAD.

---

## 0. Kernaussagen

1. **Die Doku beschreibt den Code korrekt.** Alle ~20 Zahlenwerte und Regeln aus
   wuerfellage-rekonstruktion.md (Onset 8 px/5 Frames, 12-Frame-Toleranz, 6 mm/2 mm, Arbeitsraum,
   RANSAC 2 cm, Split 80/20 Seed 17, alle fünf Gates, 3-cm-Stereo-Verwurf, Zielfenster, z = 0,915,
   Projektionsgate 6/5 px/10 px, Griffmetrik 2 cm/5 Frames, SHA-256, „keine Pose-Schreibung nach
   Frame 0") stehen exakt so im Code. Das ist für dieses Projekt nicht selbstverständlich.
2. **Es gibt heute zwei parallele Implementierungen**, und die Doku macht das nicht deutlich:
   - **Pfad A (operativ):** `server_rl_run.sh render` → `render_cotrain_dataset.py` mit
     `layout.json` aus `extract_block_layout.py` (Verfahren v2, Blob + Pinhole) **und weiterhin
     `scan.json`-Greifpunkt als Fallback** — genau das, was §6.1 der Doku verbietet.
   - **Pfad B (v4, neu):** `replay-calibrate` → `replay-poses` → `replay-render`. Code-komplett,
     lokal getestet, **auf dem GPU-Server noch nie gelaufen** (laut damaliger Übergabenotiz), und
3. **Der Dataset-Modus von Pfad B ist tot.** `REPLAY_OUTPUT_MODE=dataset` verlangt
   `wrist_calibration.dataset_ready` in der Kalibrierung
   ([run_dataset_replay_videos.py:542-547](../../Simulation/g1_dex3_sim/run_dataset_replay_videos.py#L542-L547));
   **kein Skript im Repo schreibt dieses Feld** — `run_calibrate` erzeugt es nicht
   ([reconstruct_cube_poses.py:257-290](../../Simulation/g1_dex3_sim/reconstruct_cube_poses.py#L257-L290)).
   Selbst ohne das Gate wären alle Episoden mit `wrist_projection_mismatch` verworfen, weil
   `wrist_observations` an den Blöcken nie befüllt wird. **Damit existiert heute kein Pfad, der
   v4-Posen in einen Co-Training-Datensatz bringt.**
4. Das v4-Design ist **methodisch deutlich besser** als v2 (gelernte, holdout-geprüfte Abbildung
   statt rekonstruierter Kamera; stationäres Messfenster statt Frame 0; harte Gates statt
   Fallback-Kette). Es ist aber **ein Validierungswerkzeug, kein Co-Training-Renderer** — beide
   Ziele (Physik-Treue vs. Bild-Treue) werden in den Dokumenten vermischt (siehe §5.1).
5. **Gesamturteil:** Architektur gut, Code-Qualität solide, Betriebsreife nicht gegeben,
   Co-Training-Nutzen offen. Vor jedem weiteren Umbau gehört zuerst der Abnahmelauf auf den
   Server — alle Gates sind Anforderungen, keine gemessenen Werte.

---

## 1. Was ist „der aktuelle Ansatz"? — Zwei Pfade

```text
Pfad A  (server_rl_run.sh render — der heutige Co-Training-Renderer)
  extract_block_layout.py      HSV-Blob (größte Komponente) in Frame 0 beider Kopfkameras
        │                      → Pinhole-Rückprojektion auf z = 0,915 (camera_geometry.py)
        │                      → Mittel beider Kameras + additiver Bias → layout.json
        ▼
  render_cotrain_dataset.py    place_cubes(): layout.json → scan.json-Greifpunkt → Zufall
                               (Fallback-Kette, render_cotrain_dataset.py:525-554)
                               → einmal setzen, PhysX, --stop-at-grasp (Default an)
                               → LeRobot v2.1 (4 Policy-Kameras), held-out per Formel

Pfad B  (replay-calibrate / replay-poses / replay-render — v4)
  collect_replay_anchors.py    Farbtracking + Bewegungsbeginn + sparsame Direct-State-FK
        ▼                      → replay_anchors.json (≤ 1 Anker je Farbe je Episode)
  reconstruct_cube_poses.py    calibrate: RANSAC-Homographie je Kopfkamera, 80/20-Holdout,
        │                      fünf Gates → geometry_calibration.json (nur wenn valid)
        │                      poses: stabile Top-Face-Pixel vor Onset → Homographie
        ▼                      → Stereo-Median / Einzelkamera → cube_poses.json
  run_dataset_replay_videos.py einmal setzen, Projektionsgate (Sim-Render vs. reales Pixel),
                               reines PhysX, Griffmetrik (nur lesend) → MP4s + Manifest v2
                               → Dataset-Modus: durch Wrist-Gate gesperrt (§0.3)
```

| | Pfad A (v2, operativ) | Pfad B (v4, neu) |
|---|---|---|
| Bildmessung | Frame 0, Vollsilhouette | Median über ≥ 5 stabile Frames vor dem Bewegungsbeginn, Top-Face-Pixel |
| Pixel → Tisch | analytisches Pinhole-Modell mit **rekonstruierter** (nicht kalibrierter) Kamerapose + Bias | **gelernte** projektive Abbildung aus realen Pick-Ereignissen, RANSAC, Holdout |
| Bezug zwischen Real- und Sim-Kamera | muss explizit stimmen (A2/A3 der alten Annahmenliste) | wird implizit absorbiert: Abbildung geht direkt von realem Pixel nach Sim-FK-Koordinate |
| Validierung | keine automatische; `layoutcheck` manuell | fünf Kalibrierungs-Gates, Stereo-Verwurf, Zielfenster, Projektionsgate, SHA-256-Kette |
| Fallbacks | layout → scan.json → Zufall (stumm) | keine — Episode wird `skipped` |
| Nach dem Setzen | reines PhysX, Schnitt am frühesten `close_step` | reines PhysX, volle Episode, Griffmetrik nur als Diagnose |
| Orientierung / z | Identität / 0,915 | Identität / 0,915 |
| Held-out | `int(total·0.8)`, eigene Formel | `int(total·0.8)`, eigene Formel |
| Ausgabe | LeRobot v2.1 — **läuft** | LeRobot v2.1 — **gesperrt** |
| Hardware-Stand | ein voller Lauf (60 Ep., 2026-08-17), Befund: Frames ab dem Griff falsch | kein Lauf |

**Geteilter Code:** Beide Pfade nutzen dieselbe Farbsegmentierung — `replay_calibration.py`
importiert `CUBE_COLORS`, `HSV_WINDOWS`, `color_mask`, `largest_blob`, `rgb_to_hsv` aus
`extract_block_layout.py`. Dieselben Konstanten (Kante 0,05 m, Mittelpunkt 0,915 m,
`cam_left_high`/`cam_right_high`) sind jedoch **je Pfad separat deklariert**.

**Nicht verdrahtet:** `estimate_grasp_pose_support.py` und `apply_grasp_support()` (die
75/25-Fusion aus Commit `c2afe2e`) ruft nichts mehr auf — toter Code mit eigener Testdatei.
(Umgesetzt 2026-08-26: entfernt.)
Nur `closure_candidates`/`fingertip_measurement` aus `grasp_pose_support.py` leben in Pfad B
weiter ([collect_replay_anchors.py:40](../../Simulation/g1_dex3_sim/collect_replay_anchors.py#L40)).

---

## 2. Doku ↔ Code: Abweichungen

Die Behauptungen stimmen (§0.1). Die Doku **verschweigt** aber einige Eigenschaften des Codes:

| # | Was der Code tut | Wo | Warum es zählt |
|---|---|---|---|
| D1 | Dataset-Modus hängt am nie erzeugten `wrist_calibration.dataset_ready`; Wrist-Gate (beide Wrist-Kameras, ≥ 2 Messungen, ≤ 10/20 px) braucht `wrist_observations`, die `poses` nicht schreibt | [run_dataset_replay_videos.py:311-320, 542-547, 761-763](../../Simulation/g1_dex3_sim/run_dataset_replay_videos.py#L542-L547) | nur in [replay-videos-aus-realdaten.md:121-125](replay-videos-aus-realdaten.md) erwähnt, in wuerfellage-rekonstruktion.md gar nicht; ohne Produzent ist es eine Sperre ohne Schlüssel |
| D2 | Im **Stereo-Pfad** der Posen-Stufe wird `stable_top_face_measurement(..., allow_full_blob=True)` aufgerufen; `world_xy_m` kann dann Top-Face- und Full-Blob-Messungen mischen. Nur der Einzelkamera-Fallback ist garantiert Top-Face-only | [reconstruct_cube_poses.py:364-367](../../Simulation/g1_dex3_sim/reconstruct_cube_poses.py#L364-L367), [replay_calibration.py:566-575](../../Simulation/g1_dex3_sim/replay_calibration.py#L566-L575) | §6.3 der Doku argumentiert genau gegen Silhouetten-Schwerpunkte; `confidence: "stereo"` klingt besser als `single_top_face`, kann aber die unsauberere Messung sein |
| D3 | Homographie: eigene DLT via SVD **ohne Hartley-Normalisierung**, eigener RANSAC (1000 Iterationen, Bestes = meiste Inlier, Gleichstand → Fehlersumme) | [replay_calibration.py:273-313](../../Simulation/g1_dex3_sim/replay_calibration.py#L273-L313) | Pixel (0–640) und Meter (0,2–0,5) in einer Matrix → schlecht konditioniert; der finale Fit auf allen Inliern minimiert algebraischen, nicht geometrischen Fehler |
| D4 | `observation.state` im Dataset-Modus ist der **simulierte erreichte** Gelenkzustand, nicht der reale | [run_dataset_replay_videos.py:362-375](../../Simulation/g1_dex3_sim/run_dataset_replay_videos.py#L362-L375) | für ein Modell mit relativen Aktionen (`action − state`) kodiert das den Sim-Tracking-Fehler in die Labels (§5.3) |
| D5 | Held-out-Grenze wird in `reconstruct_cube_poses.py:53-67` und `run_dataset_replay_videos.py:74-86` **neu berechnet**, `split.json`/`TRAIN_SPLIT_RATIO` werden nicht gelesen | s. links; Referenz [lib_split.sh:48](../../Training/scripts/lib_split.sh#L48) | stimmt heute zufällig überein; divergiert stumm, sobald jemand die Ratio nur auf einer Seite ändert |
| D6 | Kalibrierungsmenge = Episoden **0–39** („ab Episode 0"), angewandt auf beliebige Ziel-Episoden bis 239 | [collect_replay_anchors.py](../../Simulation/g1_dex3_sim/collect_replay_anchors.py) via `--num-episodes` | Holdout prüft nur Generalisierung **innerhalb** 0–39; Drift der Kamera/Tischlage über 301 Episoden (mehrere Aufnahmesitzungen?) bleibt unbemerkt |
| D7 | Griffmetrik ändert nichts am Status: Episoden mit `grasp_success=false` bekommen `status: "ok"` und landen im Datensatz | [run_dataset_replay_videos.py:806-817](../../Simulation/g1_dex3_sim/run_dataset_replay_videos.py#L806-L817) | laut Doku gewollt („verändert die Simulation nicht") — aber für Co-Training sind Frames nach einem Fehlgriff genau die falsch beschrifteten Paare, die co-training.md §3.2a als „schlimmer als fehlende" einstuft |
| D8 | Projektionsgate vergleicht die **Farbdetektion im Sim-Render** mit dem **realen** `top_uv` | [run_dataset_replay_videos.py:273-307](../../Simulation/g1_dex3_sim/run_dataset_replay_videos.py#L273-L307) | gut — aber es prüft Homographie ∘ Sim-Kamera ≈ Identität, also die **Kamerapose** der Sim, nicht die Würfelpose; ein systematischer Fehlschlag heißt „Sim-Kamera falsch", nicht „Pose falsch". Außerdem laufen die für Realbilder getunten HSV-Fenster auf Sim-Materialien |

Positiv geprüft (keine Abweichung): Direct-State-FK ruft nach `write_joint_state_to_sim` korrekt
`sim.forward()` + `robot.update()` ([collect_replay_anchors.py:75-84](../../Simulation/g1_dex3_sim/collect_replay_anchors.py#L75-L84));
env-lokaler Rahmen wird bei Kalibrierung (Abzug `env_origins`) und Platzierung (Addition) konsistent
verwendet; keine ungeschützten `np.median`-Aufrufe; keine `except: pass`; keine Reste von 75/25,
AABB-FOV oder `--stop-at-grasp` in Pfad B.

---

## 3. Gemeinsamkeiten und Unterschiede zu den Ideen der Doku-Historie

Die Revision `641ffc4` (2026-08-18) enthielt eine Annahmenliste A1–A7, einen offenen Abnahmetest
und sechs „Ansatzpunkte zum Weiterbauen". `d73c6aa` hat all das gestrichen. Was daraus wurde:

| Idee (641ffc4 §8) | Schicksal in v4 |
|---|---|
| 8.2a Nur die Oberseite segmentieren, Ebene z = 0,94 statt 0,915 | **übernommen** — Kern von `stationary_top_face_homography` (Ebene wird durch die gelernte Abbildung implizit) |
| 8.2b Bekannte 5-cm-Box in die Silhouette fitten (x, y, optional Yaw) | **nicht gebaut**; Orientierung bleibt Identität |
| 8.3 Kamerapose per PnP gegen vermessene Tischgeometrie kalibrieren (A2) | **umgangen** statt gelöst — die Homographie ersetzt die explizite Kamera; das Projektionsgate (D8) ist der indirekte A2/A3-Test |
| 8.4 Verdeckung in Frame 0 / erste vollständig sichtbare Frames (A7) | **teilweise** — stabiles Messfenster bis zum Onset plus Einzelkamera-Fallback |
| 8.5 Greif-Intervalle + **kinematisches Attach** („der eigentliche nächste Schritt für den Renderer") | **bewusst verworfen** (§5 der Doku: „kein Tracking und kein kinematisches Attach") — co-training.md §3.2a nennt es weiterhin den richtigen nächsten Schritt → Widerspruch |
| 8.6 Wrist-Kameras einbeziehen | **halb** — nur als (nicht befülltes) Abnahmegate, nicht als Messquelle |
| Annahme A5: Tischhöhe/2-cm-Schwebe-„Kuriosum" | Fakt unverändert, Hinweis gestrichen; Doku nennt Tischoberkante 0,87 (alt) vs. Code-Kommentar 0,89 ([g1_dex3_blockstack_env.py:422](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L422)) — klären |
| Orientierung offen (§9) | weiterhin offen, aber nicht mehr als offen markiert |

**Nie irgendwo erwogen** (weder in einer Revision noch in co-training.md/umsetzungsnotizen.md):
Stereo-Triangulation (die „Stereobasis" dient nur als Plausibilitätscheck), Marker/ArUco, gelernte
Detektoren, Render-and-Compare/ICP, Kalibrierung je Episode oder je Sitzung, **Place-Position als
zweiter Anker**, mehrere Anker je Farbe und Episode (v4 nimmt nur den ersten Onset je Farbe,
[collect_replay_anchors.py:221-241](../../Simulation/g1_dex3_sim/collect_replay_anchors.py#L221-L241)).

**Einzige v3-Messung** („AABB-FOV + 75/25"): die 3,68-cm-Kantenschätzung — ohne Datum, Episoden,
Kontext. Sie lebt nur als ein Satz in §3 der HEAD-Doku.

---

## 4. Bewertung

| Dimension | Urteil | Begründung |
|---|---|---|
| Methodik v4 | **gut** | gelernte Pixel→Tisch-Abbildung mit Holdout ist der richtige Schritt weg von der rekonstruierten Kamera; Top-Face-Pixel konsistent in Kalibrierung *und* Anwendung → systematischer Bias hebt sich auf (v2 brauchte dafür einen additiven Bias); Anker nur bei eindeutiger Hand und plausiblem Arbeitsraum |
| Robustheit / Fail-safe | **gut** | keine stummen Fallbacks; Artefakt-Versionen, Methoden, SHA-256-Kette; Overwrite umgeht keine Eingabeprüfung; Aktionen werden nach der Verarbeitung auf Unverändertheit geprüft |
| Code-Qualität | **mittel–gut** | lesbar, klein geschnitten, 40 Unit-Tests grün. Aber: hausgemachte DLT ohne Normalisierung (D3), duplizierte Konstanten und Split-Formel (D5), ein totes Modul, zwei unabhängige „Wann schließt die Hand"-Algorithmen (`find_grasp_points` vs. `closure_candidates`), Tracker nur auf synthetischen Rechtecken getestet, `test_reconstruct_cube_poses.py` testet keine Funktion seines Namensgebers (umbenannt 2026-08-26 in `test_replay_calibration.py`, da real `replay_calibration.py`/`camera_geometry.py` getestet werden) |
| Betriebsreife | **schwach** | nie auf Isaac/GPU gelaufen; Dataset-Modus gesperrt (D1); kein Eintrag in der Diagnose-Chronik; alle Schwellwerte sind Soll-, keine Ist-Werte |
| Co-Training-Tauglichkeit | **offen** | Pfad B kann heute keinen Datensatz schreiben; Pfad A schreibt einen mit v2-Posen und verbotenem Fallback. Ob v4-Posen die Griffquote im Replay (v2-Lauf: 101/116 Griffe mit > 6 cm Kuppenöffnung) verbessern, ist unbekannt |
| Dokumentation | **mittel** | wuerfellage-rekonstruktion.md ist präzise, aber: CLAUDE.md, co-training.md, docs/README.md und docs/simulation/README.md beschreiben noch v2 als aktuell; CLAUDE.md kennt die v4-Dateien nicht; `historie.md:710` verweist auf §2.1 (jetzt §6.1) |

**Einordnung:** v4 ist das erste Verfahren im Projekt, das seine Kalibrierung *gegen sich selbst
prüft* und bei Misserfolg keine Datei schreibt. Das ist die richtige Kultur. Der Preis ist, dass es
derzeit nur Videos produzieren kann, und dass seine Qualität nur im Rahmen derselben Messmethode
(Farbtracker auf beiden Seiten des Holdouts) belegt ist — eine unabhängige Wahrheit (Handannotation)
fehlt.

---

## 5. Was vermutlich vergessen wurde

### 5.1 Zwei Ziele, ein Renderer

Die Dokumente behandeln „Replay mit korrekter Würfellage" als ein Ziel. Es sind zwei, und sie
widersprechen sich in einem Punkt:

| Ziel | Was das Bild zeigen muss | Was die Physik tun darf |
|---|---|---|
| **Physik-Validierung** (v4-Manifest, Griffmetrik): Greift der Sim-Roboter unter Originalaktionen den Würfel dort, wo er real lag? | egal | reines PhysX, keine Eingriffe — genau v4 |
| **Co-Training-Bild** (Schritt 4): Bild zu `action[t]` muss zeigen, was die reale Kamera bei `action[t]` sah | Roboter **in der realen Pose** (`observation.state[t]`), Würfel in der Hand, sobald er real in der Hand war | darf kinematisch geführt sein; ein Attach zwischen Pick und Release ist hier kein Makel, sondern Bildtreue |

v4 hat das Attach „bewusst verworfen" — richtig für Ziel 1, falsch für Ziel 2. co-training.md hält
es weiter für den richtigen Schritt — richtig für Ziel 2. Beide haben recht; das Feature braucht
einen Schalter (`REPLAY_ROBOT_DRIVE=action|state`, `REPLAY_CUBE_MODE=physx|attach`) statt einer
Entscheidung.

### 5.2 Der Anker ist nicht der Würfelmittelpunkt

Der Fingerkuppen-Schwerpunkt beim Griff liegt hand-abhängig versetzt zum Würfelmittelpunkt
(Daumen vs. zwei Finger; links/rechts gespiegelt). Eine Homographie je Kamera mittelt beide Hände
— der Versatz erscheint als Rauschen im Holdout statt als erkennbarer Bias. `hand` steht im Anker;
die Residuen **je Hand** auszuwerten kostet nichts. Ebenso: das z-Fenster 0,82–1,05 m ist 23 cm breit
bei 5 cm Würfelkante — die z-Verteilung der akzeptierten Anker gehört in den Report.

### 5.3 Welcher State gehört in den gerenderten Datensatz?

Pfad A schrieb `observation.state` = erreicht und zusätzlich `observation.state_real` (undeklariert);
Pfad B schreibt nur „erreicht". Bei relativer Aktionskodierung wird der PD-Tracking-Fehler der Sim
zum Label-Rauschen. Erst messen (erreicht vs. real, je Gelenk, als Verteilung ins Manifest), dann
entscheiden — und bis dahin beides schreiben.

### 5.4 Tracking-Fehler ist die andere Hälfte des Problems

Die Kalibrierung misst Fingerkuppen per **Direct-State** (ideal), das Replay fährt per **PD-Regler
auf Aktionen** (mit Nachlauf). Selbst bei perfekter Würfelpose greift die Sim-Hand daneben, wenn der
Arm 2 cm hinterherläuft. Der v2-Befund (101/116 Griffe > 6 cm Öffnung) ist wahrscheinlich
überwiegend dieses Problem, nicht die Würfellage. Die Griffmetrik von v4 wird das sichtbar machen —
aber nur, wenn man die Fingerkuppen-Distanz zum Würfel **vor** dem erwarteten Pick mitloggt.

### 5.5 Orientierung

Reale Würfel liegen erkennbar verdreht. Ein `minAreaRect`/PCA auf der Top-Face-Maske liefert Yaw
mod 90° aus denselben Pixeln, die v4 ohnehin misst. Einfluss: Fingerkontakt (Kante vs. Fläche) und
Bildtreue. Seit `641ffc4` offen, in HEAD nicht mehr als offen markiert.

### 5.6 Kalibrierungsmenge und Drift

301 Episoden, Kalibrierung auf 0–39. Wenn die Aufnahmen über mehrere Sitzungen gingen (Tisch
verschoben, Kamera neu montiert), ist die Abbildung ab Episode 40 möglicherweise falsch, und
niemand merkt es — das Projektionsgate (D8) würde dann zwar anschlagen, aber mit der falschen
Diagnose („Sim-Kamera"). Residuum gegen Episodenindex plotten; Kalibrierungsepisoden über den
Datensatz streuen.

### 5.7 Unabhängige Wahrheit

Holdout-Fehler ≤ 1,5 cm beweist, dass Tracker + Homographie **in sich** konsistent sind. Ob der
Tracker das Richtige misst, sagt er nicht. 20–30 Frames mit per Hand geklicktem Würfelmittelpunkt
(beide Kameras) sind ein Nachmittag Arbeit und geben: Tracker-Fehler in px, Homographie-Fehler in cm
unabhängig vom Tracker, und einen Regressionstest auf echten Bildern.

### 5.8 Was noch fehlt

- **Erfolgskriterium für das Teilfeature.** Wann ist die Rekonstruktion „gut genug" für
  Co-Training? Vorschlag: Holdout ≤ 1,5 cm (hat v4), `grasp_success` ≥ 70 % auf den Ziel-Episoden,
  Sichtprüfung Overlay real/sim bei Frame 0 und am Pick für 10 Episoden, Domain-Gap-Metrik
  (`measure_domain_gap.py`) gegen den v2-Datensatz.
- **Fehlgriffe im Datensatz** (D7): Wenn `grasp_success=false`, Episode ab dem erwarteten Pick
  abschneiden (Onset aus dem Manifest — das ist der saubere Ersatz für das alte
  `--stop-at-grasp` aus `scan.json`).
- **Spawn-Band des Eval-/RL-Envs** ist `x 0,30–0,40, y −0,20–0,20`
  ([g1_dex3_blockstack_env.py:420-421](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L420-L421));
  v4 akzeptiert `0,25–0,45 / −0,25–0,25`. Die rekonstruierten Realposen sind die **empirische
  Verteilung**, aus der das Env-Band kommen sollte — ein Gratis-Beitrag gegen den Domain-Gap aus
  Lauf 32/34.
- **Gestapelter Start / Verdeckung.** Ein Würfel, der bereits auf einem anderen liegt, hat seine
  Oberseite auf z ≈ 0,99 — die Homographie gilt nur für die Tischebene. Prüfen, ob solche Episoden
  im Datensatz vorkommen; sonst mindestens die Annahme dokumentieren.
- **Domain Randomization.** `replay-render` läuft mit `DR_ENABLED=0`; das Projektionsgate mit
  HSV-Fenstern verträgt keine Farbrandomisierung. Reihenfolge festlegen: Gate ohne DR, Rendern mit DR.
- **Provenienz.** `calibration_sha256` sichert die Datei, nicht ihre Entstehung: Git-Commit,
  USD-Hash und `_reach_offsets` der Fingerkuppen gehören in die Kalibrierungs-Payload.
- **Held-out** (D5): `split.json` lesen statt Formel duplizieren; bei `TRAIN_TEST_SPLIT=0` ist die
  Sperre der letzten 20 % konservativ, aber unbegründet.
- **Doku-Drift:** CLAUDE.md (Dateiliste, „The right source for render"), co-training.md §3.0/§3.2a,
  docs/README.md:35, docs/simulation/README.md:15, historie.md:710.

---

## 6. Empfehlungen (priorisiert)

### P0 — bevor irgendetwas weitergebaut wird

1. **Abnahmelauf auf dem Server:** `replay-calibrate` auf 40 Episoden, `calibration_report.json`
   lesen, Zahlen (Anker je Episode, Fit/Holdout-Median/p90 je Kamera, Kameradifferenz, z-Verteilung
   der Anker, `diagnostic_observed_edge_median_m`) in die Diagnose-Chronik. Fällt ein Gate, ist das
   der erste echte Befund zu v4.
2. **Dataset-Modus entsperren** (D1) — Entscheidung: (a) Wrist-Produzent bauen (reale Wrist-Frames
   an den stationären Frames auswerten, `wrist_observations` in `poses` schreiben,
   `wrist_calibration` in `calibrate`), oder (b) Wrist-Gate auf Diagnose zurückstufen und den
   Dataset-Modus nur am Kopfkamera-Gate hängen. Empfehlung: **(b) jetzt, (a) später** — das
   Kopfkamera-Gate ist die Messung, die die Pose bestätigt; Wrist-Kameras sehen den Würfel im
   Stationärfenster oft gar nicht.
3. **Pfad A absichern:** `scan.json`-Fallback in `place_cubes` (render_cotrain_dataset.py:532-540)
   zu einem harten Fehler machen, solange §6.1 der Doku gilt. Oder Pfad A als „legacy" markieren
   und `render` auf Pfad B umhängen, sobald 2. erledigt ist.

### P1 — Qualität der Posen

4. **Hartley-Normalisierung** in `fit_homography` (D3), optional Gauß-Newton auf dem geometrischen
   Fehler der Inlier. Zehn Zeilen; verändert die Holdout-Zahlen messbar.
5. **`allow_full_blob=False` im Stereo-Pfad** oder `top_face_world_xy_m` auch dort verwenden (D2);
   `confidence` entsprechend ehrlich setzen.
6. **Residuen je Hand, je Kamera, je Episode** in den Report; z-Fenster aus der Verteilung ableiten
   (§5.2); Kalibrierungsepisoden über 0–239 streuen (§5.6).
7. **Handannotierte Referenz** (§5.7) als Fixture + Test.
8. **Yaw aus der Top-Face-Maske** (§5.5) — `yaw_rad` und `orientation_wxyz` sind im Schema schon da.
9. `split.json` lesen (D5); Konstanten (Kante, z, Kameras, Arbeitsraum) in ein Modul ziehen.

### P2 — Co-Training-Nutzen

10. **Zwei Betriebsarten** im Renderer (§5.1): `action|state`-Antrieb und `physx|attach` für den
    Würfel. Co-Training-Datensätze mit `state` + `attach`, Validierung mit `action` + `physx`.
11. **Tracking-Fehler messen** und **beide States schreiben** (§5.3, §5.4).
12. **Fehlgriff-Schnitt** aus dem Manifest (§5.8).
13. **Env-Spawn-Band** aus `cube_poses.json` ableiten (§5.8).
14. **Aufräumen:** `estimate_grasp_pose_support.py` + `apply_grasp_support` entfernen oder
    begründen (Umgesetzt 2026-08-26: entfernt); `find_grasp_points` durch `closure_candidates` ersetzen; Doku-Drift (§5.8);
    wuerfellage-rekonstruktion.md um D1/D7/D8 und §5.1 ergänzen.

---

## 7. Offene Fragen ans Team

1. War das Wrist-Gate eine bewusste Sperre („Dataset erst nach Wrist-Kalibrierung") oder ein
   unvollständiger Commit (`c2afe2e` am selben Tag wie `d73c6aa`)?
2. Soll Pfad A (render) weiterleben, oder ersetzt `replay-render` ihn, sobald der Dataset-Modus
   frei ist?
3. Wurde der Realdatensatz in einer Sitzung aufgenommen? (Entscheidet über §5.6.)
4. Welches Verhältnis aus Bildtreue und Physik-Ehrlichkeit will der Co-Training-Datensatz? (§5.1)
5. Welcher State (real/erreicht) ist für GR00T mit relativer Aktionskodierung der richtige? (§5.3)

---

## Anhang A — Befundliste mit Schwere

| Schwere | Befund | Ort |
|---|---|---|
| HOCH | Dataset-Modus durch unerzeugbares `wrist_calibration.dataset_ready` und leere `wrist_observations` dauerhaft gesperrt | run_dataset_replay_videos.py:311-320, 542-547, 761-763 |
| HOCH | Pfad A verwendet den in §6.1 verbotenen `scan.json`-Greifpunkt weiterhin als Fallback, nur mit Warnung | render_cotrain_dataset.py:532-540 |
| MITTEL | Stereo-Pfad mischt Top-Face- und Full-Blob-Messungen | reconstruct_cube_poses.py:364-367 |
| MITTEL | DLT ohne Normalisierung, algebraischer Fehler im Endfit | replay_calibration.py:273-287 |
| MITTEL | Held-out-Formel dreimal dupliziert, `split.json` ungenutzt | reconstruct_cube_poses.py:53-67, run_dataset_replay_videos.py:74-86, render_cotrain_dataset.py:233-256 |
| MITTEL | Kalibrierung nur auf Episoden 0–39, keine Drift-Prüfung | collect_replay_anchors.py (`--num-episodes`) |
| MITTEL | Fünf verschiedene Arbeitsraum-Fenster ohne gemeinsame Konstante (A: 3, B: 2) | render_cotrain_dataset.py:527/536/547; collect_replay_anchors.py:136; replay_calibration.py:576 |
| MITTEL | `grasp_success=false` → trotzdem `status: ok` im Datensatz | run_dataset_replay_videos.py:806-817 |
| NIEDRIG | `estimate_grasp_pose_support.py`/`apply_grasp_support` tot, aber mit Test (Umgesetzt 2026-08-26: entfernt) | Simulation/g1_dex3_sim/ |
| NIEDRIG | `test_reconstruct_cube_poses.py` testet keine Funktion aus `reconstruct_cube_poses.py`; Tracker nie auf Realbildern getestet (umbenannt 2026-08-26 in `test_replay_calibration.py`, da real `replay_calibration.py`/`camera_geometry.py` getestet werden) | Simulation/g1_dex3_sim/test_*.py |
| NIEDRIG | `count_frames()` dekodiert jedes Video doppelt (nach dem Schreiben und in `validate_dataset`) | run_dataset_replay_videos.py:789-790, 503-504 |
| NIEDRIG | RANSAC-Warnungen `divide by zero` ungefiltert im Log (harmlos, wird per `isfinite` gefangen) | replay_calibration.py:270, 307 |
| NIEDRIG | Tischoberkante 0,87 (alte Doku) vs. 0,89 (Env-Kommentar) | g1_dex3_blockstack_env.py:422 |

## Anhang B — Verifizierte Behauptungs-Tabelle (Doku → Code)

| Behauptung | Code | Status |
|---|---|---|
| Onset ≥ 8 px über 5 Frames | replay_calibration.py:137 | ✓ |
| Kamera-Onsets ≤ 12 Frames | collect_replay_anchors.py:24, 223 | ✓ |
| ≥ 5 stabile Messungen, > 12 px zum Median verworfen | replay_calibration.py:166, 170 | ✓ |
| Hand schließt ≥ 6 mm, ≥ 2 mm mehr als die andere | replay_calibration.py:541 | ✓ |
| Arbeitsraum x 0,20–0,50 / y −0,30–0,30 / z 0,82–1,05 | collect_replay_anchors.py:136 | ✓ |
| FK nur an Intervallgrenzen und Onset −2/0/+2 | collect_replay_anchors.py:99-135 | ✓ |
| RANSAC-Schwelle 2 cm | replay_calibration.py:291 | ✓ |
| 80/20 auf ganzen Episoden, Seed 17 | replay_calibration.py:253-263, 316-321 | ✓ |
| Gates 24/8, 6/3, 12 cm, 1,5/3 cm, 2/3 cm | replay_calibration.py:326-377 | ✓ |
| Stereo-Median, > 3 cm verworfen | replay_calibration.py:554, 563-564 | ✓ |
| Einzelkamera ≥ 5 Top-Face-Detektionen, Top-Face-Median | replay_calibration.py:566-575 | ✓ |
| Zielfenster x 0,25–0,45 / y −0,25–0,25 | replay_calibration.py:576 | ✓ |
| z = 0,915, Orientierung Identität | reconstruct_cube_poses.py:43, 430 | ✓ |
| Projektionsgate 6 Messungen, ≤ 5 px Median, ≤ 10 px p90 | run_dataset_replay_videos.py:680-682 | ✓ |
| Griff: ≥ 2 cm Hub für 5 Frames | replay_grasp_metrics.py:170-171 | ✓ |
| SHA-256 für Kalibrierung, Posen, Aktionen | replay_grasp_metrics.py:15-47; run_dataset_replay_videos.py:536-541, 599-606, 800-802 | ✓ |
| Keine Pose-Schreibung nach Frame 0 | run_dataset_replay_videos.py:672-677 | ✓ |
| Kein 75/25, kein Greifpunkt-Fallback, keine AABB-FOV-Korrektur | grep über Pfad B | ✓ (nur Diagnosehinweis reconstruct_cube_poses.py:289) |
| Wrist-Gate / `dataset_ready` | run_dataset_replay_videos.py:542-547 | **nicht in der Doku** |
