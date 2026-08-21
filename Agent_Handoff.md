# Projekt-Handoff

## Sim-Eval: synchrones Kamerarendering

- Der Schnellpfad `CAMERA_RENDER_EVERY_N=EXECUTION_HORIZON` setzt den automatischen
  Renderer von `DirectRLEnv` aus und rendert direkt vor jeder Policy-Anfrage.
- In Isaac-Lab 3.0 aktualisiert `self.sim.render()` allein die Python-seitigen
  `TiledCamera`-RGB-Puffer nicht garantiert. Deshalb ruft
  `G1Dex3BlockstackEnv.force_policy_camera_render()` danach für jede Kamera
  `camera.update(0.0)` auf.
- Private Sensor-Zeitstempel sind versionsabhängig und dienen nur der Diagnose. Bleiben
  sie statisch, wird einmal gewarnt, aber die Eval nicht abgebrochen.

## Dataset-Replay-Videos mit einmalig gesetzten Würfeln

- Neuer Ablauf: `replay-prepare` → `replay-calibrate` → `replay-poses` → `replay-render`
  in `Simulation/server_rl_run.sh`; Bedienung unter
  `docs/simulation/replay-videos-aus-realdaten.md`.
- Aktueller Artefaktstand ist Version 4: `replay-calibrate` sammelt Pick-Anker aus
  Farbtracking, Bewegungsbeginn und sparsamer Direct-State-Fingerkuppen-FK. Es spielt keine
  Actions ab und lernt je Kopfkamera eine `pick_anchored_homography` von Pixel nach Tisch-XY.
- Kalibriert wird separat auf den ersten 40 Episoden. Ganze Episoden werden mit Seed 17 in
  80 % Fit und 20 % Holdout geteilt. Gates: mindestens 24/6 Fit-/Holdout-Anker aus
  mindestens 8/3 Episoden, 12 cm Arbeitsraumabdeckung, je Kamera höchstens 1,5/3 cm
  Holdout-Median/p90 und höchstens 2/3 cm Kameraabweichung im Holdout.
- `REPLAY_NUM_EPISODES`, `REPLAY_START_EPISODE` und `REPLAY_EPISODE_IDS` wählen nur die
  Ziel-Episoden für Posen und Replay. `replay-poses` schreibt
  `cube_poses.json` Version 4 (`stationary_top_face_homography`) aus stabilen Frames vor
  der ersten Bewegung. Es gibt keine FOV-Korrektur aus der AABB, keine 75/25-Fusion und
  keinen Greifpunkt-, Zufalls- oder Alt-Layout-Fallback. Der Einzelkamera-Fallback nutzt
  ausschließlich den Median aus mindestens fünf echten Top-Face-Detektionen.
- Die drei Würfel werden nach dem Roboter-Startzustand genau einmal aus `cube_poses.json`
  gesetzt. Danach gibt es weder Pose-Schreibzugriffe noch Attach/Tracking.
- `REPLAY_OUTPUT_MODE=videos` schreibt fünf Prüf-MP4s; `dataset` schreibt LeRobot v2.1
  mit vier Policy-Kameras, Sim-State und per SHA-256 geprüften Original-Actions.
- Das Replay-Manifest ist Version 2. Es misst Fingerkuppenabstände und Würfelhub
  ausschließlich lesend aus PhysX. Der erwartete erste Griff gilt als erfolgreich, wenn
  der zugeordnete Würfel mindestens 2 cm für fünf aufeinanderfolgende Frames angehoben ist.
  `calibration_sha256` und `poses_sha256` stehen auf Manifestebene;
  `pick_anchor_diagnostics` steht auf Ebene der jeweiligen `cube_poses`-Episode. Alte oder
  hash-inkompatible Ausgaben benötigen `REPLAY_OVERWRITE=1` oder ein neues Ziel.
- Der Renderer akzeptiert nur das exakte v4-Artefaktpaar mit passenden Methoden,
  Quelldatensatz und Kalibrierungs-Hash. `REPLAY_OVERWRITE=1` umgeht diese Eingabeprüfung
  nicht; es steuert nur die Neuberechnung der gewählten Posen beziehungsweise Renderausgaben.
- Der Replay überschreibt nur seine lokale Env-Config auf exakt 30 Hz und deaktiviert den
  Success-Auto-Reset. Eval/RL behalten ihre bisherigen Defaults.
- Die neue Homographie-, Renderer- und Griffmetrik braucht noch die Abnahme auf dem
  GPU-/Isaac-Server. Lokal sind nur reine Python-Tests, Syntax und Shell-Struktur prüfbar.
