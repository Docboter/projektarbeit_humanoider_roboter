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
- Default sind höchstens zehn Episoden. `REPLAY_EPISODE_IDS` überschreibt die fortlaufende
  Auswahl, `REPLAY_MAX_FRAMES=60` eignet sich für den Techniktest.
- `replay-calibrate` ist ein reiner CPU/CV-Schritt: Er liest höchstens 30 Frames der beiden
  Kopfkameras, prüft die Skala gegen die bekannte 5-cm-Würfelkante und hält den
  Würfelmittelpunkt fest auf z=0,915 m. Es gibt keine Bewegungsanker, Homographien oder
  vollständige Trajektoriensimulation während der Kalibrierung.
- `replay-poses` ergänzt die Stereo-CV-Schätzung standardmäßig mit
  `REPLAY_GRASP_SUPPORT=1`. Mögliche Schließintervalle kommen aus den aufgezeichneten
  Fingerzuständen; Isaac wertet nur wenige direkt gesetzte Zustände per Vorwärtskinematik
  aus. Eine eindeutige Handposition wird 75/25 mit dem nächsten CV-Würfel kombiniert.
  `REPLAY_GRASP_SUPPORT=0` erzeugt eine reine CV-Vergleichsdatei.
- Die drei Würfel werden nach dem Roboter-Startzustand genau einmal aus `cube_poses.json`
  gesetzt. Danach gibt es weder Pose-Schreibzugriffe noch Attach/Tracking.
- `REPLAY_OUTPUT_MODE=videos` schreibt fünf Prüf-MP4s; `dataset` schreibt LeRobot v2.1
  mit vier Policy-Kameras, Sim-State und per SHA-256 geprüften Original-Actions.
- Der Replay überschreibt nur seine lokale Env-Config auf exakt 30 Hz und deaktiviert den
  Success-Auto-Reset. Eval/RL behalten ihre bisherigen Defaults.
- Hardware-/Isaac-Abnahme der sparsamen FK-Stütze steht noch aus; lokal sind nur die
  reinen Python-Tests, Syntax und Shell-Struktur ausführbar.
