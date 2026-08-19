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
- Die drei Würfel werden nach dem Roboter-Startzustand genau einmal aus `cube_poses.json`
  gesetzt. Danach gibt es weder Pose-Schreibzugriffe noch Attach/Tracking.
- Der Replay überschreibt nur seine lokale Env-Config auf exakt 30 Hz und deaktiviert den
  Success-Auto-Reset. Eval/RL behalten ihre bisherigen Defaults.
- Hardware-/Isaac-Abnahme des neuen Pfads steht noch aus; lokal wurden nur Syntax und
  Shell-Struktur geprüft.
