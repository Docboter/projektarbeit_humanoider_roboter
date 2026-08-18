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
