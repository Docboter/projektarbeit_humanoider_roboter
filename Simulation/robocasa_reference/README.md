# robocasa_reference — Basismodell-Referenz-Eval

Isolierte Referenzaufgabe: das **un-finetunte** GR00T-N1.6-Basismodell auf dem
**RoboCasa GR-1 Tabletop** Benchmark (robosuite/MuJoCo, zero-shot) — zur Validierung
unserer GR00T-Inferenz-Pipeline gegen NVIDIAs publizierte Erfolgsquoten.

- **Orchestrierung:** [`run_robocasa_ref_eval.sh`](run_robocasa_ref_eval.sh) — Setup → Server → Client → `summary.json`.
- **KISSKI-Job:** [`../kisski_robocasa_ref_submit.sh`](../kisski_robocasa_ref_submit.sh).
- **Bedienung & Caveats:** [`docs/simulation/robocasa-referenz-eval.md`](../../docs/simulation/robocasa-referenz-eval.md)
- **Hintergrund & Plan:** [`docs/simulation/basismodell-referenzaufgabe.md`](../../docs/simulation/basismodell-referenzaufgabe.md)

Stört keine bestehende Implementierung: eigene Venv (`robocasa_uv`), eigener ZMQ-Port
(5757), eigene `RC_*`-Env-Vars, eigenes Ergebnisverzeichnis (`/data/robocasa_ref`).
Braucht **keine RT-Cores** (läuft auf A100/H100).
