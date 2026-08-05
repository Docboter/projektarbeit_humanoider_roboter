# RoboCasa GR-1 Referenz-Eval — Bedienung

**Stand:** 2026-07-19 · **Status:** ✅ Validiert auf 2× RTX PRO 6000 — Aggregat-Mittel über 12 Tasks **47,7 % ≈ 47,8 %** erwartet. Ergebnis: [docs/ergebnisse/basismodell-referenz-eval.md](../ergebnisse/basismodell-referenz-eval.md). `full`-Lauf bei 12/24 Tasks manuell gestoppt; Re-Run offen.

Operative Anleitung für die **Referenzaufgabe** aus
[basismodell-referenzaufgabe.md](basismodell-referenzaufgabe.md): das **Basismodell**
`nvidia/GR00T-N1.6-3B` (zero-shot, ohne Finetuning) auf dem **RoboCasa GR-1 Tabletop**
Closed-Loop-Benchmark (robosuite/MuJoCo) laufen lassen und prüfen, ob unsere
GR00T-Inferenz-Pipeline die **publizierten Erfolgsquoten** reproduziert.

> **Was das validiert:** Checkpoint-Laden, `--embodiment-tag GR1`, Sim-Policy-Wrapper,
> ZMQ-Server/Client, Obs/Action-Mapping. **Was es nicht validiert:** unsere Isaac-Lab-Szene
> (anderer Simulator) — das wäre Phase 2 im Plan.

## Artefakte (alle additiv, ändern nichts Bestehendes)
| Datei | Zweck |
|---|---|
| [`Simulation/robocasa_reference/run_robocasa_ref_eval.sh`](../../Simulation/robocasa_reference/run_robocasa_ref_eval.sh) | Orchestrierung: Setup → Server (Hintergrund) → Client je Task → Ergebnis-JSON. Läuft in jedem schreibbaren GR00T-Container. |
| [`Simulation/server_robocasa_ref_run.sh`](../../Simulation/server_robocasa_ref_run.sh) | **Helferskript für generischen Docker-GPU-Server** (`preflight`/`setup`/`smoke`/`eval`/`videos`/`fix-flash-attn`/`shell`/`clean`). Automatisiert Pfad A, Server-Pfade als Defaults. Siehe **Pfad A2**. |
| [`Simulation/kisski_robocasa_ref_submit.sh`](../../Simulation/kisski_robocasa_ref_submit.sh) | SLURM-Job (KISSKI, Partition `kisski`/A100) — **Eval-only-Template**, Setup vorab. |

## Architektur (zwei Prozesse, eine Maschine)
```
GR00T-Venv (.venv)                         robocasa_uv-Venv (eigene, isoliert)
run_gr00t_server.py  ──ZMQ tcp:5757──►  rollout_policy.py  ──►  robosuite/MuJoCo (EGL)
  --model-path <base>                       --env_name gr1_unified/…Env
  --embodiment-tag GR1                       success rate: <float>  → /data/robocasa_ref/
  --use-sim-policy-wrapper
```
Die beiden Venvs sind getrennt (Torch-Versionen kollidieren sonst). Der Runner startet
den Server im Hintergrund, wartet auf den Port, fährt die Tasks und beendet den Server am Ende.

---

## Voraussetzungen
- **GPU:** beliebige NVIDIA-GPU mit EGL-Rendering — **keine RT-Cores nötig** (MuJoCo, nicht Isaac Sim). A100/H100/L40/4090 alle ok.
- **EGL:** im Trainings-Image bereits enthalten (`libegl1` + NVIDIA-EGL-ICD + `MUJOCO_GL=egl`, siehe [Training/Dockerfile](../../Training/Dockerfile)).
- **Setup-Phase braucht Internet** (git-Submodul `robosuite`/`robocasa-gr1-tabletop-tasks`, `uv pip`, HF-Assets) **und einen schreibbaren Image-Baum** (`git submodule update` schreibt nach `.git`).
- **HF-Token** (`HF_TOKEN`) für Asset-Download und — falls kein lokales Modell — für den Modell-Download.

---

## Pfad A — schreibbarer Docker-Container (empfohlen: vast.ai / lokale GPU)
Einfachster Weg: ein **langlebiger** Container (kein `--rm`), Setup einmal, dann Eval.

**1) Setup (einmalig, mit Internet):**
```bash
docker run -d --name groot-robocasa-ref --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN=hf_... \
  -v "$(pwd)/Simulation/robocasa_reference:/workspace/robocasa_reference" \
  --entrypoint bash lucam03/projekt-humanoider-roboter:latest \
  -lc "RC_SETUP_ONLY=1 bash /workspace/robocasa_reference/run_robocasa_ref_eval.sh"
docker logs -f groot-robocasa-ref     # Setup beobachten (~10–20 min)
```

**2) Eval (Top-Task, schnelle Validierung):**
```bash
docker exec -e RC_PRESET=top -e RC_N_EPISODES=100 \
  -e RC_MODEL_PATH=nvidia/GR00T-N1.6-3B -e HF_TOKEN=hf_... \
  groot-robocasa-ref \
  bash -lc "bash /workspace/robocasa_reference/run_robocasa_ref_eval.sh"
```
> `RC_MODEL_PATH=nvidia/GR00T-N1.6-3B` lädt das Basismodell von HF. Liegt es bereits
> lokal (z. B. `/data/models/GR00T-N1.6-3B`), diesen Pfad angeben — dann offline.

**Ergebnis:** `/data/robocasa_ref/summary-<ts>.json` (+ Server-/Client-Logs). Per
`docker cp groot-robocasa-ref:/data/robocasa_ref ./robocasa_ref` herausholen.

---

## Pfad A2 — Helferskript für generischen Docker-GPU-Server ★ empfohlen

[`Simulation/server_robocasa_ref_run.sh`](../../Simulation/server_robocasa_ref_run.sh) automatisiert Pfad A:
ein langlebiger „Workbench"-Container (`sleep infinity`), in dem Setup und Eval per `docker exec` laufen — so
bleiben Modell-Cache und `robocasa_uv`-Venv zwischen Läufen erhalten. Server-Pfade sind als Defaults verdrahtet
(`/home/lmuecke/project/data/RoboCasa → /data`, Image, `--gpus all`); alles per `RC_*`-Env überschreibbar.

```bash
./Simulation/server_robocasa_ref_run.sh preflight              # Torch + flash-attn auf GPU (Blackwell/sm_120)
HF_TOKEN=hf_... ./Simulation/server_robocasa_ref_run.sh setup  # einmalig (Internet)
HF_TOKEN=hf_... ./Simulation/server_robocasa_ref_run.sh smoke  # 1 Task/5 Ep. Kettentest
HF_TOKEN=hf_... ./Simulation/server_robocasa_ref_run.sh eval   # Default top, 100 Ep.
./Simulation/server_robocasa_ref_run.sh videos                 # Rollout-Videos ins externe /data holen
```

- **Blackwell / kein Rebuild:** Das Trainings-Image (`torch 2.7.1`/`cu128`) bringt sm_120-Kernels mit; auch
  `flash-attn 2.7.4.post1` läuft. `preflight` prüft beides. Nur falls flash-attn scheitert: `fix-flash-attn`
  installiert ein sm_120-Wheel nach (**kein** voller Rebuild).
- **Videos:** `rollout_policy.py` schreibt Rollout-Videos nach Container-`/tmp` (flüchtig). `eval`/`smoke` und die
  Aktion `videos` verschieben sie nach `/data/robocasa_ref/videos/` (extern, per `scp` abholbar). Nur abgeschlossene
  Episoden, gerendert alle `steps_per_render` Frames.
- **GPU-Auslastung:** Das Modell läuft auf `cuda:0`; bei zwei sichtbaren GPUs rendert MuJoCo/EGL auch auf GPU 1
  (harmlos). GPU 1 freihalten: `RC_GPUS='"device=0"'`.

> Validiert am 2026-07-19 auf 2× RTX PRO 6000 Blackwell — Ergebnis: [docs/ergebnisse/basismodell-referenz-eval.md](../ergebnisse/basismodell-referenz-eval.md).

---

## Pfad B — KISSKI (Apptainer + SLURM, Sekundär/Template)
KISSKI ist schwieriger (Image read-only, Compute-Nodes ggf. offline). Daher: **Setup
einmalig auf dem Login-Node** in einem schreibbaren Sandbox/`--writable-tmpfs`, danach
Eval-Job mit `RC_SKIP_SETUP=1`.

**1) Setup einmalig (Login-Node, Internet):**
```bash
module load apptainer
# Schreibbaren gr00t-Baum + Internet bereitstellen; robocasa_uv landet im VAST-Fork.
apptainer exec --nv --writable-tmpfs \
  --bind /mnt/vast-kisski/projects/kisski-humrob/data:/data \
  --bind $GROOT_FORK_DIR/gr00t:/app/Groot-1.6/gr00t \
  --bind $GROOT_FORK_DIR/external_dependencies:/app/Groot-1.6/external_dependencies \
  --bind $REPO_DIR/Simulation/robocasa_reference:/workspace/robocasa_reference \
  $SERVER_SIF bash -lc "RC_SETUP_ONLY=1 bash /workspace/robocasa_reference/run_robocasa_ref_eval.sh"
```
> ⚠️ **Klärungspunkt:** `git submodule update` im read-only Image kann scheitern (siehe
> §Klärung). Falls ja, das `robocasa-gr1-tabletop-tasks`-Repo vorab per `git clone` in
> `$GROOT_FORK_DIR/external_dependencies/` ablegen.

**2) Eval-Job:**
```bash
export RC_PRESET=top RC_N_EPISODES=100
sbatch Simulation/kisski_robocasa_ref_submit.sh
squeue -u $USER
```

---

## Env-Var-Referenz (alle Prefix `RC_`, kollisionsfrei)
| Variable | Default | Bedeutung |
|---|---|---|
| `RC_PRESET` | `top` | `top` (bester Einzeltask, 78,7 %), `smoke` (1 Task/5 Ep.), `full` (alle 24), `custom` |
| `RC_TASKS` | — | bei `custom`: env-Namen, space/komma-separiert |
| `RC_N_EPISODES` | `50` | Episoden je Task (für enge KIs ≥100) |
| `RC_N_ENVS` | `8` | parallele robosuite-Envs |
| `RC_N_ACTION_STEPS` | `8` | Action-Steps je Policy-Query |
| `RC_MAX_EPISODE_STEPS` | `720` | Max. Steps je Episode |
| `RC_MODEL_PATH` | `/data/models/GR00T-N1.6-3B` | lokal bevorzugt; sonst HF-ID `nvidia/GR00T-N1.6-3B` |
| `RC_EMBODIMENT_TAG` | `GR1` | Embodiment des Basismodells |
| `RC_PORT` | `5757` | ZMQ-Port (bewusst ≠ 5555 der bestehenden Sim) |
| `RC_SETUP_ONLY` | `0` | `1` = nur Setup, dann Ende |
| `RC_SKIP_SETUP` | `0` | `1` = Setup überspringen (Venv muss existieren) |
| `RC_RESULTS_DIR` | `/data/robocasa_ref` | Ergebnis-/Log-Verzeichnis |
| `RC_SERVER_WAIT_TIMEOUT` | `900` | s, Wartezeit auf Server-Port (Modell-Load) |

---

## Akzeptanzkriterien & Interpretation
| Task | Publiziert (zero-shot) |
|---|---|
| `…FromPlateToPlateSplitA` (Top, Default) | **78,7 %** |
| Mittel über alle 24 (`full`) | **47,6 %** |

- **Bestanden:** gemessene Quote liegt im **95-%-Konfidenzintervall** um den publizierten Wert. Bei `n_episodes=100` ist die Halbbreite ~±5–6 pp; bei `full` den Mittelwert über alle 24 Tasks vergleichen.
- Der Runner schreibt `delta_pp` (gemessen − erwartet) je Task in die Summary-JSON.
- **Deutlich darunter** → Fehler in unserer Inferenz-Hälfte (Embodiment-Tag, Wrapper, Obs/Action, Modell-Pfad). **Im Rahmen** → Inferenz-Pipeline validiert; Phase-2-Schritt (Isaac-Lab-Nachbau) kann angegangen werden.
- Stochastik: Diffusion-Head ist nicht-deterministisch → ausreichend Episoden; einzelne Prozent Abweichung sind normal.

## Ergebnis-JSON (`summary-<ts>.json`)
```json
{ "timestamp": "...", "model_path": "...", "embodiment_tag": "GR1", "preset": "top",
  "results": [ {"task": "...", "n_episodes": 100, "success_rate": 0.78,
                "measured_pct": 78.0, "expected_pct": 78.7, "delta_pp": "-0.7", "client_rc": 0} ] }
```

---

## Isolation — was bewusst NICHT verändert wird
- Keine Änderung an bestehenden Entrypoints, `run_finetuning*.sh`, Isaac-Lab-Sim-Code, `Training/Dockerfile`, `Dockerfile.vastai` oder bestehenden `kisski_*`-Skripten.
- Eigene, isolierte `robocasa_uv`-Venv — die GR00T-`.venv` bleibt unberührt.
- Eigener ZMQ-Port (`5757` statt `5555`), eigene Env-Vars (`RC_*`), eigenes Ergebnisverzeichnis (`/data/robocasa_ref`).
- Setup-Artefakte (robosuite-Submodul, Assets, Venv) liegen unter `gr00t/eval/sim/robocasa-gr1-tabletop-tasks/` (generierte Dateien, kein getrackter Quellcode).

## Bekannte Klärungspunkte / Risiken

**Geklärt am ersten Hardware-Lauf (2026-07-19, Docker-Server, 2× RTX PRO 6000):**
- EGL, robosuite-Setup und Modell-Download liefen im Trainings-Image ohne Zusatz-Deps durch (`libGLU` war kein Problem).
- Die Warnungen `No object files found for category 'book' in registry 'sketchfab'` und `distractor_obj … skip` sind
  **benign**: `book`/Distraktoren sind optionale Hintergrund-Objekte (nie Target); `download_tabletop_assets.py`
  provisioniert bewusst nur `sketchfab`+`lightwheel` (objaverse ist nicht Teil des offiziellen Setups, NVIDIA nutzte
  dasselbe). Kein Asset-Nachbau nötig; verzerrt die Quote allenfalls nach oben.

Verbleibend (v. a. KISSKI/Apptainer):
1. **`git submodule update` im read-only Apptainer-Image** (KISSKI): kann fehlschlagen. Mitigation: Setup in schreibbarem Docker (Pfad A) oder `--writable-tmpfs`/Sandbox bzw. Submodul vorab klonen.
2. **Compute-Node-Netzwerk auf KISSKI:** Setup braucht Internet; daher Setup auf dem Login-Node, Eval offline mit `RC_SKIP_SETUP=1`.
3. **Venv-Pfad-Konsistenz:** Die `robocasa_uv`-Venv hat absolute Pfade — Setup und Eval müssen denselben Container-Mount-Pfad (`/app/Groot-1.6/gr00t/…`) verwenden.
4. **`libGLU`** ist im Image ungeprüft; falls MuJoCo es verlangt → `MUJOCO_GL=osmesa` (CPU-Fallback, langsam).
5. **N1.6-vs-N1.5-Soll:** Die 47,6-%-/78,7-%-Werte stammen aus dem Submodul-README für N1.6 — als verbindliche Referenz nehmen.

Diese Punkte sind erst am realen Lauf abschließend zu klären (lokal keine GPU/Docker verfügbar).
