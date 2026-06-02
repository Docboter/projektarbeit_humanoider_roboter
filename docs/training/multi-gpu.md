# Multi-GPU-Training (bis zu 4× A100)

Status: **geplant / noch nicht umgesetzt.** Dieses Dokument hält fest, **warum** sich Multi-GPU
lohnt und **was konkret zu tun ist**. Der aktuelle Lauf nutzt 1× A100 (`NUM_GPUS=1`).

> **Kernbefund:** Die eigentliche verteilte Trainingslogik (DDP / DeepSpeed ZeRO, Gradient-Sync,
> Batch-Splitting, Rank-aware Logging/Checkpointing) ist im GR00T-Code **bereits vollständig
> implementiert**. Es fehlt nur die **Start-Verkabelung** (torchrun statt plain `python`) plus die
> SLURM-Ressourcen. Aufwand daher **gering** — Konfig-/Launcher-Ebene, kein Logik-Umbau.

---

## Vorteile von 4× A100

| Vorteil | Details |
|---|---|
| **~3,5× schnelleres Training** | Single-Node-A100 mit NVLink skaliert typisch zu 80–95 %. Der 175k-Schritte-Lauf (~30 h auf 1 GPU) sinkt auf grob **~8–10 h**. |
| **Größerer effektiver Batch** | Alternativ gleiche Walltime, aber 4× größerer Batch → stabilere Gradienten. `global_batch_size` wird über die GPUs aufgeteilt (`per_device = global // num_gpus`). |
| **Mehr Experimente pro Zeit** | Kürzere Iterationszyklen → schnelleres Hyperparameter-Tuning, mehr Läufe im 48-h-Walltime-Fenster. |
| **Optimizer-State-Sharding (DeepSpeed ZeRO-2)** | Der 13 GB große `optimizer.pt` wird über die GPUs verteilt → weniger VRAM-Druck pro GPU, Spielraum für größere Batches. |
| **Headroom für größere Modelle/Auflösungen** | Falls später ein größeres Backbone oder höhere Bildauflösung genutzt wird, ist die Infrastruktur bereits da. |

### Realistische Erwartung

- **Keine 4,0× Beschleunigung** — NCCL-Kommunikation kostet 5–20 %.
- Skalierung ist am besten, wenn die GPUs gut ausgelastet sind (genügend `per_device`-Batch +
  `dataloader_num_workers`).
- 4× A100 auf der `kisski`-Partition sind ggf. **nicht sofort verfügbar** (längere Queue als 1 GPU).

---

## ToDos (konkrete Änderungen)

### 1. Launcher auf `torchrun` umstellen — `Training/scripts/run_finetuning.sh`

Aktuell startet **ein** Python-Prozess; das bleibt immer Single-GPU, egal was `num_gpus` sagt
(torchrun setzt erst die `WORLD_SIZE`/`LOCAL_RANK`-Env, die `experiment.py` erwartet):

```bash
# vorher:
uv run --no-sync python "$GROOT_ROOT/gr00t/experiment/launch_finetune.py" ...

# nachher (rückwärtskompatibel — bei NUM_GPUS=1 bleibt es plain python):
if [[ "${NUM_GPUS:-1}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS")
else
  LAUNCH=(python)
fi
uv run --no-sync "${LAUNCH[@]}" "$GROOT_ROOT/gr00t/experiment/launch_finetune.py" ...
```

### 2. SLURM-Ressourcen anfordern — `Training/kisski_submit.sh`

```bash
#SBATCH -G A100:4        # statt A100:1
#SBATCH -c 64            # genug CPU-Kerne (dataloader_num_workers × 4 GPUs)
#SBATCH --mem=256G       # bereits gesetzt — passt
# ...
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"   # MUSS durch NUM_GPUS teilbar sein
```

> Assertion im Code: `global_batch_size % num_gpus == 0`
> ([experiment.py:37](../../app/Groot-1.6/gr00t/experiment/experiment.py)). Bei 4 GPUs muss
> `GLOBAL_BATCH_SIZE` ∈ {4, 8, 12, 16, 32, …} sein.

### 3. (Optional) DDP statt DeepSpeed erzwingen

- Default bei `num_gpus > 1`: **DeepSpeed ZeRO-2** (`use_ddp=False`).
- `use_ddp` ist **nicht** als CLI-Flag in `FinetuneConfig` exponiert. Für reines DDP (das 3B-Modell
  passt locker auf eine 80-GB-A100) müsste `use_ddp` einmalig durchgereicht werden — eine winzige
  Code-Ergänzung in `gr00t/configs/finetune_config.py` + `launch_finetune.py`.
- Empfehlung: zunächst beim **DeepSpeed-Default bleiben** (ist verdrahtet, `deepspeed==0.17.6` ist
  Dependency), DDP nur falls Probleme auftreten.

---

## Verifikation (hier steckt die eigentliche Zeit, nicht im Coden)

| Punkt | Warum prüfen | Priorität |
|---|---|---|
| **Smoke-Test** | Kurzer Lauf (z. B. `MAX_STEPS=50 NUM_GPUS=2`) — startet torchrun, laufen alle Ranks, sinkt der Loss? | hoch |
| **Checkpoint-Format** | DeepSpeed ZeRO-2 speichert Optimizer-State sharded; das finale `model.safetensors` sollte normal ladbar bleiben. Prüfen, dass Sim-Eval + `upload_checkpoint.py` das Layout weiterhin lesen. | hoch |
| **GPU-Auslastung** | `nvidia-smi` während des Laufs — alle 4 GPUs ausgelastet? Sonst Batch/Worker erhöhen. | mittel |
| **Effektiver Batch / LR** | Wird der globale Batch erhöht, ggf. Lernrate anpassen (Linear-/Sqrt-Scaling). | mittel |
| **Queue-Wartezeit** | 4× A100 bekommt man evtl. nicht sofort — `squeue`/`sinfo` checken. | niedrig |

---

## Was im Code schon fertig ist (Referenz)

- [experiment.py:107-112](../../app/Groot-1.6/gr00t/experiment/experiment.py) — `nccl`-Init bei
  `WORLD_SIZE > 1`, `LOCAL_RANK`-Handling, Device-Set.
- [experiment.py:181-188](../../app/Groot-1.6/gr00t/experiment/experiment.py) — DeepSpeed-Config
  bei `num_gpus > 1 and not use_ddp`; `per_device_batch = global_batch // num_gpus`.
- `gr00t/configs/training/training_config.py` — `use_ddp=False`, `deepspeed_stage=2`,
  `num_gpus=1` (Defaults).
- Rank-aware Logging/W&B/Checkpoints sind bereits `global_rank == 0`-geschützt.
- `pyproject.toml` — `deepspeed==0.17.6` als Dependency (x86_64-Linux → im A100-Image vorhanden).

> **Fazit:** geschätzt **1–2 fokussierte Stunden** inkl. Smoke-Test. Die schwierige Hälfte
> (verteiltes Training) ist bereits erledigt.

## Verwandte Dokumentation

- [trainingsverfahren.md](trainingsverfahren.md) — was für ein Training hier läuft
- [kisski-hpc.md](kisski-hpc.md) — SLURM-Job, Monitoring, KISSKI-Spezifika
- [env-vars.md](env-vars.md) — alle Env-Vars inkl. `NUM_GPUS`, `GLOBAL_BATCH_SIZE`
