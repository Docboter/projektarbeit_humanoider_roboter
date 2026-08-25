#!/usr/bin/env python
# TL;DR: Trainingseinstieg für zwei Datensätze (Mix-Ratio); von run_finetuning_cotrain.sh gestartet.
"""
launch_cotrain.py — Trainings-Einstieg für ZWEI Datensätze (echt + gerendert).

Warum dieses Skript überhaupt existiert
───────────────────────────────────────
Die Misch-Fähigkeit steckt bereits im Fork: ``DataConfig.datasets`` ist eine Liste von
``SingleDatasetConfig``, jede mit eigenem ``mix_ratio``; ``DatasetFactory.build`` baut
daraus eine ``ShardedMixtureDataset`` mit normierten Sampling-Gewichten. Nur der
CLI-Einstieg ``gr00t/experiment/launch_finetune.py`` schreibt genau EINEN Eintrag mit
``mix_ratio: 1.0`` fest.

Diese Datei ist deshalb eine Kopie von ``launch_finetune.py`` mit exakt einer
Abweichung — der Datensatz-Liste — und lebt bewusst HIER statt im Submodul: eine
Änderung an ``app/Groot-1.6`` hieße Fork pushen, Dockerfile-Pin nachziehen, Image neu
bauen (~60 min). ``Training/scripts/`` wird ins Image kopiert und auf KISSKI von
``kisski_submit.sh`` als ``/scripts`` eingehängt, wirkt dort also sofort.

Beim Anheben der GR00T-Version diese Datei gegen das Original gegenprüfen:
alles außer dem ``datasets``-Block soll identisch bleiben.

Aufruf (macht ``run_finetuning_cotrain.sh``):
    python launch_cotrain.py --cotrain_dataset_path /data/cotrain/g1_dex3_rendered \
                             --cotrain_mix_ratio 0.5 \
                             --base_model_path ... --dataset_path ... [wie launch_finetune.py]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import tyro
from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def split_own_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Die beiden eigenen Flags vor tyro abfangen.

    tyro baut sein CLI aus den Feldern von ``FinetuneConfig``; unbekannte Flags wären ein
    Fehler. Deshalb erst mit argparse abschöpfen und nur den Rest weiterreichen.
    """
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--cotrain_dataset_path", type=str, default="")
    own.add_argument("--cotrain_mix_ratio", type=float, default=0.5)
    return own.parse_known_args(argv)


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    own_args, rest = split_own_args(sys.argv[1:])
    ft_config = tyro.cli(FinetuneConfig, description=__doc__, args=rest)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    # ── Datensatz-Liste: echt + gerendert ─────────────────────────────────────
    # mix_ratio ist ein relatives Sampling-Gewicht; ShardedMixtureDataset normiert es
    # (und rechnet zusätzlich die Shard-Größen heraus, damit nicht der größere Datensatz
    # allein durch seine Länge dominiert). Wir geben den Anteil des GERENDERTEN Materials
    # an — 0.5 heißt: jede zweite Stichprobe kommt aus der Sim.
    mix = float(own_args.cotrain_mix_ratio)
    datasets = [
        {
            "dataset_paths": [ft_config.dataset_path],
            "mix_ratio": 1.0 - mix,
            "embodiment_tag": embodiment_tag,
        }
    ]
    if own_args.cotrain_dataset_path:
        if not (Path(own_args.cotrain_dataset_path) / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"Gerenderter Datensatz unvollständig: {own_args.cotrain_dataset_path}/meta/"
                "info.json fehlt. Erst `server_rl_run.sh render` fahren."
            )
        if not 0.0 < mix < 1.0:
            raise ValueError(
                f"--cotrain_mix_ratio muss zwischen 0 und 1 liegen (ist {mix}). "
                "0 und 1 sind keine Mischung — dafür den jeweiligen Datensatz allein fahren."
            )
        datasets.append(
            {
                "dataset_paths": [own_args.cotrain_dataset_path],
                "mix_ratio": mix,
                "embodiment_tag": embodiment_tag,
            }
        )
        print(
            f"[cotrain] Zwei Datensätze: real={ft_config.dataset_path} (Gewicht {1 - mix:.2f}), "
            f"gerendert={own_args.cotrain_dataset_path} (Gewicht {mix:.2f})"
        )
    else:
        print("[cotrain] WARNUNG: --cotrain_dataset_path leer — das ist ein normaler "
              "Ein-Datensatz-Lauf, kein Co-Training.")

    config = get_default_config().load_dict({"data": {"download_cache": False,
                                                      "datasets": datasets}})
    config.load_config_path = None

    # ── Ab hier Zeile für Zeile wie launch_finetune.py ────────────────────────
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)
