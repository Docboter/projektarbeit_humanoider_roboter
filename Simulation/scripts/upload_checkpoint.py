#!/usr/bin/env python3
r"""
Checkpoint zu HuggingFace Hub hochladen.

Verwendung:
    python upload_checkpoint.py --checkpoint ./checkpoint-110000/ --repo luca-mue/groot-g1dex3-checkpoint
    python upload_checkpoint.py --checkpoint "C:\Users\mueck\Documents\docker\Checkpoints\20260602\checkpoint-110000" --repo luca-mue/groot-g1dex3-checkpoint --token hf_
    python upload_checkpoint.py --help

Wiederaufnahme nach Abbruch: einfach denselben Befehl erneut ausführen.
Bereits hochgeladene Dateien werden übersprungen.

Beschleunigung:
    - hf_transfer (Rust, paralleler Chunk-Upload) wird automatisch aktiviert, wenn das
      Paket installiert ist:  pip install hf_transfer
    - upload_large_folder lädt mehrere Dateien parallel hoch und nutzt Multipart-Upload
      für große safetensors. Worker-Zahl per --workers steuerbar.
"""

import argparse
import os
import sys


def _enable_hf_transfer():
    """hf_transfer aktivieren, falls verfügbar (Rust-Backend, deutlich schneller)."""
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        print("Hinweis: hf_transfer nicht installiert -> langsamerer Upload. "
              "Für Tempo:  pip install hf_transfer", file=sys.stderr)
        return False
    # Muss VOR dem Import von huggingface_hub gesetzt sein.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("hf_transfer aktiviert (paralleler Chunk-Upload).")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True, help="Lokaler Checkpoint-Ordner")
    parser.add_argument("--repo", required=True, help="HF Repo-ID, z. B. luca-mue/groot-g1dex3-checkpoint")
    parser.add_argument("--token", default=None, help="HF-Token (alternativ: HF_TOKEN env-var)")
    parser.add_argument("--private", action="store_true", default=True, help="Repo privat anlegen (default: ja)")
    parser.add_argument("--path-in-repo", default="", help="Zielpfad im Repo (default: Root)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Anzahl paralleler Upload-Worker (default: huggingface_hub-Automatik)")
    parser.add_argument("--include-optimizer", action="store_true", default=False,
                        help="optimizer.pt / rng_state.pth / scheduler.pt mitladen (nur für Training-Resume nötig, +12 GB)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("FEHLER: HF-Token fehlt. --token setzen oder HF_TOKEN env-var.", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isdir(checkpoint_path):
        print(f"FEHLER: Verzeichnis nicht gefunden: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # hf_transfer aktivieren, BEVOR huggingface_hub importiert wird.
    _enable_hf_transfer()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub nicht installiert. Bitte: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    # Repo anlegen falls nicht vorhanden
    try:
        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
        print(f"Repo: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"FEHLER beim Anlegen des Repos: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Lade hoch: {checkpoint_path}")
    print(f"  -> {args.repo}  (path_in_repo='{args.path_in_repo or '/'}')")
    print("Bereits hochgeladene Dateien werden übersprungen.\n")

    # ".*" schließt sämtliche Dotfiles aus — fängt auch unfertige rsync-Teildownloads ab
    # (z. B. ".optimizer.pt.3l2ZiT", versteckte Temp-Datei mit Zufallssuffix). Kein benötigter
    # Checkpoint-Bestandteil beginnt mit einem Punkt.
    ignore = ["*.tmp", "__pycache__", "*.pyc", ".*", "*.partial", "*.incomplete"]
    if not args.include_optimizer:
        ignore += [
            "optimizer.pt*",          # HF Trainer
            "*optim_states*",         # DeepSpeed ZeRO (mp_rank_XX_optim_states.pt)
            "zero_to_fp32.py",        # DeepSpeed ZeRO Konvertierungsscript
            "bf16_zero_pp_rank_*",    # DeepSpeed ZeRO Stage 3 Shards
            "rng_state*.pth",
            "scheduler.pt",
            "training_args.bin",
        ]
        print("Info: Optimizer-States übersprungen (HF + DeepSpeed ZeRO). --include-optimizer zum Mitladen.")

    # upload_large_folder: parallel + Multipart + robustes Resume (commit-pro-Datei).
    # Unterstützt jedoch KEIN path_in_repo -> bei gesetztem Zielpfad auf upload_folder ausweichen.
    use_large = hasattr(api, "upload_large_folder") and not args.path_in_repo
    if use_large:
        print(f"Verwende upload_large_folder (workers={args.workers or 'auto'}).")
        api.upload_large_folder(
            folder_path=checkpoint_path,
            repo_id=args.repo,
            repo_type="model",
            ignore_patterns=ignore,
            num_workers=args.workers,
            print_report=True,
        )
    else:
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=args.repo,
            repo_type="model",
            path_in_repo=args.path_in_repo or None,
            ignore_patterns=ignore,
            run_as_future=False,
        )

    print(f"\nFertig! https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
