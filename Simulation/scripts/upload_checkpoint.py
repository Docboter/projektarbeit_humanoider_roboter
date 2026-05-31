#!/usr/bin/env python3
"""
Checkpoint zu HuggingFace Hub hochladen.

Verwendung:
    python upload_checkpoint.py --checkpoint ./checkpoint-3000/ --repo luca-mue/groot-g1dex3-checkpoint
    python upload_checkpoint.py --checkpoint "C:\Users\mueck\Documents\docker\Checkpoints\20260529\checkpoint-3000" --repo luca-mue/groot-g1dex3-checkpoint --token hf_
    python upload_checkpoint.py --help

Wiederaufnahme nach Abbruch: einfach denselben Befehl erneut ausführen.
Bereits hochgeladene Dateien werden übersprungen.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True, help="Lokaler Checkpoint-Ordner")
    parser.add_argument("--repo", required=True, help="HF Repo-ID, z. B. luca-mue/groot-g1dex3-checkpoint")
    parser.add_argument("--token", default=None, help="HF-Token (alternativ: HF_TOKEN env-var)")
    parser.add_argument("--private", action="store_true", default=True, help="Repo privat anlegen (default: ja)")
    parser.add_argument("--path-in-repo", default="", help="Zielpfad im Repo (default: Root)")
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

    ignore = ["*.tmp", "__pycache__", "*.pyc"]
    if not args.include_optimizer:
        ignore += ["optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"]
        print("Info: optimizer.pt / rng_state.pth / scheduler.pt übersprungen (--include-optimizer zum Mitladen).")

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
