import os
from huggingface_hub import snapshot_download

DATASET_ID = "unitreerobotics/G1_Dex3_BlockStacking_Dataset"
MODEL_ID = "nvidia/groot-1.7"  # ggf. anpassen, falls anderer Repo-Name

def folder_has_files(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0

def download_dataset(target_dir):
    if folder_has_files(target_dir):
        print(f"[INFO] Dataset existiert bereits in {target_dir}")
        return

    print("[INFO] Lade Dataset...")
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )

def download_model(target_dir):
    if folder_has_files(target_dir):
        print(f"[INFO] Modell existiert bereits in {target_dir}")
        return

    print("[INFO] Lade Modell...")
    snapshot_download(
        repo_id=MODEL_ID,
        repo_type="model",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )

if __name__ == "__main__":
    download_dataset("/data/raw")
    download_model("/models")