from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
    repo_type="dataset",
    local_dir="./data/LAFAN1",
    local_dir_use_symlinks=False,   # set True if you want symlinks to cache
)
print("Downloaded to:", local_dir)
