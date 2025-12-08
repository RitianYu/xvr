from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="eigenvivek/xvr-data",
    repo_type="dataset",
    local_dir="/nas2/home/yuhao/code/xvr/data",
    local_dir_use_symlinks=False
)