from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="eigenvivek/xvr",
    local_dir="/nas2/home/yuhao/code/xvr/ckpts",   # 指定本地目录
    local_dir_use_symlinks=False     
)
