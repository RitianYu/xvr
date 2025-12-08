import pdb
import torch
from registrar import RegistrarModel

def debug_register_model():

    # ===== 1. 设置你的输入参数 =====
    xray_path = "/nas2/home/yuhao/code/xvr/data/deepfluoro/subject01/xrays/100.dcm"
    volume_path = "/nas2/home/yuhao/code/xvr/data/deepfluoro/subject01/volume.nii.gz"
    mask_path = "/nas2/home/yuhao/code/xvr/data/deepfluoro/subject01/mask.nii.gz"
    ckpt_path = "/nas2/home/yuhao/code/xvr/models/pelvis/finetuned/deepfluoro01.pth"
    out_path = "debug_output"
    # warp = "/nas2/home/yuhao/code/xvr/data/deepfluoro/subject01/warp.mat"
    warp = None

    # ===== 2. 创建 registrar 对象（与 CLI 一样）=====
    registrar = RegistrarModel(
        volume=volume_path,
        mask=mask_path,
        ckptpath=ckpt_path,
        labels="1,2,3,4,7",         # 改成你需要的
        crop=100,
        subtract_background=False,
        linearize=True,
        reducefn="max",
        warp=warp,
        invert=False,
        scales="24,12,6",
        reverse_x_axis=True,
        renderer="siddon",
        parameterization="euler_angles",
        convention="ZXY",
        lr_rot=1e-2,
        lr_xyz=1e0,
        patience=10,
        threshold=1e-4,
        max_n_itrs=50,              # debug 时可降低迭代
        max_n_plateaus=2,
        init_only=False,
        saveimg=True,
        verbose=2,
    )

    print(">>> DEBUG: RegistrarModel created")

    # ===== 4. 执行 registration（== CLI 的 registrar(i2d, outpath)）=====
    registrar(xray_path, out_path)

    print(">>> DEBUG FINISHED")


if __name__ == "__main__":
    debug_register_model()
