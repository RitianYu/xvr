import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import pickle
from diffdrr.pose import RigidTransform
from huggingface_hub import snapshot_download
from xvr.metrics import Evaluator
from xvr.registrar import RegistrarModel
from xvr.utils import XrayTransforms
from xvr.visualization import plot_registration


def load_regis_model(subject_id, regime):
    if regime == "patient_agnostic":
        model_path = f"/nas2/home/yuhao/code/xvr/models/pelvis/{regime}/model.pth"
    else:
        model_path = f"/nas2/home/yuhao/code/xvr/models/pelvis/{regime}/deepfluoro{subject_id:02d}.pth"
        
    data = f"deepfluoro/subject{subject_id:02d}/"
    datapath = Path("/nas2/home/yuhao/code/xvr/data") / data

    model = RegistrarModel(
        volume=datapath / "volume.nii.gz",
        mask=datapath / "mask.nii.gz",
        ckptpath=model_path,
        labels="1,2,3,4,7", 
        crop=100,
        subtract_background=False,
        linearize=True,
        reducefn="max",
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
        max_n_itrs=500, 
        max_n_plateaus=3,
        init_only=False,
        saveimg=True,
        verbose=2,
    )

    return model, datapath

model_type = "finetuned"  # "patient_specific" or "patient_agnostic"
with open(f"/nas2/home/yuhao/code/xvr/results/deepfluoro/xvr_deepfluoro_{model_type}_init_pose.pkl", "rb") as f:
    all_refine_poses = pickle.load(f)

dfs = []
for subject_id in range(1, 7):
    model, datapath = load_regis_model(subject_id, model_type) 
    subj_key = f"subject{subject_id:02d}"
    refine_poses = all_refine_poses[subj_key]
    
    fiducials = torch.load(datapath / "fiducials.pt", weights_only=False).cuda()
    evaluator = Evaluator(model.drr, fiducials)
    
    pred_poses = []
    true_poses = []
    for i in range(len(refine_poses)):
        gt = datapath / "xrays" / f"{i:03d}.pt"
        true_pose = RigidTransform(torch.load(gt, weights_only=False)["pose"])
        true_poses.append(true_pose.matrix)
        pred_poses.append(torch.from_numpy(refine_poses[i]))
    pred_poses = RigidTransform(torch.concat(pred_poses)).cuda()
    true_poses = RigidTransform(torch.concat(true_poses)).cuda()

    metrics = evaluator(true_poses, pred_poses)     

    df = pd.DataFrame(metrics, columns=["mPD", "mRPE", "mTRE", "dGeo"])
    df["xray"] = range(len(metrics))
    df["subject"] = subject_id
    df["model"] = model_type
    dfs.append(df)

df = pd.concat(dfs)
df.to_csv(f"/nas2/home/yuhao/code/xvr/results/deepfluoro/eval_results_init_pose_xvr_deepfluoro_{model_type}.csv", index=False)
print("\nAll results saved successfully!")
