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
        model_path = f"/nas2/home/yuhao/code/xvr/models/vessels/{regime}/model.pth"
    else:
        model_path = f"/nas2/home/yuhao/code/xvr/models/vessels/{regime}/ljubljana{subject_id:02d}.pth"
        
    data = f"ljubljana/subject{subject_id:02d}/"
    datapath = Path("/nas2/home/yuhao/code/xvr/data") / data

    model = RegistrarModel(
        volume=datapath / "volume.nii.gz",
        mask=None,
        ckptpath=model_path,
        linearize=True,            # Convert X-rays from exponential to log form
        subtract_background=True,  # Subtract the mean intensity from input X-rays
        reverse_x_axis=False,      # Flip the horizontal axis of the image
        scales="15,7.5,5",         # Downsampling scales for multiscale pose refinement
        patience=5,                # Number of allowed iterations with no improvement
        max_n_itrs=50,             # Number of iterations per scale
        # saveimg=True,
    )

    return model, datapath

model_type = "patient_agnostic"  # "patient_specific" or "patient_agnostic"
with open(f"/nas2/home/yuhao/code/xvr/results/ljubljana/xvr_lju_{model_type}_final_pose.pkl", "rb") as f:
    all_refine_poses = pickle.load(f)

dfs = []
for subject_id in range(1, 11):
    model, datapath = load_regis_model(subject_id, model_type) 
    subj_key = f"subject{subject_id:02d}"
    refine_poses = all_refine_poses[subj_key]
    
    fiducials = torch.load(datapath / "fiducials.pt", weights_only=False).cuda()
    evaluator = Evaluator(model.drr, fiducials)
    
    pred_poses = []
    true_poses = []
    for i in range(len(refine_poses)):
        gt = datapath / "xrays/frontal.pt" if i == 0 else datapath / "xrays/lateral.pt"
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
df.to_csv(f"/nas2/home/yuhao/code/xvr/results/ljubljana/eval_results_final_pose_xvr_lju_{model_type}.csv", index=False)
print("\nAll results saved successfully!")
