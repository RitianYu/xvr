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


regimes = ["patient_specific", "finetuned"]

dfs = []


final_pose_dicts = {regime: {} for regime in regimes}
true_pose_dicts  = {regime: {} for regime in regimes}
for regime in regimes:
    print(f"\n========== Processing regime: {regime} ==========\n")
    for subject_id in range(1, 7):
        subj_key = f"subject{subject_id:02d}"

        final_pose_dicts[regime][subj_key] = []
        true_pose_dicts[regime][subj_key] = []

        model, datapath = load_regis_model(subject_id, regime)
        xrays = datapath / "xrays"

        fiducials = torch.load(datapath / "fiducials.pt", weights_only=False).cuda()
        evaluator = Evaluator(model.drr, fiducials)

        pred_poses = []
        true_poses = []

        print(f"Evaluating Subject {subject_id} ({subj_key}) ...")
        for img, gt in tqdm(list(zip(sorted(xrays.glob("*.dcm")),
                                     sorted(xrays.glob("*.pt"))))):

            true_pose = RigidTransform(torch.load(gt, weights_only=False)["pose"])
            _, _, _, init_pose, final_pose, _ = model.run(img, beta=0.5)
            final_cpu = final_pose.cpu()

            pred_poses.append(final_cpu.matrix)
            true_poses.append(true_pose.matrix)

            final_pose_dicts[regime][subj_key].append(final_cpu.matrix)
            true_pose_dicts[regime][subj_key].append(true_pose.matrix)

        pred_poses = RigidTransform(torch.concat(pred_poses)).cuda()
        true_poses = RigidTransform(torch.concat(true_poses)).cuda()

        metrics = evaluator(true_poses, pred_poses)

        df = pd.DataFrame(metrics, columns=["mPD", "mRPE", "mTRE", "dGeo"])
        df["xray"] = range(len(metrics))
        df["subject"] = subject_id
        df["model"] = regime
        dfs.append(df)


out_dir = "/nas2/home/yuhao/code/xvr/eval_results/deepfluoro"
df = pd.concat(dfs)
df.to_csv(f"{out_dir}/eval_results.csv", index=False)

for regime in regimes:
    with open(f"{out_dir}/xvr_deepfluoro_{regime}_pred_pose.pkl", "wb") as f:
        pickle.dump(final_pose_dicts[regime], f)

    with open(f"{out_dir}/xvr_deepfluoro_{regime}_true_pose.pkl", "wb") as f:
        pickle.dump(true_pose_dicts[regime], f)

print("\nAll results saved successfully!")
