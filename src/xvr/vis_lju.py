import matplotlib.pyplot as plt
import torch
from diffdrr.visualization import plot_drr

from metrics import Evaluator
from utils import XrayTransforms
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import pickle
import os
from diffdrr.pose import RigidTransform
from huggingface_hub import snapshot_download
from xvr.metrics import Evaluator
from xvr.registrar import RegistrarModel
from xvr.utils import XrayTransforms


def plot_registration(drr, fiducials, gt, pred_pose, true_pose, save_path):
    # Get the registration error
    evaluator = Evaluator(drr, fiducials)
    mtre = evaluator(true_pose, pred_pose)[2]

    # Compute true and predicted DRRs and fiducials
    with torch.no_grad():
        pred_pts = drr.perspective_projection(pred_pose, fiducials).cpu().squeeze()
        true_pts = drr.perspective_projection(true_pose, fiducials).cpu().squeeze()
        pred_img = drr(pred_pose).cpu()
        true_img = drr(true_pose).cpu()
        error = (true_img - pred_img)
    
    xt = XrayTransforms(drr.detector.height, drr.detector.width)
    gt = xt(gt)
    pred_img = xt(pred_img)

    # Plot the fiducials
    axs = plot_drr(torch.concat([pred_img, gt, error]))
    axs[1].scatter(true_pts[..., 0], true_pts[..., 1], color="dodgerblue", label="True")
    axs[1].scatter(pred_pts[..., 0], pred_pts[..., 1], color="darkorange", label="Pred")
    for x, y in zip(pred_pts, true_pts):
        axs[1].plot([x[0], y[0]], [x[1], y[1]], "w--")
    axs[1].legend()

    # Plot the predicted, true, and error images
    plot_drr(
        torch.concat([pred_img, gt, error]),
        title=["DRR from Predicted Pose", "Ground truth X-ray",
               f"Error (mTRE = {mtre:.2f} mm)"],
        ticks=False,
        axs=axs,
    )
    axs[2].imshow(error[0].permute(1, 2, 0), cmap="bwr",
                  vmin=-error.abs().max(),
                  vmax=error.abs().max())

    plt.tight_layout()

    # 🔥 保存图像到指定路径
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 关闭图像避免内存累积
    plt.close()


def load_regis_model(subject_id, regime="finetuned"):
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
    )

    return model, datapath


# load xvr gt pose
with open("/nas2/home/yuhao/code/xvr/eval_results/deepfluoro/xvr_deepfluoro_finetuned_true_pose.pkl", "rb") as f:
    xvr_gt_poses = pickle.load(f)
# load pred pose
with open("/nas2/home/yuhao/code/xvr/results/deepfluoro/xvr_deepfluoro_finetuned_init_pose.pkl", "rb") as f:
    xvr_pred_poses = pickle.load(f)

# Get a couple of the 3D fiducial for error visualization
for subject_id in range(1, 11):
    subj_key = f"subject{subject_id:02d}"
    gt_poses = xvr_gt_poses[subj_key]
    pred_poses = xvr_pred_poses[subj_key]
    model, datapath = load_regis_model(subject_id, "finetuned")
    fiducials = torch.load(datapath / "fiducials.pt", weights_only=False).cuda()
    indices = [0, 25, 50, 75, 110, 150, 250, 375, 500, 750, 1000]
    for i in range(len(gt_poses)):
        img = datapath / "xrays/frontal.dcm" if i == 0 else datapath / "xrays/lateral.dcm"
        gt, _, _, _, _, _, _, _ = model.initialize_pose(img)
        gt_pose = gt_poses[i]
        pred_pose = pred_poses[i]
        gt_pose = RigidTransform(gt_pose).cuda()
        pred_pose = RigidTransform(torch.from_numpy(pred_pose)).cuda()
        # plot
        save_path = f"/nas2/home/yuhao/code/xvr/figures/deepfluoro/subject{subject_id:02d}_view{i:03d}_registration.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_registration(model.drr, fiducials[:, indices], gt, pred_pose, gt_pose, save_path)