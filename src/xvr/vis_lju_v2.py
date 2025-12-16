import matplotlib.pyplot as plt
import torch
import numpy as np  # 🔥 添加numpy导入
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
with open("/nas2/home/yuhao/code/xvr/eval_results/lju/xvr_lju_finetuned_true_pose.pkl", "rb") as f:
    xvr_gt_poses = pickle.load(f)

with open("/nas2/home/yuhao/code/DiffPose/diffpose_deepfluoro_true_pose.pkl", "rb") as f:
    diffpose_gt_poses = pickle.load(f)

# specimen-1 
xvr_gt_pose_1 = xvr_gt_poses["subject01"][1]
org_gt_pose_1 = diffpose_gt_poses["specimen_1"][1]
ours_zeroshot_pose_1 = np.array([[-9.9980837e-01, -2.2383314e-03, -1.9429363e-02, -4.4289948e+01],
                                [ 1.9442350e-02, -6.2415954e-03, -9.9979180e-01,  1.1688117e+02],
                                [ 2.1171235e-03, -9.9997783e-01,  6.2847910e-03,  9.1103371e+01],
                                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

ours_pose_1 = np.array([[-9.9983215e-01,  4.0400508e-04, -1.8297276e-02, -4.1272644e+01],
                      [ 1.8295093e-02, -3.5081902e-03, -9.9982673e-01,  1.1597821e+02],
                      [-4.6759733e-04, -9.9999356e-01,  3.5010795e-03,  9.1007309e+01],
                      [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

roma_pose_1 = np.array([[-9.9979508e-01,  4.0328964e-03, -1.9816289e-02, -2.7944374e+01],
                      [ 1.9795354e-02, -5.0070072e-03, -9.9979180e-01,  1.1645459e+02],
                      [-4.1307467e-03, -9.9997908e-01,  4.9270173e-03,  9.1511574e+01],
                      [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

# specimen-2
xvr_gt_pose_2 = xvr_gt_poses["subject02"][1]
org_gt_pose_2 = diffpose_gt_poses["specimen_2"][1]
ours_zeroshot_pose_2 = np.array([[-9.99559879e-01, -3.53437359e-03, -2.94549037e-02,  2.18611088e+01],
                                [ 2.94432566e-02,  3.30320443e-03, -9.99561012e-01,  1.06093155e+02],
                                [ 3.63011751e-03, -9.99988317e-01, -3.19768721e-03,  9.10738983e+01],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
                                )
ours_pose_2 = np.array([[-9.9958503e-01,  9.8504068e-05, -2.8821087e-02,  1.8513218e+01],
                      [ 2.8820643e-02,  7.1755238e-04, -9.9958462e-01,  1.0593500e+02],
                      [-7.7455486e-05, -1.0000002e+00, -7.1943889e-04,  9.1044281e+01],
                      [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
roma_pose_2 = np.array([[-9.99415696e-01, -5.24871517e-04, -3.41886170e-02,  1.47406406e+01],
                    [ 3.41888033e-02, -1.36083621e-03, -9.99414682e-01,  1.05116486e+02],
                    [ 4.78367030e-04, -9.99999464e-01,  1.37863937e-03,  9.06689911e+01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# specimen-6
xvr_gt_pose_6 = xvr_gt_poses["subject06"][1]
org_gt_pose_6 = diffpose_gt_poses["specimen_6"][1]
ours_zeroshot_pose_6 = np.array([[-9.99924362e-01,  6.46424713e-04, -1.22781228e-02, -3.72666893e+01],
                               [ 1.22776488e-02, -5.72025310e-04, -9.99924898e-01,  1.18099625e+02],
                               [-6.53781055e-04, -1.00000024e+00,  5.64005342e-04,  9.28364563e+01],
                               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
ours_pose_6 = np.array([[-9.9993736e-01, -1.8212531e-03, -1.1039609e-02, -3.3871609e+01],
                      [ 1.1044299e-02, -2.6442530e-03, -9.9993587e-01,  1.1808441e+02],
                      [ 1.7915609e-03, -9.9999535e-01,  2.6641614e-03,  9.2170113e+01],
                      [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
roma_pose_6 = np.array([[-9.9984431e-01, -3.9787744e-03, -1.7189713e-02, -4.0651978e-01],
                      [ 1.7214814e-02, -6.3913334e-03, -9.9983180e-01,  1.1862542e+02],
                      [ 3.8678544e-03, -9.9997216e-01,  6.4587914e-03,  9.2349419e+01],
                      [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

xvr_gt_poses = {}
org_gt_poses = {}
ours_zeroshot_poses = {}
ours_poses = {}
roma_poses = {}
for subject_id in [1, 2, 6]:
    subj_key = f"subject{subject_id:02d}"
    if subject_id == 1:
        xvr_gt_poses[subj_key] = xvr_gt_pose_1
        org_gt_poses[subj_key] = org_gt_pose_1
        ours_zeroshot_poses[subj_key] = ours_zeroshot_pose_1
        ours_poses[subj_key] = ours_pose_1
        roma_poses[subj_key] = roma_pose_1
    elif subject_id == 2:
        xvr_gt_poses[subj_key] = xvr_gt_pose_2
        org_gt_poses[subj_key] = org_gt_pose_2
        ours_zeroshot_poses[subj_key] = ours_zeroshot_pose_2
        ours_poses[subj_key] = ours_pose_2
        roma_poses[subj_key] = roma_pose_2
    elif subject_id == 6:
        xvr_gt_poses[subj_key] = xvr_gt_pose_6
        org_gt_poses[subj_key] = org_gt_pose_6
        ours_zeroshot_poses[subj_key] = ours_zeroshot_pose_6
        ours_poses[subj_key] = ours_pose_6
        roma_poses[subj_key] = roma_pose_6

for id in [1,2,6]:
    subject_id = id
    method = "xvr"
    model, datapath = load_regis_model(subject_id, "finetuned")
    img = datapath / "xrays/lateral.dcm"
    gt, _, _, _, _, _, _, _ = model.initialize_pose(img)
    # Load the ground truth 3D fiducials
    fiducials = torch.load(datapath / "fiducials.pt", weights_only=False).cuda()
    indices = [0, 25, 50, 75, 110, 150, 250, 375, 500, 750, 1000]

    subj_key = f"subject{subject_id:02d}"
    xvr_gt_pose = xvr_gt_poses[subj_key]
    org_gt_pose = org_gt_poses[subj_key]
    roma_pose = roma_poses[subj_key]
    ours_pose = ours_poses[subj_key]
    ours_zeroshot_pose = ours_zeroshot_poses[subj_key]

    # 计算从原方法坐标系到xvr坐标系的变换矩阵
    gt_pose = RigidTransform(xvr_gt_pose).cuda()
    org_gt_pose = RigidTransform(torch.from_numpy(org_gt_pose)).cuda()
    T_xvr_to_org = gt_pose.compose(org_gt_pose.inverse())
    # 将pred_pose从原方法坐标系转换到xvr坐标系
    if method == "roma":
        pred_pose_org = RigidTransform(torch.from_numpy(roma_pose)).cuda()
    if method == "ours":
        pred_pose_org = RigidTransform(torch.from_numpy(ours_pose)).cuda()
    if method == "ours_zeroshot":
        pred_pose_org = RigidTransform(torch.from_numpy(ours_zeroshot_pose)).cuda()
    pred_pose = T_xvr_to_org.compose(pred_pose_org)

    # plot
    save_path = f"/nas2/home/yuhao/code/xvr/figures/lju/subject1_lat_registration.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_registration(model.drr, fiducials, gt, pred_pose, gt_pose, save_path)