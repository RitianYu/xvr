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
        model_path = f"/nas2/home/yuhao/code/xvr/models/pelvis/{regime}/model.pth"
    else:
        model_path = f"/nas2/home/yuhao/code/xvr/models/pelvis/{regime}/ljubljana{subject_id:02d}.pth"
        
    data = f"ljubljana/subject{subject_id:02d}/"
    datapath = Path("/nas2/home/yuhao/code/xvr/data") / data

    # Initialize the model
    model = RegistrarModel(
        volume=datapath / "volume.nii.gz",
        mask=None,
        ckptpath=model_path,
        crop=100,
        init_only=True,
    )
    return model, datapath


# load xvr gt pose
with open("/nas2/home/yuhao/code/xvr/eval_results/lju/xvr_lju_finetuned_true_pose.pkl", "rb") as f:
    xvr_gt_poses = pickle.load(f)
# load pred pose
with open("/nas2/home/yuhao/code/xvr/results/ljubljana/xvr_lju_finetuned_init_pose.pkl", "rb") as f:
    xvr_pred_poses = pickle.load(f)

with open("/nas2/home/yuhao/code/DiffPose/diffpose_deepfluoro_final_pose.pkl", "rb") as f:
    diffpose_pred_poses = pickle.load(f)

with open("/nas2/home/yuhao/code/DiffPose/diffpose_deepfluoro_true_pose.pkl", "rb") as f:
    diffpose_gt_poses = pickle.load(f)

xvr_gt_pose = xvr_gt_poses["subject04"][23]
xvr_pred_pose = xvr_pred_poses["subject04"][23]
diffpose_gt_pose = diffpose_gt_poses["subject04"][23]
diffpose_pred_pose = diffpose_pred_poses["subject04"][23]
diffdrr_pred_pose = np.array([[ 1.1181402e-01,  1.4440361e-01, -9.8318118e-01,  2.5493057e+02],
                            [ 9.7201121e-01, -2.2161019e-01,  7.7995002e-02,  3.4208435e+02],
                            [-2.0662016e-01, -9.6438402e-01, -1.6514111e-01,  1.5238107e+02],
                            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
rayemb_pred_pose = np.array([[ 1.3893247e-02,  1.9994569e-01, -9.7970843e-01,  2.3723639e+02],
                            [ 9.9812734e-01, -6.1147571e-02,  1.6750393e-03,  3.6666797e+02],
                            [-5.9571885e-02, -9.7789705e-01, -2.0042074e-01,  1.8902480e+02],
                            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
ours_gt_pose = np.array([[-6.6595594e-03,  2.2345352e-01, -9.7469169e-01,  2.3340295e+02]
                    [ 9.9938536e-01, -3.2059167e-02, -1.4178041e-02,  3.6855161e+02]
                    [-3.4415815e-02, -9.7418731e-01, -2.2310278e-01,  1.9730029e+02]
                    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
ours_pose = np.array([[-7.6367264e-03, 2.2336294e-01, -9.7470528e-01, 2.3307372e+02],
                      [ 9.9935824e-01, -3.2406747e-02, -1.5256212e-02, 3.6919470e+02],
                      [-3.4994580e-02, -9.7419661e-01, -2.2297221e-01 ,1.9707440e+02],
                      [ 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
roma_pose = np.array([[ 3.0052885e-03,  2.2452125e-01, -9.7446442e-01,  2.3662077e+02]
                      [ 9.9995452e-01, -9.4896518e-03,  8.9741865e-04,  3.6271997e+02]
                      [-9.0457201e-03, -9.7442311e-01, -2.2453967e-01,  2.0729166e+02]
                      [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

method = "xvr"
model, datapath = load_regis_model(4, "finetuned")
img = datapath / "xrays/023.dcm" 
gt, _, _, _, _, _, _, _ = model.initialize_pose(img)
# Load the ground truth 3D fiducials
fiducials = torch.load(datapath / "fiducials.pt", weights_only=False).cuda()

if method == "xvr":
    gt_pose = RigidTransform(xvr_gt_pose).cuda()
    pred_pose = RigidTransform(torch.from_numpy(xvr_pred_pose)).cuda()
else: 
    # 计算从原方法坐标系到xvr坐标系的变换矩阵
    gt_pose = RigidTransform(xvr_gt_pose).cuda()
    org_gt_pose = RigidTransform(torch.from_numpy(ours_gt_pose)).cuda()
    T_xvr_to_org = gt_pose.compose(org_gt_pose.inverse())
    # 将pred_pose从原方法坐标系转换到xvr坐标系
    if method == "diffpose":
        pred_pose_org = RigidTransform(torch.from_numpy(diffpose_pred_pose)).cuda()
    if method == "roma":
        pred_pose_org = RigidTransform(torch.from_numpy(roma_pose)).cuda()
    if method == "ours":
        pred_pose_org = RigidTransform(torch.from_numpy(ours_pose)).cuda()
    pred_pose = T_xvr_to_org.compose(pred_pose_org)

# plot
save_path = f"/nas2/home/yuhao/code/xvr/figures/deepfluoro/subject04_0023_registration.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plot_registration(model.drr, fiducials, gt, pred_pose, gt_pose, save_path)