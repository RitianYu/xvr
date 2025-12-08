from diffdrr.data import read
from diffdrr.drr import DRR
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from deepfluoro import DeepFluoroDataset
from diffdrr.data import load_example_ct
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.pose import RigidTransform


def ap_to_pa_euler_and_xyz(rot, xyz):
    # flip matrix equivalent on Euler angles
    rot_pa = rot.clone()
    rot_pa[:, 1] = -rot_pa[:, 1]   # flip Y
    rot_pa[:, 2] = -rot_pa[:, 2]   # flip Z

    # flip translation
    xyz_pa = xyz.clone()
    xyz_pa[:, 1] = -xyz_pa[:, 1]
    xyz_pa[:, 2] = -xyz_pa[:, 2]

    return rot_pa, xyz_pa


def initialize_drr(
    volume,
    mask,
    labels,
    orientation,
    height,
    width,
    sdd,
    delx,
    dely,
    x0,
    y0,
    reverse_x_axis,
    renderer,
    read_kwargs={},
    drr_kwargs={},
    device="cuda",
):
    # Load the CT volume
    if labels is not None:
        labels = [int(x) for x in labels.split(",")]
    subject = read(volume, mask, labels, orientation, **read_kwargs)
    # Initialize the DRR module at full resolution
    drr = DRR(
        subject,
        sdd,
        height,
        delx,
        width,
        dely,
        x0,
        y0,
        reverse_x_axis=reverse_x_axis,
        renderer=renderer,
        **drr_kwargs,
    ).to(device)
    return drr, subject.volume.affine


def convert_df_pose_to_torchio_world(
    pose_old: torch.Tensor, 
    lps2volume: torch.Tensor,
    affine: torch.Tensor,
):
    """
    Convert DeepFluoro DiffDRR (old coordinate system) pose into the
    TorchIO world coordinate system used by the new DRR class.

    Args:
        pose_old: (4, 4) SE(3) pose from DeepFluoroDataset (__getitem__ output)
        lps2volume: (4, 4) DeepFluoro LPS → volume voxel transform
        affine: (4, 4) TorchIO voxel → RAS world transform

    Returns:
        pose_new_world: (4, 4) pose usable by new DRR.forward()
    """

    # ---------------------------
    # Step 1: Old DiffDRR world  →  LPS world
    # ---------------------------
    # pose_old used lps2volume.inverse() as last step,
    # so undo that to return to DeepFluoro LPS world coordinates.
    pose_lps = lps2volume @ pose_old

    # ---------------------------
    # Step 2: LPS → RAS
    # ---------------------------
    LPS2RAS = torch.diag(torch.tensor([-1., -1., 1., 1.], dtype=pose_old.dtype, device=pose_old.device))
    pose_ras = LPS2RAS @ pose_lps

    # ---------------------------
    # Step 3: RAS world  →  TorchIO world
    # ---------------------------
    # TorchIO: voxel → RAS world = affine
    # So camera pose in TorchIO world = affine @ pose_ras
    pose_new_world = affine @ pose_ras

    return pose_new_world


def cam2lps_to_cam2ras(cam2lps):
    """
    cam2lps: 4×4 matrix (camera → LPS world)

    return: 4×4 matrix (camera → RAS world)
    """
    LPS2RAS = torch.diag(torch.tensor([-1., -1., 1., 1.], dtype=cam2lps.dtype, device=cam2lps.device))
    cam2ras = LPS2RAS @ cam2lps
    return cam2ras


if __name__ == '__main__':
    # Example usage
    device = "cpu"
    drr, affine = initialize_drr(
        volume='/nas2/home/yuhao/code/xvr/data/deepfluoro/subject01/volume.nii.gz',
        mask='/nas2/home/yuhao/code/xvr/data/deepfluoro/subject01/mask.nii.gz',
        labels='1,2,3,4,7',
        orientation="PA",
        height=1436,
        width=1436,
        sdd=1020.0,
        delx=0.194,
        dely=0.194,
        x0=0.0,
        y0=0.0,
        reverse_x_axis=True,
        renderer="trilinear",
        device=device,
    )
    specimen = DeepFluoroDataset(1, filename="/nas2/home/yuhao/code/xvr/data/ipcai_2020_full_res_data.h5", drr=drr)
    # pose_old = specimen[0][1].get_matrix().numpy()[0].T
    # lps2vol = specimen.lps2volume.get_matrix().numpy()[0].T
    # pose= convert_df_pose_to_torchio_world(
    #         torch.from_numpy(pose_old).float(),
    #         torch.from_numpy(lps2vol).float(),
    #         torch.from_numpy(affine).float()
    #     )
    # pose = RigidTransform(pose)

    # cam2lps = specimen[0][1].get_matrix().numpy()[0].T
    # pose = cam2lps_to_cam2ras(
    #     torch.from_numpy(cam2lps).float()
    # )
    # pose = RigidTransform(pose)
    # img = drr(pose)
    # plot_drr(img, ticks=False)
    # plt.savefig("drr_output.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # Generate a DRR at a specific pose
    # rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]]).to(device)
    rot_ap = torch.tensor([[0.0, 0.0, 0.0]]).float().to(device)
    xyz_ap = torch.tensor([0, -(specimen.volume.shape[1]) * specimen.spacing[0] / 2, 0]).unsqueeze(0).float().to(device)
    img = drr(rot_ap, xyz_ap, parameterization="euler_angles", convention="ZXY")
    plot_drr(img, ticks=False)
    plt.savefig("drr_output_deepfluoro.png", dpi=300, bbox_inches="tight")
    plt.close()