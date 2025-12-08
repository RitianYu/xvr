import numpy as np
import pickle
from src.xvr.deepfluoro import DeepFluoroDataset

TA1 = np.array([[ 9.9738e-01, -2.6244e-04, -7.2396e-02,  7.4656e+00],
                [ 2.8413e-03,  9.9936e-01,  3.5520e-02, -7.4414e+02],
                [ 7.2341e-02, -3.5633e-02,  9.9674e-01,  1.3736e+01],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
TA2 = np.array([[ 9.7903e-01, -9.9657e-03, -2.0349e-01, -1.0057e+02],
                [ 1.0254e-02,  9.9995e-01,  3.6325e-04, -7.9759e+02],
                [ 2.0348e-01, -2.4422e-03,  9.7908e-01, -3.1658e+01],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

TB1 = np.array([[ 5.5160746e-04, -7.2153442e-02, -9.9739355e-01,  1.9127332e+02],
                [ 9.9934024e-01,  3.6267675e-02, -2.0710249e-03,  3.3263861e+02],
                [ 3.6322486e-02, -9.9673408e-01,  7.2125822e-02,  1.6569489e+02],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
TB2 = np.array([[-8.8069960e-03, -2.0238407e-01, -9.7926646e-01,  3.0431927e+02],
                [ 9.9995518e-01,  1.6294795e-03, -9.3298508e-03,  3.8507867e+02],
                [ 3.4838570e-03, -9.7930497e-01,  2.0236057e-01,  1.3757379e+02],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

dA = np.linalg.inv(TA2) @ TA1
dB = np.linalg.inv(TB2) @ TB1

def rotation_angle(R):
    # Clamp for numerical stability
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# Compute rotation angles
angle_gt = rotation_angle(dA[:3,:3])
angle_pred = rotation_angle(dB[:3,:3])

print("GT ΔR angle:", angle_gt)
print("Pred ΔR angle:", angle_pred)

print('GT Δt norm:', np.linalg.norm(dA[:3,3]))
print('Pred Δt norm:', np.linalg.norm(dB[:3,3]))

X1 = TA1 @ np.linalg.inv(TB1)
X2 = TA2 @ np.linalg.inv(TB2)

print("X1\n", X1)
print("X2\n", X2)
print("X1 - X2\n", X1 - X2)

