import numpy as np
import pickle


with open("/nas2/home/yuhao/code/xvr/baseline_pose/rayemb/deepfluoro/rayemb_refine_pose.pkl", "rb") as f:
    result = pickle.load(f)
breakpoint()
pose = result["specimen_4"][23]
print(pose)