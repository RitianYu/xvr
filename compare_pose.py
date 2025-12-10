import torch
import pickle


with open("/nas2/home/yuhao/code/xvr/results/deepfluoro/deepfluoro_final_pose_dict.pkl", "rb") as f:
    data_1 = pickle.load(f)

with open("/nas2/home/yuhao/code/xvr/eval_results_new/deepfluoro/xvr_deepfluoro_finetuned_pred_pose.pkl", "rb") as f:
    data_2 = pickle.load(f)

print("Subjects in data_1:", list(data_1.keys()))
print("Subjects in data_2:", list(data_2.keys()))