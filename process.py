import pandas as pd
import pickle
from collections import defaultdict

# 读取 csv（你可以改为你的路径）
df = pd.read_csv("/nas2/home/yuhao/code/xvr/results/ljubljana/eval_results_final_pose_xvr_lju_finetuned.csv")

result = {}
for _, row in df.iterrows():
    subject = f"subject{int(row['subject']):02d}"   # 0 -> subject00, 1 -> subject01 ...
    model = row["model"]

    # 先保证这一层 model 存在
    model_dict = result.setdefault(model, {})

    # 再保证这一层 subject 存在
    subj_dict = model_dict.setdefault(
        subject, {"mPE": [], "mTRE": [], "dGeo": []}
    )
    subj_dict["mPE"].append(float(row["mPD"]))
    subj_dict["mTRE"].append(float(row["mTRE"]))
    subj_dict["dGeo"].append(float(row["dGeo"]))

for model_name, model_dict in result.items():
    pkl_name = f"/nas2/home/yuhao/code/xvr/results/ljubljana/xvr_lju_{model_name}_metrics.pkl"
    with open(pkl_name, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"保存完成: {pkl_name}")