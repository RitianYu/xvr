import pandas as pd


df = pd.read_csv("/nas2/home/yuhao/code/xvr/results/deepfluoro/eval_results_final_pose_xvr_deepfluoro_patient_agnostic.csv")

# median by model
medians = df.groupby('model')[['mPD','mTRE','dGeo']].median()

# success rates (<1)
total = len(df)
success_counts = {
    metric: int((df[metric] < 1).sum())
    for metric in ['mPD','mTRE','dGeo']
}
success_rates = {k: v/total for k,v in success_counts.items()}

print("Median Metrics by Model:")
print(medians)
print("\nSuccess Rates:")
for metric, rate in success_rates.items():
    print(f"{metric}: {rate:.2%}")