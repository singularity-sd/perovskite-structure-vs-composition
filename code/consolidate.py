from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
blocks=[]
audit=[]
for name, mfile, afile, dfile, bfile in [
    ("Split 1","metrics_seed.csv","audit_split.csv","dataset_index.csv","bootstrap_formulas.csv"),
    ("Split 2","split2_metrics_seed.csv","split2_audit_split.csv","split2_dataset_index.csv","split2_bootstrap_formulas.csv")]:
    m=pd.read_csv(P/mfile); m["split"]=name; blocks.append(m)
    a=pd.read_csv(P/afile)
    formulas=pd.read_csv(P/dfile).groupby("split").reduced_formula.apply(set)
    audit.append({"split":name,"train_rows":int(a.loc[a.split=="train","rows"].iloc[0]),"validation_rows":int(a.loc[a.split=="validation","rows"].iloc[0]),"test_rows":int(a.loc[a.split=="test","rows"].iloc[0]),"train_formulas":len(formulas["train"]),"validation_formulas":len(formulas["validation"]),"test_formulas":len(formulas["test"]),"train_validation_overlap":len(formulas["train"]&formulas["validation"]),"train_test_overlap":len(formulas["train"]&formulas["test"]),"validation_test_overlap":len(formulas["validation"]&formulas["test"])})
metrics=pd.concat(blocks,ignore_index=True)
metrics.to_csv(P/"metrics_split_comparison.csv",index=False)
pd.DataFrame(audit).to_csv(P/"audit_formula.csv",index=False)
summary=metrics.groupby(["split","model"],as_index=False).agg(mae_mean=("mae_eV_cell","mean"),mae_sd=("mae_eV_cell","std"),rmse_mean=("rmse_eV_cell","mean"),r2_mean=("r2","mean"),mean_fit_seconds=("fit_seconds","mean"))
summary.to_csv(P/"metrics_summary_full.csv",index=False)
boots=[pd.read_csv(P/f).iloc[0].to_dict() for f in ["bootstrap_formulas.csv","split2_bootstrap_formulas.csv"]]
data={
 "forecast_id":"T3v3-2026-0098","forecast_position":6,"forecast_evidence_count":1160,
 "dataset":"Matbench perovskites","n_samples":18928,"n_unique_reduced_formulas":9646,
 "target":"DFT formation energy in eV per unit cell",
 "split_audits":audit,"metrics_by_split":summary.replace({np.nan:None}).to_dict(orient="records"),
 "formula_cluster_bootstrap":boots,
 "interpretation":"The CrystalGNN had much lower held-out MAE than composition-only baselines in both reduced-formula-group splits. Results are restricted to one computational perovskite dataset and do not establish performance on experimental ceramics or composites.",
 "warnings":[{"category":"software_warning","message":"LightGBM emitted a non-fatal scikit-learn feature-name warning when predicting from the composition NumPy array. The feature order is unchanged; the model completed and returned results."}]
}
(P/"results_full.json").write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")
pd.DataFrame(data["warnings"]).to_csv(P/"warnings.csv",index=False)
fig,axs=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
models=["CrystalGNN","LightGBM_composition","Ridge_composition"]
colors=["#5b88b4","#e0a13c","#7a9c70"]
for ax,split in zip(axs,["Split 1","Split 2"]):
    d=summary[summary.split==split].set_index("model").reindex(models)
    ax.bar(["Crystal GNN","LightGBM\n(composition)","Ridge\n(composition)"],d.mae_mean,yerr=d.mae_sd.fillna(0),capsize=4,color=colors)
    ax.set_title(split+" (formula groups held out)")
    ax.set_ylabel("Test MAE (eV per cell)")
    ax.grid(axis="y",alpha=.2)
fig.tight_layout()
fig.savefig(P.parent/"fig_results.png",dpi=220)
plt.close(fig)
print(summary.to_string(index=False))
print(pd.DataFrame(audit).to_string(index=False))
print(pd.DataFrame(boots).to_string(index=False))
