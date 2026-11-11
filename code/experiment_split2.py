"""Crystal-graph message passing for the Matbench perovskites dataset."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from pymatgen.core import Structure; from pymatgen.core.periodic_table import Element
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"matbench_perovskites.json.gz"
OUT=ROOT
SEEDS=[17,29,43]
MAX_EPOCHS,PATIENCE,BATCH=60,8,64
N_NEIGHBORS=16
RADIUS=8.0
DEVICE=torch.device("cpu")
torch.set_num_threads(4)

def metrics(y,p):
    return {"mae_eV_cell":float(mean_absolute_error(y,p)),"rmse_eV_cell":float(np.sqrt(mean_squared_error(y,p))),"r2":float(r2_score(y,p))}

def composition_features(structures):
    out=np.zeros((len(structures),118),dtype=np.float32)
    for i,s in enumerate(structures):
        amounts=s.composition.get_el_amt_dict()
        total=sum(amounts.values())
        for symbol,amt in amounts.items():
            z=Element(symbol).Z
            out[i,int(z)-1]=float(amt/total)
    return out

def build_graphs(structures):
    atom_z=[]; node_ptr=[0]
    edge_src=[]; edge_dst=[]; edge_dist=[]; edge_ptr=[0]
    formulas=[]
    for ix,s in enumerate(structures):
        formulas.append(s.composition.reduced_formula)
        base=len(atom_z)
        atom_z.extend([int(site.specie.Z) for site in s])
        all_nbs=s.get_all_neighbors(r=RADIUS,include_index=True)
        for center,nbs in enumerate(all_nbs):
            chosen=sorted(nbs,key=lambda nb:nb.nn_distance)[:N_NEIGHBORS]
            for nb in chosen:
                edge_src.append(base+int(nb.index))
                edge_dst.append(base+center)
                edge_dist.append(float(nb.nn_distance))
        node_ptr.append(len(atom_z)); edge_ptr.append(len(edge_src))
        if (ix+1)%2500==0: print(f"Prepared graphs: {ix+1}/{len(structures)}",flush=True)
    return (np.asarray(atom_z,dtype=np.int64),np.asarray(node_ptr,dtype=np.int64),
            np.asarray(edge_src,dtype=np.int64),np.asarray(edge_dst,dtype=np.int64),
            np.asarray(edge_dist,dtype=np.float32),np.asarray(edge_ptr,dtype=np.int64),formulas)

def collate_graphs(gids, graphs):
    atom_z,node_ptr,src,dst,dist,edge_ptr,_=graphs
    zs=[]; ss=[]; ds=[]; dd=[]; graph_id=[]; off=0
    for local,g in enumerate(gids):
        ns,ne=node_ptr[g],node_ptr[g+1]
        es,ee=edge_ptr[g],edge_ptr[g+1]
        zs.append(atom_z[ns:ne])
        ss.append(src[es:ee]-ns+off); ds.append(dst[es:ee]-ns+off); dd.append(dist[es:ee])
        graph_id.append(np.full(ne-ns,local,dtype=np.int64))
        off+=ne-ns
    return (torch.as_tensor(np.concatenate(zs),dtype=torch.long,device=DEVICE),
            torch.as_tensor(np.stack([np.concatenate(ss),np.concatenate(ds)]),dtype=torch.long,device=DEVICE),
            torch.as_tensor(np.concatenate(dd),dtype=torch.float32,device=DEVICE),
            torch.as_tensor(np.concatenate(graph_id),dtype=torch.long,device=DEVICE))

class CrystalGNN(nn.Module):
    def __init__(self,width=64,radial=16,layers=3):
        super().__init__()
        self.atom=nn.Embedding(119,width)
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*width+radial,128),nn.SiLU(),nn.Linear(128,width),nn.SiLU()) for _ in range(layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*width,width),nn.SiLU()) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))
        self.register_buffer("centers",torch.linspace(0,RADIUS,radial))
    def forward(self,z,edge_index,distance,graph_id):
        h=self.atom(z)
        src,dst=edge_index[0],edge_index[1]
        rbf=torch.exp(-0.5*((distance[:,None]-self.centers[None,:])/.5)**2)
        n=h.shape[0]
        for msgnet,upnet,norm in zip(self.messages,self.updates,self.norms):
            msg=msgnet(torch.cat([h[src],h[dst],rbf],dim=1))
            agg=torch.zeros_like(h); agg.index_add_(0,dst,msg)
            deg=torch.zeros((n,1),device=h.device); deg.index_add_(0,dst,torch.ones((len(dst),1),device=h.device))
            agg=agg/deg.clamp_min(1)
            h=norm(h+upnet(torch.cat([h,agg],dim=1)))
        ng=int(graph_id.max().item())+1
        pooled=torch.zeros((ng,h.shape[1]),device=h.device); pooled.index_add_(0,graph_id,h)
        counts=torch.zeros((ng,1),device=h.device); counts.index_add_(0,graph_id,torch.ones((len(graph_id),1),device=h.device))
        pooled=pooled/counts.clamp_min(1)
        return self.head(pooled).squeeze(-1)

def predict_gnn(model,ids,graphs,batch_size=BATCH):
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):
            b=ids[start:start+batch_size]
            z,e,d,g=collate_graphs(b,graphs)
            out.extend(model(z,e,d,g).cpu().numpy().tolist())
    return np.asarray(out)

def main():
    df=pd.read_json(DATA,compression="gzip",orient="split")
    structures=[Structure.from_dict(x) for x in df["structure"]]
    y=df["e_form"].to_numpy(dtype=np.float32)
    formulas=np.asarray([s.composition.reduced_formula for s in structures])
    if not (len(df)==18928 and len(np.unique(formulas))==9646):
        raise RuntimeError("Unexpected Matbench perovskites dataset audit.")
    cache=OUT/"graphs_cache.npz"
    if cache.exists():
        c=np.load(cache)
        graphs=(c["atom_z"],c["node_ptr"],c["edge_src"],c["edge_dst"],c["edge_dist"],c["edge_ptr"],formulas)
    else:
        graphs=build_graphs(structures)
        atom_z,node_ptr,edge_src,edge_dst,edge_dist,edge_ptr,_=graphs
        np.savez_compressed(cache,atom_z=atom_z,node_ptr=node_ptr,edge_src=edge_src,edge_dst=edge_dst,edge_dist=edge_dist,edge_ptr=edge_ptr)
    atom_z,node_ptr,edge_src,edge_dst,edge_dist,edge_ptr,_=graphs
    x_comp=composition_features(structures)
    idx=np.arange(len(y))
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=2028).split(idx,y,groups=formulas))
    tr_local,val_local=next(GroupShuffleSplit(n_splits=1,test_size=.125,random_state=2029).split(trainval,y[trainval],groups=formulas[trainval]))
    train,val=trainval[tr_local],trainval[val_local]
    split=pd.DataFrame({"split":["train","validation","test"],"rows":[len(train),len(val),len(test)],"unique_formulas":[len(set(formulas[train])),len(set(formulas[val])),len(set(formulas[test]))]})
    split.to_csv(OUT/"split2_audit_split.csv",index=False)
    pd.DataFrame({"index":idx,"reduced_formula":formulas,"e_form_eV_per_cell":y,"split":np.where(np.isin(idx,train),"train",np.where(np.isin(idx,val),"validation","test"))}).to_csv(OUT/"split2_dataset_index.csv",index=False)
    results=[]; pred_rows=[]; all_warnings=[]
    # Composition-only baselines, fit on the same training formulas.
    models=[("Ridge_composition",make_pipeline(StandardScaler(),Ridge(alpha=10.0))),
            ("LightGBM_composition",LGBMRegressor(n_estimators=700,learning_rate=.04,num_leaves=31,min_child_samples=20,subsample=.85,colsample_bytree=.8,reg_lambda=1.0,random_state=2026,n_jobs=4,verbosity=-1))]
    for name,model in models:
        start=time.perf_counter(); model.fit(x_comp[train],y[train]); seconds=time.perf_counter()-start
        pr=model.predict(x_comp[test]); row={"model":name,"seed":2026,"fit_seconds":seconds,**metrics(y[test],pr)}
        results.append(row)
        for j,i in enumerate(test): pred_rows.append({"model":name,"seed":2026,"index":int(i),"formula":formulas[i],"true_e_form":float(y[i]),"pred_e_form":float(pr[j])})
        print(json.dumps(row),flush=True)
    target_mean=float(y[train].mean()); target_sd=float(y[train].std())
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model=CrystalGNN().to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float("inf"); best_state=None; wait=0; best_epoch=0
        start=time.perf_counter()
        for epoch in range(1,MAX_EPOCHS+1):
            model.train(); rng=np.random.default_rng(seed+epoch); order=rng.permutation(train)
            for st in range(0,len(order),BATCH):
                ids=order[st:st+BATCH]
                z,e,d,g=collate_graphs(ids,graphs)
                target=torch.as_tensor((y[ids]-target_mean)/target_sd,dtype=torch.float32,device=DEVICE)
                opt.zero_grad(set_to_none=True)
                loss=torch.mean((model(z,e,d,g)-target)**2)
                loss.backward(); opt.step()
            val_pred=predict_gnn(model,val,graphs)*target_sd+target_mean
            val_mae=float(mean_absolute_error(y[val],val_pred))
            if val_mae < best-1e-5:
                best,best_epoch,wait=val_mae,epoch,0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                wait+=1
                if wait>=PATIENCE: break
            if epoch%10==0: print(f"seed={seed} epoch={epoch} validation_mae={val_mae:.5f}",flush=True)
        seconds=time.perf_counter()-start
        model.load_state_dict(best_state)
        test_pred=predict_gnn(model,test,graphs)*target_sd+target_mean
        row={"model":"CrystalGNN","seed":seed,"best_epoch":best_epoch,"validation_mae_eV_cell":best,"fit_seconds":seconds,**metrics(y[test],test_pred)}
        results.append(row)
        for j,i in enumerate(test): pred_rows.append({"model":"CrystalGNN","seed":seed,"index":int(i),"formula":formulas[i],"true_e_form":float(y[i]),"pred_e_form":float(test_pred[j])})
        print(json.dumps(row),flush=True)
    met=pd.DataFrame(results); met.to_csv(OUT/"split2_metrics_seed.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(OUT/"split2_test_predictions.csv",index=False)
    # Bootstrap paired MAE differences by held-out reduced formula.
    ptab=pd.DataFrame(pred_rows)
    gnn=ptab[ptab.model=="CrystalGNN"].copy()
    gnn["pred_e_form"]=gnn.groupby("index").pred_e_form.transform("mean")
    base=ptab[ptab.model=="LightGBM_composition"].set_index("index")
    gnn=gnn.drop_duplicates("index").set_index("index").join(base[["pred_e_form"]],rsuffix="_baseline")
    gr=gnn.groupby("formula").apply(lambda a:(a.index.to_numpy(),a.true_e_form.to_numpy(),a.pred_e_form.to_numpy(),a.pred_e_form_baseline.to_numpy()),include_groups=False)
    items=list(gr)
    rng=np.random.default_rng(20261107); boot=[]
    for _ in range(2000):
        chosen=rng.integers(0,len(items),size=len(items))
        ys=np.concatenate([items[k][1] for k in chosen])
        pg=np.concatenate([items[k][2] for k in chosen])
        pb=np.concatenate([items[k][3] for k in chosen])
        boot.append(mean_absolute_error(ys,pg)-mean_absolute_error(ys,pb))
    bootrow={"model_a":"CrystalGNN","model_b":"LightGBM_composition","mae_difference_eV_cell":float(mean_absolute_error(gnn.true_e_form.to_numpy(),gnn.pred_e_form.to_numpy())-mean_absolute_error(gnn.true_e_form.to_numpy(),gnn.pred_e_form_baseline.to_numpy())),"bootstrap_95_low":float(np.quantile(boot,.025)),"bootstrap_95_high":float(np.quantile(boot,.975)),"formula_clusters":len(items),"replicates":len(boot)}
    pd.DataFrame([bootrow]).to_csv(OUT/"split2_bootstrap_formulas.csv",index=False)
    summary=met.groupby("model").agg(mean_mae=("mae_eV_cell","mean"),sd_mae=("mae_eV_cell","std"),mean_rmse=("rmse_eV_cell","mean"),mean_r2=("r2","mean"),mean_fit_seconds=("fit_seconds","mean")).reset_index()
    summary.to_csv(OUT/"split2_metrics_summary.csv",index=False)
    (OUT/"split2_results.json").write_text(json.dumps({"forecast_id":"T3v3-2026-0098","dataset":"Matbench perovskites","target":"DFT formation energy, eV per unit cell","n_samples":len(y),"n_unique_reduced_formulas":len(set(formulas)),"split":"80/10/20 group split by reduced formula; no reduced formula crosses train, validation, and test","n_train":len(train),"n_validation":len(val),"n_test":len(test),"split_audit":split.to_dict(orient="records"),"models":summary.to_dict(orient="records"),"paired_bootstrap":bootrow,"notes":["Perovskites are a ceramic materials family; this tests one materials-property task, not all ceramics or composites.","Formation energies are calculated values, not experimental measurements.","Only one group split was evaluated."]},indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(7.5,4.0))
    names=list(summary.model); means=summary.mean_mae.to_numpy(); sds=summary.sd_mae.fillna(0).to_numpy()
    ax.bar(names,means,yerr=sds,capsize=4,color=["#6d91b4","#dfa243","#769c70"][:len(names)])
    ax.set_ylabel("Test MAE (eV per unit cell)"); ax.set_title("Prediction for unseen reduced formulas"); ax.tick_params(axis="x",labelrotation=12); fig.tight_layout(); fig.savefig(OUT/"split2_fig_results.png",dpi=220); plt.close(fig)
    (OUT/"split2_warnings.csv").write_text("category,message\n",encoding="utf-8")
    print(summary.to_string(index=False),flush=True); print(bootrow,flush=True)

if __name__=="__main__": main()




