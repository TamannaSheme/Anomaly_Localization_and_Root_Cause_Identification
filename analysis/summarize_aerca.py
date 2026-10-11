"""Audit saved official runs and export reproducible per-sequence oracle metrics."""
import inspect
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"external/AERCA"))
from args.linear_args import create_arg_parser
from models.aerca import AERCA
from utils.utils import topk_at_step


def main():
    torch.set_num_threads(1)
    rows=[]; cases=[]
    for seed in [0,1,2]:
        folder=ROOT/f"analysis/results/aerca_seed{seed}"
        log=(ROOT/f"analysis/logs/aerca_seed{seed}.log").read_text()
        assert log.rstrip().endswith("done"),f"Seed {seed} incomplete"
        options=vars(create_arg_parser().parse_args(["--seed",str(seed),"--device","cpu"]))
        params=inspect.signature(AERCA).parameters
        kwargs={k:v for k,v in options.items() if k in params}
        kwargs['data_name']='linear'
        # Constructor uses current cwd for saved_models; isolate its side effect.
        import os
        previous=os.getcwd(); os.chdir(folder)
        try:
            model=AERCA(**kwargs)
        finally:
            os.chdir(previous)
        checkpoints=[p for p in (folder/"saved_models").glob("*.pt") if not p.name.startswith("._")]
        assert len(checkpoints)==1
        model.load_state_dict(torch.load(checkpoints[0],map_location="cpu",weights_only=True)); model.eval()
        stem=checkpoints[0].with_suffix("")
        center=np.load(str(stem)+"_us_mean_encoder.npy")
        scale=np.load(str(stem)+"_us_std_encoder.npy")
        data=folder/"datasets/linear"
        xs=np.load(data/"x_ab_list.npy")[10:]
        labels=np.load(data/"label_list.npy")[10:]
        saved_scores=[]
        for i,(x,label) in enumerate(zip(xs,labels)):
            with torch.no_grad():
                us=model.encoding(x)[0].numpy()[model.window_size:]
            scores=-(us-center)/scale
            lab=label[2*model.window_size:]
            metric=topk_at_step(scores,lab)
            # Independent calculation verifies official metric and temporal alignment.
            anomaly_steps=np.flatnonzero(lab.sum(axis=1)>0)
            assert len(anomaly_steps)==1
            t=int(anomaly_steps[0]); relevant=set(np.flatnonzero(lab[t]))
            rank=np.argsort(scores[t])[::-1]
            ac1=len(set(rank[:1])&relevant)/min(1,len(relevant))
            ac3=len(set(rank[:3])&relevant)/min(3,len(relevant))
            assert np.isclose(ac1,metric[0]) and np.isclose(ac3,metric[2])
            cases.append(dict(seed=seed,case_id=i+10,anomaly_time=t+2*model.window_size,
                              n_sources=len(relevant),sources=";".join(str(j) for j in sorted(relevant)),
                              top1=int(rank[0]),ac1=ac1,ac3=ac3))
            saved_scores.append(scores)
        np.savez_compressed(folder/"oracle_scores.npz",scores=np.stack(saved_scores),labels=labels[:,2:])
        training_log=(folder/"logs/linear.log").read_text()
        epochs=[int(v) for v in re.findall(r'Epoch (\d+)/5000',training_log)]
        best=[int(v) for v in re.findall(r'Saving model at epoch (\d+)',training_log)]
        subset=pd.DataFrame(cases).query("seed == @seed")
        reported1=float(re.search(r'Root cause analysis AC@1: ([\d.]+)',log)[1])
        reported3=float(re.search(r'Root cause analysis AC@3: ([\d.]+)',log)[1])
        assert abs(subset.ac1.mean()-reported1)<1e-5 and abs(subset.ac3.mean()-reported3)<1e-5
        rows.append(dict(seed=seed,test_sequences=len(xs),epochs_run=max(epochs),best_epoch=best[-1],
                         ac1=subset.ac1.mean(),ac3=subset.ac3.mean(),
                         ac_star1=float(re.search(r'Root cause analysis AC\*@1: ([\d.]+)',log)[1]),
                         causal_auroc=float(re.search(r'Causal discovery AUROC: ([\d.]+)',log)[1]),
                         single_source_cases=int((subset.n_sources==1).sum())))
    pd.DataFrame(rows).to_csv(ROOT/"analysis/results/table_7_2.csv",index=False)
    pd.DataFrame(cases).to_csv(ROOT/"analysis/results/aerca_cases.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
