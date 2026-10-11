"""Small SMD affected-variable localisation case study; no claim of physical RCA."""
import json
import math
from pathlib import Path
from urllib.request import urlopen
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/data/smd"
OUT = ROOT / "analysis/results"
FIG = ROOT / "analysis/figures"
COMMIT = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"


def download():
    DATA.mkdir(parents=True,exist_ok=True)
    records=[]
    for category in ["train","test","test_label","interpretation_label"]:
        url=f"https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/{COMMIT}/ServerMachineDataset/{category}/machine-1-1.txt"
        path=DATA/f"{category}.txt"
        if not path.exists():
            path.write_bytes(urlopen(url,timeout=90).read())
        data=path.read_bytes()
        records.append(dict(file=str(path.relative_to(ROOT)),url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (DATA/"sources.json").write_text(json.dumps(dict(repository_commit=COMMIT,files=records),indent=2))


def ranking_metrics(ranking, relevant, k=3):
    relevant=set(relevant)
    hit1=float(ranking[0] in relevant)
    hit3=float(bool(set(ranking[:k]) & relevant))
    dcg=sum((int(v in relevant)/np.log2(i+2)) for i,v in enumerate(ranking[:k]))
    ideal=sum(1/np.log2(i+2) for i in range(min(k,len(relevant))))
    return dict(hit1=hit1,hit3=hit3,ndcg3=dcg/ideal,recall3=len(set(ranking[:k]) & relevant)/len(relevant))


def main():
    download()
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    train=np.loadtxt(DATA/"train.txt",delimiter=",")
    test=np.loadtxt(DATA/"test.txt",delimiter=",")
    labels=np.loadtxt(DATA/"test_label.txt",delimiter=",").astype(int)
    assert train.shape[1] == test.shape[1] == 38 and len(test)==len(labels)
    assert np.isfinite(train).all() and np.isfinite(test).all()
    mu=train.mean(axis=0); raw_sd=train.std(axis=0,ddof=1)
    sd=np.maximum(raw_sd,1e-8)
    z=(test-mu)/sd
    np.savez_compressed(OUT/"smd_normalization.npz",mean=mu,std=sd,raw_std=raw_sd)
    events=[]; score_rows=[]; first=None; sensitivity=[]
    coverage=np.zeros(len(test),dtype=bool)
    for event_id,line in enumerate((DATA/"interpretation_label.txt").read_text().splitlines()):
        bounds,features=line.split(":")
        start,end=map(int,bounds.split("-"))
        ids=[int(x) for x in features.split(",")]
        relevant=[x-1 for x in ids]
        assert 0 <= start < end <= len(test) and all(1<=x<=38 for x in ids)
        assert labels[start:end].all(), "Interpretation range contains non-anomalous points."
        assert start==0 or labels[start-1]==0, "Interpretation start does not match binary event start."
        coverage[start:end]=True
        scores=np.abs(z[start:end]).max(axis=0)
        ranking=np.argsort(-scores,kind="stable")
        metrics=ranking_metrics(ranking,relevant)
        # Numerical-floor sensitivity, disclosed separately from the primary result.
        for floor in [1e-8,1e-3,1e-2,1e-1]:
            score_alt=(np.abs(test[start:end]-mu)/np.maximum(raw_sd,floor)).max(axis=0)
            ranking_alt=np.argsort(-score_alt,kind="stable")
            sensitivity.append(dict(event_id=event_id,std_floor=floor,**ranking_metrics(ranking_alt,relevant)))
        binary_end=end
        while binary_end<len(labels) and labels[binary_end]:
            binary_end+=1
        m=len(relevant); p=test.shape[1]
        random_hit3=1-math.comb(p-m,3)/math.comb(p,3) if p-m>=3 else 1.
        ideal=sum(1/np.log2(i+2) for i in range(min(3,m)))
        random_ndcg3=(m/p)*sum(1/np.log2(i+2) for i in range(3))/ideal
        events.append(dict(event_id=event_id,start=start,end_exclusive=end,samples=end-start,
                           binary_end_exclusive=binary_end,affected_dimensions=";".join(map(str,ids)),
                           n_affected=m,top1=int(ranking[0]+1),top3=";".join(str(v+1) for v in ranking[:3]),
                           **metrics,random_hit1=m/p,random_hit3=random_hit3,
                           random_ndcg3=random_ndcg3,random_recall3=3/p))
        for rank,j in enumerate(ranking,1):
            score_rows.append(dict(event_id=event_id,dimension=int(j+1),rank=rank,peak_abs_z=float(scores[j]),affected=j in relevant))
        if first is None:
            first=(start,end,ids,ranking,scores)
    frame=pd.DataFrame(events)
    frame.to_csv(OUT/"smd_events.csv",index=False)
    pd.DataFrame(score_rows).to_csv(OUT/"smd_rankings.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/"smd_floor_sensitivity_events.csv",index=False)
    pd.DataFrame(sensitivity).groupby("std_floor")[["hit1","hit3","ndcg3","recall3"]].mean().to_csv(OUT/"smd_floor_sensitivity.csv")
    table=[]
    for method,prefix in [("Magnitude ranking",""),("Uniform random ranking (exact expectation)","random_")]:
        table.append(dict(method=method,events=len(frame),**{m:float(frame[prefix+m].mean()) for m in ["hit1","hit3","ndcg3","recall3"]}))
    pd.DataFrame(table).to_csv(OUT/"table_7_3.csv",index=False)
    audit=dict(machine="machine-1-1",train_shape=list(train.shape),test_shape=list(test.shape),
               event_count=len(frame),binary_anomalous_points=int(labels.sum()),
               interpreted_anomalous_points=int(coverage.sum()),excluded_uninterpreted_points=int((labels.astype(bool)&~coverage).sum()),
               indexing="zero-based, end-exclusive time slices; one-based dimension IDs mapped to zero-based columns",
               indexing_evidence="All eight starts and seven ends match binary anomaly runs; first interpretation end is 27 samples earlier. Dimension IDs interpreted as one-based; raw IDs retained for audit.",
               constant_training_dimensions=(np.flatnonzero(raw_sd<1e-8)+1).tolist(),std_floor=1e-8,
               task="affected-variable localisation with supplied event intervals; not detection or true RCA",
               event_aggregation="maximum absolute train-standardized z-score within each interpretation interval",
               selection="machine-1-1 selected before scoring; all 8 interpretation intervals; first chronological event shown",
               metrics="Hit@K: at least one labelled dimension in top K; NDCG@3: binary relevance; event-macro averages",
               figure_first_event=dict(start=first[0],end_exclusive=first[1],affected=first[2]))
    (OUT/"smd_audit.json").write_text(json.dumps(audit,indent=2))
    start,end,ids,ranking,scores=first
    plotted=list(ranking[:3])
    fig,(ax,bar)=plt.subplots(2,1,figsize=(8,5.6),layout="constrained",gridspec_kw={"height_ratios":[2,1.25]})
    colors=plt.get_cmap("tab10")(np.linspace(0,1,len(plotted)))
    left=max(0,start-150); right=min(len(test),end+150)
    for j,c in zip(plotted,colors):
        suffix="; labelled" if j+1 in ids else "; unlabelled"
        rank=int(np.where(ranking==j)[0][0])+1
        ax.plot(np.arange(left,right),z[left:right,j],color=c,lw=1.1,label=f"D{j+1} (rank {rank}{suffix})")
    ax.axvspan(start,end,alpha=.12,color="black")
    ax.set_yscale("symlog",linthresh=1)
    ax.set_xlabel("Test sample index (zero-based)"); ax.set_ylabel("Training z-score (symlog)")
    ax.set_title("SMD machine-1-1: first annotated event")
    ax.legend(fontsize=9,ncol=1,loc="upper left")
    top=ranking[:10]
    bar.bar(range(10),scores[top],color=["#285f88" if j+1 in ids else "#a7adb3" for j in top])
    bar.set_yscale("log")
    bar.set_xticks(range(10),[f"D{j+1}" for j in top]); bar.set_ylabel("Peak |z-score| (log scale)")
    bar.set_xlabel("Dimensions ordered by magnitude score (blue = labelled affected)")
    fig.savefig(FIG/"figure_7_8.png",dpi=200)
    fig.savefig(FIG/"figure_7_8.pdf")
    plt.close(fig)
    print(pd.DataFrame(table).to_string(index=False)); print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()
