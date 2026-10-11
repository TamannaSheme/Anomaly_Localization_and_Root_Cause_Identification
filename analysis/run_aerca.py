"""Run unmodified official AERCA linear entry point in isolated result directories."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    args = parser.parse_args()
    repo = ROOT / "external/AERCA"
    commit = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    for seed in args.seeds:
        folder = ROOT / f"analysis/results/aerca_seed{seed}"
        folder.mkdir(parents=True,exist_ok=True)
        log = ROOT / f"analysis/logs/aerca_seed{seed}.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        bootstrap = "import runpy,sys,torch; torch.set_num_threads(1); sys.path.insert(0,sys.argv[1]); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0]+'/main.py',run_name='__main__')"
        command = [sys.executable,"-u","-c",bootstrap,str(repo),"--dataset_name","linear","--device","cpu","--seed",str(seed)]
        start = time.perf_counter()
        with log.open("w") as stream:
            run = subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT)
        metadata = dict(commit=commit, seed=seed, command=command, returncode=run.returncode,
                        runtime_seconds=time.perf_counter()-start, device="cpu", torch_threads=1,
                        protocol="official linear defaults; 5000 maximum epochs; official patience 20; 10 normal training/validation and 100 test sequences")
        (folder / "run_manifest.json").write_text(json.dumps(metadata,indent=2))
        print(f"seed {seed}: exit {run.returncode}, {metadata['runtime_seconds']:.1f}s",flush=True)
        if run.returncode:
            raise RuntimeError(f"AERCA failed; see {log}")

if __name__ == "__main__":
    main()
