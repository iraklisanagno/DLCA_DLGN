"""Two-GPU bounded queue: factorial first, then independent full-S replication."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import os
import subprocess
import sys
import threading
from .common import ROOT, REPO, read, write_json
from .verify import audit_run, verify_freeze


def worker(gpu,entries,stop):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(); torch.ones(1,device="cuda")'],cwd=REPO,env=env,check=True)
    for row in entries:
        if stop.is_set(): return
        stamp=ROOT/'logs'/f"{row['name']}.complete.json"
        if stamp.exists():
            if read(stamp)!=audit_run(row):raise RuntimeError('completed output changed')
            continue
        output=REPO/row['output']
        if output.exists() and any(output.iterdir()):raise RuntimeError('unclassified output; no automatic restart')
        write_json(ROOT/'logs'/f"{row['name']}.started.json",dict(gpu=gpu,parent_pid=os.getpid(),
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),row=row))
        print('GPU',gpu,'starting',row['name'],flush=True)
        with (ROOT/'logs'/f"{row['name']}.log").open('x') as stream:
            try:
                subprocess.run([sys.executable,'-m','experiments.coverage_dlgn.unified_refinement.train_adapter',
                    '--config',str(REPO/row['config'])],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                write_json(stamp,audit_run(row))
            except BaseException:
                stop.set()
                raise
        print('GPU',gpu,'complete',row['name'],flush=True)


def run(phases,gpus):
    if len(set(gpus))!=len(gpus) or not gpus:raise ValueError('distinct nonempty GPU IDs required')
    verify_freeze()
    subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.free','--format=csv'],check=True)
    matrix=read(ROOT/'matrix.json')
    for phase in phases:
        stop=threading.Event()
        entries=[r for r in matrix['entries'] if r['phase']==phase and not r['reuse']]
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,gpu,entries[i::len(gpus)],stop) for i,gpu in enumerate(gpus)]
            for future in futures:future.result()
        from .report import training
        training(phase)
        print('Complete fixed phase:',phase,flush=True)
    print('All requested validation training complete; zero test queries',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phases',nargs='+',choices=['factorial','full_s'],default=['factorial','full_s'])
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1])
    args=parser.parse_args();run(args.phases,args.gpus)
