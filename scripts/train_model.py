#!/usr/bin/env python3
"""Train a family, run grouped CV and export full-fit evidence. No student-test fitting."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pathsnap.training import MODELS,train_one

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--prepared',default='outputs/prepared')
    p.add_argument('--output',default='outputs/experiments/models');p.add_argument('--folds',type=int,default=3)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--skip-cv',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args();meta=train_one(a.model,ROOT/a.prepared,ROOT/a.output/a.model,a.folds,a.seed,a.device,
                                   skip_cv=a.skip_cv,resume=a.resume,allow_smoke=a.allow_smoke)
    print(meta)
if __name__=='__main__':main()
