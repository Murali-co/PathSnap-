#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pathsnap.experiments import export_artifact

def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',default='outputs/experiments/student_test/selection.json')
    p.add_argument('--models',default='outputs/experiments/models');p.add_argument('--prepared',default='outputs/prepared')
    p.add_argument('--artifacts',default='artifacts');a=p.parse_args()
    selection=json.loads((ROOT/a.selection).read_text());name=selection['model']
    print(export_artifact(ROOT/a.models/name/'selected_model.joblib',ROOT/a.artifacts,ROOT/a.prepared,name,selection))
if __name__=='__main__':main()
