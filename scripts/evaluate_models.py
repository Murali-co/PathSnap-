#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pathsnap.experiments import compare

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',default='outputs/prepared');p.add_argument('--models',default='outputs/experiments/models')
    p.add_argument('--output',default='outputs/experiments/student_test');a=p.parse_args()
    for row in compare(ROOT/a.prepared,ROOT/a.models,ROOT/a.output):print(row)
if __name__=='__main__':main()
