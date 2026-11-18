#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pathsnap.preparation import audit

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--output',default='outputs/audit'); args=p.parse_args()
    root=Path(args.data_root).expanduser().resolve()
    if not (root/'train').is_dir() or not (root/'student_test').is_dir():
        raise SystemExit('Data root must contain train/ and student_test/')
    print(audit(root,args.output))
if __name__=='__main__': main()
