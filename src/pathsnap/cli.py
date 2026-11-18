#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from .config import config, data_root

def main():
    p=argparse.ArgumentParser(prog='pathsnap')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('audit'); d.add_argument('--data-root'); d.add_argument('--output',default='outputs/audit')
    f=sub.add_parser('prepare'); f.add_argument('--data-root'); f.add_argument('--output',default='outputs/prepared'); f.add_argument('--config',default='configs/default.json'); f.add_argument('--limit',type=int)
    b=sub.add_parser('benchmark'); b.add_argument('--prepared',default='outputs/prepared'); b.add_argument('--output',default='outputs/experiments/baseline'); b.add_argument('--model',choices=['logistic','decision_tree','random_forest','xgboost','pytorch_mlp'])
    m=sub.add_parser('serve'); m.add_argument('--host',default='0.0.0.0'); m.add_argument('--port',type=int,default=8000)
    args=p.parse_args()
    if args.command=='audit':
        from .preparation import audit
        root=data_root(args.data_root); print(json.dumps(audit(root,args.output),indent=2))
    elif args.command=='prepare':
        from .preparation import prepare
        root=data_root(args.data_root); cfg=config(args.config); prepare(root,Path(args.output),cfg,resume=True,limit=args.limit)
    elif args.command=='benchmark':
        from .training import benchmark,load_prepared
        out=Path(args.output); out.mkdir(parents=True,exist_ok=True); manifest,rows=load_prepared(args.prepared)
        rows_cv,oof=benchmark(out,rows,folds=manifest['config'].get('folds',3),model_filter=args.model)
        print(json.dumps({'models':sorted(set(r['model'] for r in rows_cv)),'records':len(rows_cv)},indent=2))
    elif args.command=='serve':
        import uvicorn
        uvicorn.run('backend.app.main:app',host=args.host,port=args.port,reload=False)
if __name__=='__main__':main()
