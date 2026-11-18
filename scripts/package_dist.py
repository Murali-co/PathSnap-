#!/usr/bin/env python3
"""Package explicit laptop roles. Required artifacts/files cannot be silently omitted."""
import argparse,hashlib,json,zipfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def package(root,files,destination):
    names=sorted(set(files))
    for path in names:
        if not path.is_file():raise FileNotFoundError('Required package file unavailable: '+str(path.relative_to(root)))
    manifest=[dict(file=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in names]
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in names:z.write(p,p.relative_to(root).as_posix())
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    return dict(file=destination.name,bytes=destination.stat().st_size,sha256=sha(destination),files=len(names))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--artifacts',default='artifacts');a=p.parse_args()
    root=Path(a.root).resolve();dist=root/'dist';dist.mkdir(exist_ok=True)
    artifact=root/a.artifacts;registry=json.loads((artifact/'model_registry.json').read_text())
    model=artifact/registry['selected_model_file']
    if model.resolve().parent!=artifact.resolve():raise ValueError('Artifact path escapes artifact directory.')
    if artifact.resolve()!= (root/'artifacts').resolve():raise ValueError('Export production artifacts to artifacts/ before packaging.')
    backend=list((root/'src').rglob('*.py'))+list((root/'backend').rglob('*.py'))
    backend += [root/n for n in ['pyproject.toml','requirements-backend.txt','requirements-selected.txt','requirements-train.txt','.env.example','README.md','scripts/check_environment.py','scripts/setup_backend.sh','scripts/start_backend.sh','scripts/setup_backend.ps1','scripts/start_backend.ps1','docs/BACKEND_LAPTOP_SETUP.md','docs/TWO_LAPTOP_DEMO.md','configs/default.json','configs/deployment.json']]
    backend += [artifact/'model_registry.json',model]
    frontend=list((root/'frontend').rglob('*.py'))+[root/'frontend/requirements.txt']+list((root/'frontend/assets').iterdir())
    frontend += [root/n for n in ['scripts/check_connection.py','scripts/setup_frontend.sh','scripts/start_frontend.sh','scripts/setup_frontend.ps1','scripts/start_frontend.ps1','docs/FRONTEND_LAPTOP_SETUP.md','docs/TWO_LAPTOP_DEMO.md']]
    results=[package(root,backend,dist/'PathSnap_Backend.zip'),package(root,frontend,dist/'PathSnap_Frontend.zip')]
    (dist/'dist_manifest.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
