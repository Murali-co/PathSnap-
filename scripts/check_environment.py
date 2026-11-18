#!/usr/bin/env python3
"""Check the selected laptop role; --require-gpu executes both library probes."""
import argparse,importlib.util,json,platform,shutil,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--role',choices=['backend','frontend','training'],default='backend');p.add_argument('--require-gpu',action='store_true');a=p.parse_args()
required={'backend':['numpy','scipy','sklearn','shapely','pyproj','fastapi','uvicorn','joblib','xgboost'],
          'frontend':['streamlit','requests','folium','streamlit_folium'],
          'training':['numpy','pandas','scipy','sklearn','shapely','pyproj','torch','xgboost','optuna','pytest','matplotlib','reportlab']}[a.role]
missing=[m for m in required if importlib.util.find_spec(m) is None]
print('Python:',platform.python_version(),sys.executable);print('Platform:',platform.platform());print('Role:',a.role)
if missing:raise SystemExit('Missing dependencies: '+', '.join(missing)+'. Run the setup script for this role.')
if a.role=='training' or a.require_gpu:
    import torch,xgboost as xgb,numpy as np
    available=torch.cuda.is_available();print('PyTorch CUDA:',available)
    if available:
        value=(torch.ones(16,device='cuda')*2).sum().item();assert value==32
        print('GPU:',torch.cuda.get_device_name(0))
        model=xgb.XGBClassifier(n_estimators=2,max_depth=2,device='cuda',tree_method='hist')
        model.fit(np.array([[0],[1],[2],[3]],dtype=np.float32),np.array([0,0,1,1]))
        device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        print('XGBoost execution device:',device)
        if a.require_gpu and not device.startswith('cuda'):raise SystemExit('XGBoost GPU probe fell back to CPU.')
    elif a.require_gpu:raise SystemExit('GPU requested but CUDA is unavailable. Fix the NVIDIA driver/PyTorch installation before GPU training.')
print('Environment check passed')
