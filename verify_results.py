"""Verify split separation, metric reproducibility and padding behavior."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from experiment import TextNet
r=Path('results_reproduction');s=pd.read_csv(r/'split_manifest.csv')
assert s.groupby('group').split.nunique().max()==1
assert s.groupby(['group','LLM_name']).size().eq(1).all()
assert s.groupby('group').LLM_name.nunique().eq(3).all()
test_ids=set(s[s.split=='test'].row_id)
count=0
for f in sorted(r.glob('*_s42.json')):
 z=json.loads(f.read_text());p=pd.read_csv(r/(f.stem+'_predictions.csv'))
 assert set(p.row_id)==test_ids and len(p)==len(test_ids)
 assert np.isclose(accuracy_score(p.true,p.predicted),z['accuracy'])
 assert np.isclose(f1_score(p.true,p.predicted,average='macro'),z['macro_f1'])
 assert confusion_matrix(p.true,p.predicted).tolist()==z['confusion_matrix']
 if z['mode']=='input':assert p.groupby('group').predicted.nunique().eq(1).all()
 count+=1
assert count==12
for kind in ['CNN','LSTM']:
 torch.manual_seed(42);model=TextNet(kind,20).eval()
 x=torch.tensor([[2,3,4,5,6,0,0,0]]);long=torch.cat([x,torch.zeros((1,7),dtype=torch.long)],dim=1)
 with torch.no_grad():assert torch.allclose(model(x,torch.tensor([5])),model(long,torch.tensor([5])),atol=1e-6)
checks={'experiments_verified':count,'prompt_split_disjoint':True,'balanced_triplets':True,'metrics_recomputed':True,'padding_invariance_CNN_LSTM':True}
(r/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
