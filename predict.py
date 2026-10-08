
import argparse,json
from pathlib import Path
import torch
from experiment import TextNet,encoded
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--text-file',required=True)
a=p.parse_args();c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
if c['mode']!='output':raise ValueError('Use an output-only checkpoint')
m=TextNet(c['kind'],len(c['vocab'])+2);m.load_state_dict(c['state_dict']);m.eval()
x,n=encoded([Path(a.text_file).read_text()],c['vocab'],c['max_len'])
with torch.no_grad():prob=m(x,n).softmax(1)[0].tolist()
print(json.dumps(dict(zip(c['labels'],prob)),indent=2))
print('Closed-set scores, not calibrated probabilities or proof of authorship.')
