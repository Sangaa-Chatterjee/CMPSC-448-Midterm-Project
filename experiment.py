
import argparse, copy, hashlib, json, platform, random, re, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LABELS = ['ChatGPT', 'Claude', 'Gemini']
def norm(s): return re.sub(r'\s+', ' ', str(s)).strip().casefold()
def tokens(s): return re.findall(r"\w+|[^\w\s]", s.casefold())
def clean_format(s): return ' '.join(re.findall(r'\w+',s.casefold()))
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def prepare(path, out):
    # The published CSV is not valid UTF-8. Latin-1 preserves every source byte.
    raw=pd.read_csv(path,encoding='latin1')
    d=raw.rename(columns={'Question':'LLM_Input','Answer':'LLM_output','Source':'LLM_name'})
    d=d[d.LLM_name.isin(LABELS)].copy(); counts={'raw_total':len(raw),'selected':len(d)}
    d=d.dropna(subset=['LLM_Input','LLM_output'])
    d=d[(d.LLM_Input.str.strip().str.len()>0)&(d.LLM_output.str.strip().str.len()>0)]
    # Conservative automated screen, not a guarantee of privacy/copyright status.
    private=r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{3}-\d{2}-\d{4}\b'
    hit=d.LLM_Input.str.contains(private,regex=True)|d.LLM_output.str.contains(private,regex=True)
    counts['privacy_pattern_removed']=int(hit.sum());d=d[~hit]
    d['prompt_key']=d.LLM_Input.map(norm)
    d['output_key']=d.LLM_output.map(norm)
    # Exclude any repeated answer text entirely to avoid answer overlap across splits.
    duplicate=d.output_key.duplicated(keep=False)
    counts['duplicate_answer_rows_removed']=int(duplicate.sum());d=d[~duplicate]
    d=d.drop_duplicates(['prompt_key','LLM_name'])
    # Complete prompt triplets eliminate prompt-label imbalance by construction.
    complete=d.groupby('prompt_key').LLM_name.nunique()
    d=d[d.prompt_key.isin(complete[complete==3].index)].copy()
    d=d.sort_values(['prompt_key','LLM_name']).reset_index(drop=True)
    d['group']=d.prompt_key.map(sha)
    d['row_id']=[sha(a+'\0'+b+'\0'+c) for a,b,c in zip(d.LLM_name,d.LLM_Input,d.LLM_output)]
    d['y']=d.LLM_name.map({v:i for i,v in enumerate(LABELS)})
    assert len(d)>300,'Too few complete triplets'
    split=GroupShuffleSplit(n_splits=1,test_size=.30,random_state=42)
    train,hold=next(split.split(d,groups=d.group))
    val,test=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=42).split(d.iloc[hold],groups=d.iloc[hold].group))
    d['split']='train';d.loc[hold[val],'split']='validation';d.loc[hold[test],'split']='test'
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert set(d[d.split==a].group).isdisjoint(d[d.split==b].group)
        assert set(d[d.split==a].output_key).isdisjoint(d[d.split==b].output_key)
    counts['retained']=len(d);counts['prompt_groups']=d.group.nunique()
    counts['split_counts']=pd.crosstab(d.split,d.LLM_name).to_dict(orient='index')
    counts['source_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    counts['identity_mentions']=int(d.LLM_output.str.contains(r'chatgpt|openai|anthropic|claude|gemini|google',case=False).sum())
    (out/'data_audit.json').write_text(json.dumps(counts,indent=2))
    d[['row_id','group','LLM_name','split']].to_csv(out/'split_manifest.csv',index=False)
    return d

class TextNet(nn.Module):
    def __init__(self,kind,vocab_size):
        super().__init__();self.kind=kind
        self.embedding=nn.Embedding(vocab_size,64,padding_idx=0)
        if kind=='CNN':
            self.convs=nn.ModuleList([nn.Conv1d(64,64,k) for k in (3,4,5)])
            self.head=nn.Linear(192,3)
        else:
            self.rnn=nn.LSTM(64,64,batch_first=True,bidirectional=True)
            self.head=nn.Linear(128,3)
        self.dropout=nn.Dropout(.3)
    def forward(self,x,length):
        e=self.embedding(x)
        if self.kind=='CNN':
            feats=[]
            for c in self.convs:
                z=torch.relu(c(e.transpose(1,2)))
                valid=(length-c.kernel_size[0]+1).clamp(min=1)
                mask=torch.arange(z.shape[-1],device=z.device)[None,:]>=valid[:,None]
                z=z.masked_fill(mask[:,None,:],-1e9)
                feats.append(z.max(-1).values)
            h=torch.cat(feats,dim=1)
        else:
            packed=nn.utils.rnn.pack_padded_sequence(e,length.cpu(),batch_first=True,enforce_sorted=False)
            _,(h,_)=self.rnn(packed);h=torch.cat([h[-2],h[-1]],dim=1)
        return self.head(self.dropout(h))

def encoded(texts,vocab,max_len):
    seq=[[vocab.get(t,1) for t in tokens(s)[:max_len]] or [1] for s in texts]
    x=torch.zeros((len(seq),max_len),dtype=torch.long)
    for i,s in enumerate(seq):x[i,:len(s)]=torch.tensor(s)
    return x,torch.tensor([len(s) for s in seq])

def stats(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
            'per_class':classification_report(y,p,labels=range(3),target_names=LABELS,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,p,labels=range(3)).tolist()}

def predict(model,loader):
    model.eval();ys=[];ps=[]
    with torch.no_grad():
        for x,n,y in loader:ys.extend(y.tolist());ps.extend(model(x,n).argmax(1).tolist())
    return np.array(ys),np.array(ps)

def run(d,kind,mode,seed,args):
    seed_all(seed);out=Path(args.out);name=f'{kind}_{mode}_s{seed}'
    if mode=='input': texts=d.LLM_Input.tolist()
    elif mode=='both':
        # Reserve half the fixed budget for each side, so a long prompt cannot erase the answer.
        texts=[' '.join(tokens(a)[:args.max_len//2-1])+ ' SEP '+ ' '.join(tokens(b)[:args.max_len//2]) for a,b in zip(d.LLM_Input,d.LLM_output)]
    elif mode=='output_half':texts=[' '.join(tokens(s)[:args.max_len//2]) for s in d.LLM_output]
    elif mode=='plain':texts=d.LLM_output.map(clean_format).tolist()
    elif mode=='masked':texts=d.LLM_output.str.replace(r'chatgpt|openai|anthropic|claude|gemini|google','MODEL',case=False,regex=True).tolist()
    else:texts=d.LLM_output.tolist()
    train_idx=np.flatnonzero(d.split=='train');val_idx=np.flatnonzero(d.split=='validation');test_idx=np.flatnonzero(d.split=='test')
    counter=Counter(t for i in train_idx for t in tokens(texts[i]))
    vocab={w:i+2 for i,(w,c) in enumerate(counter.most_common(19998))}
    x,n=encoded(texts,vocab,args.max_len);y=torch.tensor(d.y.to_numpy())
    loaders={}
    for label,ix in [('train',train_idx),('val',val_idx),('test',test_idx)]:
        loaders[label]=DataLoader(TensorDataset(x[ix],n[ix],y[ix]),batch_size=64,shuffle=label=='train')
    model=TextNet(kind,len(vocab)+2);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    history=[];best=-1;state=None;start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train();total=0
        for bx,bn,by in loaders['train']:
            opt.zero_grad();loss=nn.functional.cross_entropy(model(bx,bn),by);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(by)
        vy,vp=predict(model,loaders['val']);score=f1_score(vy,vp,average='macro',zero_division=0)
        history.append({'epoch':epoch,'train_loss':total/len(train_idx),'val_macro_f1':float(score)})
        if score>best:best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch
        print(name,history[-1],flush=True)
    model.load_state_dict(state);ty,tp=predict(model,loaders['test'])
    result={'name':name,'model':kind,'mode':mode,'seed':seed,'best_epoch':best_epoch,'val_macro_f1':float(best),
            'seconds':round(time.time()-start,2),'parameters':sum(p.numel() for p in model.parameters()),'history':history,**stats(ty,tp)}
    # Group bootstrap preserves dependence among the three answers to each prompt.
    rng=np.random.default_rng(123);groups=d.iloc[test_idx].group.to_numpy();unique=np.unique(groups)
    group_indices=[np.flatnonzero(groups==g) for g in unique];scores=[]
    for _ in range(1000):
        ix=np.concatenate([group_indices[i] for i in rng.integers(len(unique),size=len(unique))])
        scores.append(accuracy_score(ty[ix],tp[ix]))
    result['accuracy_ci95_group_bootstrap']=np.quantile(scores,[.025,.975]).tolist()
    pd.DataFrame({'row_id':d.iloc[test_idx].row_id.to_numpy(),'group':groups,'true':ty,'predicted':tp}).to_csv(out/f'{name}_predictions.csv',index=False)
    (out/f'{name}.json').write_text(json.dumps(result,indent=2))
    torch.save({'state_dict':state,'vocab':vocab,'kind':kind,'mode':mode,'max_len':args.max_len,'labels':LABELS},out/f'{name}.pt')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/txd22.zip');p.add_argument('--out',default='results')
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--max-len',type=int,default=128)
    p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=prepare(args.data,out)
    metadata={**vars(args),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'device':'CPU'}
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2))
    summaries=[]
    for seed in args.seeds:
        for kind in ['CNN','LSTM']:
            for mode in ['input','output','both','output_half','plain','masked']:
                target=out/f'{kind}_{mode}_s{seed}.json'
                r=run(d,kind,mode,seed,args)
                summaries.append({k:r[k] for k in ['name','model','mode','seed','best_epoch','val_macro_f1','accuracy','macro_f1','seconds']})
                pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    # Optional simple comparison; never fit vocabulary on held-out text.
    tr=d.split=='train';te=d.split=='test'
    v=TfidfVectorizer(max_features=20000,ngram_range=(1,2));a=v.fit_transform(d.loc[tr,'LLM_output']);b=v.transform(d.loc[te,'LLM_output'])
    clf=LogisticRegression(max_iter=1000,random_state=42).fit(a,d.loc[tr,'y']);pred=clf.predict(b)
    (out/'tfidf_baseline.json').write_text(json.dumps(stats(d.loc[te,'y'],pred),indent=2))
    features=[]
    for label,g in d[te].groupby('LLM_name'):
        s=g.LLM_output;words=s.map(lambda x:re.findall(r'\w+',x.casefold()))
        features.append({'family':label,'n':len(g),'mean_words':float(words.map(len).mean()),'mean_type_token_ratio':float(words.map(lambda w:len(set(w))/max(1,len(w))).mean()),'mean_newlines':float(s.str.count('\n').mean()),'fraction_bullets':float(s.str.contains(r'(?m)^\s*[-*•]|^\s*\d+[.)]').mean())})
    pd.DataFrame(features).to_csv(out/'style_features.csv',index=False)
    print('Finished',flush=True)
if __name__=='__main__':main()
