"""Supplementary analysis: preserve all V2 results."""
import argparse, hashlib, json, platform, os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedGroupKFold
from analysis_pipeline import ROOT, GROUPS, SPECS, SEED, ID, TARGET, METRICS, fit, pred, metrics, plt
from compare_results import COMPARISONS

GROUPS['F3_noRace'] = [c for c in GROUPS['F3'] if c != 'Race']
ALL = list(SPECS) + [('F3_noRace','Ridge')]
F3 = [('F3',m) for m in ['Ridge','Lasso','Logistic','RF','HistGB']]
PAIRS = [(rq,(ga,ma),(gb,mb)) for rq,pairs in COMPARISONS.items() for ga,ma,gb,mb in pairs] + [('Sensitivity',('F3_noRace','Ridge'),('F3','Ridge'))]

def nested(d,seed):
    import analysis_pipeline as pipeline
    pipeline.GROUPS["F3_noRace"] = [c for c in pipeline.GROUPS["F3"] if c != "Race"]
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(d,d[TARGET],d[ID])); y=d[TARGET].to_numpy(); rows=[]
    for tr,te in splits:
        assert not set(d.iloc[tr][ID]) & set(d.iloc[te][ID])
        assert d.iloc[tr][TARGET].nunique()==d.iloc[te][TARGET].nunique()==2
    with threadpool_limits(1):
        for g,m in ALL:
            p=np.full(len(d),np.nan)
            for i,(tr,te) in enumerate(splits):
                model,_=fit(d.iloc[tr],g,m,seed+i+1)
                p[te]=pred(model,d.iloc[te],g)
            assert np.isfinite(p).all()
            rows.extend(dict(group=g,model=m,metric=k,value=float(v)) for k,v in zip(METRICS,metrics(y,p)))
    return rows

def sampled(d,seed):
    idx=np.random.default_rng(seed).integers(len(d),size=len(d))
    return d.iloc[idx].reset_index(drop=True),np.setdiff1d(np.arange(len(d)),np.unique(idx))

def oob(d,b):
    sample,idx=sampled(d,SEED+10000+b); test=d.iloc[idx]; rows=[]
    assert not set(sample[ID]) & set(test[ID])
    with threadpool_limits(1):
        for g,m in F3:
            model,_=fit(sample,g,m,SEED+10000+b); p=pred(model,test,g)
            rows.extend(dict(group=g,model=m,metric=k,value=float(v),n_oob=len(test),events_oob=int(test[TARGET].sum())) for k,v in zip(METRICS,metrics(test[TARGET].to_numpy(),p)))
    return rows

def work(kind,b,d,cache):
    path=cache/f'{kind}_{b:05d}.csv'
    if path.exists(): return str(path)
    if kind=='repeat': rows=nested(d,SEED+b)
    elif kind=='outer': rows=nested(sampled(d,SEED+100000+b)[0],SEED)
    else: rows=oob(d,b)
    result=pd.DataFrame(rows); result.insert(0,'iteration',b)
    temp=path.with_suffix('.tmp'); result.to_csv(temp,index=False); os.replace(temp,path)
    return str(path)

def run(kind,n,d,cache,jobs):
    files=Parallel(n_jobs=jobs,verbose=10)(delayed(work)(kind,b,d,cache) for b in range(n))
    out=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    assert out.iteration.nunique()==n and not out.duplicated(['iteration','group','model','metric']).any()
    assert len(out)==n*len(METRICS)*(len(F3) if kind=='oob' else len(ALL))
    return out

def summarize(raw,point,kind):
    rows=[]
    for (g,m,k),s in raw.groupby(['group','model','metric']):
        v=s.value.dropna(); q=v.quantile([.025,.975])
        rows.append(dict(group=g,model=m,metric=k,reference=point.get((g,m,k),np.nan),mean=v.mean(),sd=v.std(),lower=q.iloc[0],upper=q.iloc[1],valid=len(v),interval_type=kind))
    return pd.DataFrame(rows)

def paired(raw,point,kind):
    wide=raw.pivot(index='iteration',columns=['group','model','metric'],values='value'); rows=[]
    for rq,a,b in PAIRS:
        for k in METRICS[:4]:
            x=(*a,k); y=(*b,k); v=(wide[x]-wide[y]).dropna(); q=v.quantile([.025,.975])
            rows.append(dict(rq=rq,model_a='_'.join(a),model_b='_'.join(b),metric=k,reference_delta=point[x]-point[y],mean_delta=v.mean(),lower=q.iloc[0],upper=q.iloc[1],valid=len(v),interval_type=kind))
    return pd.DataFrame(rows)

def brier632(ap,oob_error,gamma):
    if gamma<=ap: return np.nan,np.nan,np.nan
    r=float(np.clip((oob_error-ap)/(gamma-ap),0,1)); w=.632/(1-.368*r)
    return (1-w)*ap+w*oob_error,r,w

def sensitivity(d,raw,ref):
    rows=[]; y=d[TARGET].to_numpy()
    with threadpool_limits(1):
        for g,m in F3:
            model,_=fit(d,g,m,SEED); p=pred(model,d,g)
            ap=float(np.mean((y-p)**2)); gamma=float(y.mean()*np.mean((1-p)**2)+(1-y.mean())*np.mean(p**2))
            r=raw.query('group==@g and model==@m and metric=="Brier"')
            err=float(np.average(r.value,weights=r.n_oob)); est,rel,w=brier632(ap,err,gamma)
            auc=float(raw.query('group==@g and model==@m and metric=="AUROC"').value.mean())
            rows.append(dict(group=g,model=m,apparent_brier=ap,oob_brier=err,no_information_brier=gamma,relative_overfit=rel,weight=w,brier_632plus_exploratory=est,mean_oob_auc=auc,v2_corrected_auc=ref.loc[(g,m,'AUROC'),'corrected'],v2_nested_auc=ref.loc[(g,m,'AUROC'),'nested_oof']))
    return pd.DataFrame(rows)

def plots(out,repeat,outer,sens,ref):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    labels=[m for _,m in F3]; colors=['#476b94','#7b8a9b','#b5a080','#287c76']
    f,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(len(labels))
    for j,(name,values) in enumerate([('Apparent',[ref.loc[('F3',m,'AUROC'),'apparent'] for m in labels]),('Optimism corrected',sens.v2_corrected_auc),('Mean OOB',sens.mean_oob_auc),('Repeated nested CV',[repeat.query('group=="F3" and model==@m and metric=="AUROC"').value.mean() for m in labels])]):
        ax.scatter(x+(j-1.5)*.12,values,label=name,color=colors[j],s=45)
    ax.set(xticks=x,xticklabels=labels,ylabel='AUROC',ylim=(.5,1.025)); ax.legend(ncol=2,frameon=False); f.tight_layout(); f.savefig(out/'validation_strategies.pdf'); f.savefig(out/'validation_strategies.png',dpi=400); plt.close(f)
    f,ax=plt.subplots(figsize=(8,5)); q=outer.query('metric=="AUROC"').reset_index(drop=True); pos=np.arange(len(q))
    ax.hlines(pos,q.lower,q.upper,color='#476b94'); ax.scatter(q.reference,pos,color='#287c76',label='Single nested CV'); ax.scatter(q['mean'],pos,marker='x',color='#476b94',label='Bootstrap mean'); ax.set(yticks=pos,yticklabels=q.group+' / '+q.model,xlabel='AUROC'); ax.invert_yaxis(); ax.legend(frameon=False); f.tight_layout(); f.savefig(out/'outer_bootstrap_intervals.pdf'); f.savefig(out/'outer_bootstrap_intervals.png',dpi=400); plt.close(f)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',type=Path,default=ROOT/'data/pone.0323330.s001.xlsx'); p.add_argument('--v2',type=Path,default=ROOT/'final_run'); p.add_argument('--out',type=Path,default=ROOT/'v3_final'); p.add_argument('--repeats',type=int,default=30); p.add_argument('--outer',type=int,default=500); p.add_argument('--oob',type=int,default=1000); p.add_argument('--jobs',type=int,default=1); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if a.smoke: a.repeats=a.outer=a.oob=2
    assert min(a.repeats,a.outer,a.oob,a.jobs)>0
    d=pd.read_excel(a.data); digest=hashlib.sha256(a.data.read_bytes()).hexdigest(); assert d[ID].is_unique and not d[ID].isna().any() and set(d[TARGET])=={0,1}
    old=json.loads((a.v2/'results/cohort.json').read_text()); assert old['sha256']==digest, 'V2 and V3 datasets differ'
    ref=pd.read_csv(a.v2/'results/performance.csv').set_index(['group','model','metric'])
    a.out.mkdir(parents=True,exist_ok=True); cache=a.out/'checkpoints'; cache.mkdir(exist_ok=True); results=a.out/'results'; results.mkdir(exist_ok=True); figures=a.out/'figures'; figures.mkdir(exist_ok=True)
    meta=dict(data_sha256=digest,seed=SEED,repeats=a.repeats,outer=a.outer,oob=a.oob,smoke=a.smoke,python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,v2_sha256=hashlib.sha256((a.v2/'results/performance.csv').read_bytes()).hexdigest(),code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['run_v3.py','analysis_pipeline.py','config.py','compare_results.py']})
    mp=a.out/'metadata.json'
    if mp.exists(): assert json.loads(mp.read_text())==meta,'Output configuration changed; use new --out'
    else: mp.write_text(json.dumps(meta,indent=2))
    print('Repeated nested CV',flush=True); repeat=run('repeat',a.repeats,d,cache,a.jobs)
    first=repeat[repeat.iteration==0].set_index(['group','model','metric']).value
    diff=(first.reindex(ref.index)-ref.nested_oof).abs(); print('V2 reproduction max absolute differences:',diff.groupby(level='metric').max().to_dict(),flush=True)
    assert np.allclose(first.reindex(ref.index),ref.nested_oof,atol=1e-4,rtol=1e-4,equal_nan=True),'V2 nested CV differs; investigate software or data'
    repeat.to_csv(results/'repeated_nested_raw.csv',index=False)
    summarize(repeat,first,'split variability quantiles; not confidence intervals').to_csv(results/'repeated_nested_summary.csv',index=False)
    paired(repeat,first,'split variability quantiles; not confidence intervals').to_csv(results/'paired_repeat.csv',index=False)
    print('Outer patient bootstrap with complete nested CV refitting',flush=True); outer=run('outer',a.outer,d,cache,a.jobs)
    outer.to_csv(results/'outer_bootstrap_raw.csv',index=False); osum=summarize(outer,first,'exploratory percentile interval for nested CV statistic'); osum.to_csv(results/'outer_bootstrap_summary.csv',index=False)
    paired(outer,first,'exploratory paired percentile interval; unadjusted').to_csv(results/'paired_outer_bootstrap.csv',index=False)
    print('OOB refits',flush=True); raw=run('oob',a.oob,d,cache,a.jobs); raw.to_csv(results/'oob_raw.csv',index=False)
    summarize(raw,first,'OOB resample variability; not confidence intervals').to_csv(results/'oob_summary.csv',index=False)
    sens=sensitivity(d,raw,ref); sens.to_csv(results/'validation_sensitivity.csv',index=False); plots(figures,repeat,osum,sens,ref)
    print('V3 DONE',flush=True)
if __name__=='__main__': main()
