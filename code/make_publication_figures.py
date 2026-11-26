from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]; R=P/'results'; F=P/'figures'; T=P/'tables'
r=pd.read_csv(R/'repeated_nested_summary.csv'); o=pd.read_csv(R/'outer_bootstrap_summary.csv'); p=pd.read_csv(R/'paired_outer_bootstrap.csv'); v=pd.read_csv(R/'performance.csv'); s=pd.read_csv(R/'validation_sensitivity.csv'); d=pd.read_csv(R/'paired_repeat.csv')
assert set(r.valid)=={30} and set(o.valid)=={500} and set(v.valid_bootstrap)=={1000}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'figure.dpi':160})
colors=['#244c66','#167c80','#ba6b37','#777f89']
labels={'Ridge':'Ridge','Lasso':'LASSO','Logistic':'Logistic','RF':'Random forest','HistGB':'Hist. boosting'}
def save(fig,name):
 fig.tight_layout(); fig.savefig(F/(name+'.pdf'),bbox_inches='tight'); fig.savefig(F/(name+'.png'),dpi=400,bbox_inches='tight'); plt.close(fig)
a=r.query('metric=="AUROC" and model=="Ridge" and group in ["F1","F2","F3","F4"]').set_index('group').loc[['F1','F2','F3','F4']]; fig,ax=plt.subplots(figsize=(7,3.5)); x=np.arange(4); ax.errorbar(x,a['mean'],yerr=[a['mean']-a.lower,a.upper-a['mean']],fmt='o-',capsize=4,color=colors[1]); ax.set(xticks=x,xticklabels=['F1\nDemographics','F2\n+ Measurements','F3\n+ Scores','F4\n+ Diagnoses'],ylabel='Mean AUROC',ylim=(.60,.85)); ax.grid(axis='y',alpha=.15); save(fig,'feature_groups')
m=['Ridge','Lasso','Logistic','RF','HistGB']; a=r.query('group=="F3" and metric=="AUROC"').set_index('model').loc[m]; q=v.query('group=="F3" and metric=="AUROC"').set_index('model').loc[m]; z=s.set_index('model').loc[m]; fig,ax=plt.subplots(figsize=(7.5,4)); x=np.arange(5)
for i,(lab,y) in enumerate([('Apparent',q.apparent),('Optimism corrected',q.corrected),('Repeated nested CV',a['mean']),('OOB',z.mean_oob_auc)]): ax.scatter(x+(i-1.5)*.13,y,s=48,label=lab,color=colors[i],marker=['o','s','D','^'][i])
ax.set(xticks=x,xticklabels=[labels[k] for k in m],ylabel='AUROC',ylim=(.70,1.025)); ax.grid(axis='y',alpha=.18); ax.legend(ncol=2,frameon=False,loc='upper left'); save(fig,'validation_comparison')
a=p.query('metric=="AUROC"').copy(); names=['F2 vs F1','F3 vs F2','F4 vs F3','LASSO vs Ridge','Logistic vs Ridge','RF vs Ridge','HistGB vs Ridge','F3 vs SOFA','F3 vs 3-variable','F3 vs TBAR','No race vs F3']; fig,ax=plt.subplots(figsize=(7.5,5.3)); y=np.arange(len(a)); ax.hlines(y,a.lower,a.upper,color=colors[0]); ax.scatter(a.mean_delta,y,color=colors[0],label='Outer bootstrap mean'); rr=d.query('metric=="AUROC"'); ax.scatter(rr.mean_delta,y,marker='D',facecolors='white',edgecolors=colors[2],label='Repeated CV mean'); ax.axvline(0,ls='--',color='gray'); ax.set(yticks=y,yticklabels=names,xlabel='AUROC difference (A − B)'); ax.invert_yaxis(); ax.legend(frameon=False,loc='lower right',fontsize=8); save(fig,'paired_differences')
# Complete tabular reporting, directly generated from aggregate results.
def f(x): return f'{x:.3f}'
order=[('F1','Ridge'),('F2','Ridge'),('F3','Ridge'),('F4','Ridge'),('F3','Lasso'),('F3','Logistic'),('F3','RF'),('F3','HistGB'),('SOFA','Logistic'),('Simple','Ridge'),('TBAR','Logistic'),('F3_noRace','Ridge')]
wide=r.pivot(index=['group','model'],columns='metric',values='mean'); lines=[]
for g,m0 in order:
 row=wide.loc[(g,m0)]; lines.append(' & '.join([g.replace('_','\\_'),labels[m0]]+[f(row[k]) for k in ['AUROC','AP','Brier','LogLoss','CalibrationIntercept','CalibrationSlope']])+r' \\')
(T/'performance.tex').write_text('\n'.join(lines))
lines=[]
for (_,row),name,(_,rep) in zip(a.iterrows(),names,rr.iterrows()): lines.append(' & '.join([name,f(rep.mean_delta),f(row.mean_delta),f'[{f(row.lower)}, {f(row.upper)}]'])+r' \\')
(T/'paired.tex').write_text('\n'.join(lines))
b=pd.read_csv(R/'baseline.csv'); lines=[]
for k,lab in [('Age','Age (years)'),('Bun','BUN'),('SOFA','SOFA'),('GCS','GCS'),('Albumin','Albumin'),('Total-Bilirubin','Total bilirubin'),('creatinine','Creatinine'),('spo2','Oxygen saturation')]:
 vals=b[b.variable.eq(k)].set_index('group'); lines.append(' & '.join([lab]+[f"{vals.loc[g,'median']:.1f} ({vals.loc[g,'q1']:.1f}--{vals.loc[g,'q3']:.1f})" for g in ['all','survivors','deaths']])+r' \\')
(T/'baseline.tex').write_text('\n'.join(lines))
