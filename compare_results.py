"""Create matched point comparisons from completed aggregate results.
No significance testing or confidence interval is implied by these differences.
"""
import argparse
from pathlib import Path
import pandas as pd

COMPARISONS = {
    'RQ1': [('F2','Ridge','F1','Ridge'), ('F3','Ridge','F2','Ridge'), ('F4','Ridge','F3','Ridge')],
    'RQ2': [('F3',m,'F3','Ridge') for m in ['Lasso','Logistic','RF','HistGB']],
    'RQ3': [('F3','Ridge',g,m) for g,m in [('SOFA','Logistic'),('Simple','Ridge'),('TBAR','Logistic')]],
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    a=parser.parse_args()
    out=a.run/'results'
    data=pd.read_csv(out/'performance.csv').set_index(['group','model','metric'])
    rows=[]
    for rq, pairs in COMPARISONS.items():
        for ga,ma,gb,mb in pairs:
            for metric in ['AUROC','AP','Brier','LogLoss']:
                x,y=data.loc[(ga,ma,metric)],data.loc[(gb,mb,metric)]
                rows.append(dict(rq=rq,model_a=f'{ga}_{ma}',model_b=f'{gb}_{mb}',metric=metric,delta_oof=x.nested_oof-y.nested_oof,delta_corrected=x.corrected-y.corrected))
    result=pd.DataFrame(rows)
    result.to_csv(out/'paired_point_differences.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':
    main()
