"""Research-only retrospective prediction analysis."""
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np, pandas as pd, sklearn, scipy, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from config import *
ROOT = Path(__file__).resolve().parent
METRICS = ['AUROC', 'AP', 'Brier', 'LogLoss', 'CalibrationIntercept', 'CalibrationSlope']
T = np.linspace(0.05, 0.3, 26)

def audit(path, out):
    d = pd.read_excel(path)
    assert d[ID].is_unique and (not d[ID].isna().any())
    assert set(d[TARGET]) == {0, 1}
    assert not d.columns.duplicated().any()
    assert set(sum(GROUPS.values(), [])) <= set(d.columns)
    rows = []
    for c in d:
        groups = ';'.join((g for g, cs in GROUPS.items() if c in cs))
        role = 'timing uncertain: sensitivity only' if c in UNCERTAIN else 'source-described early measurement; no individual timestamps' if c in VITAL + LAB + SCORE else 'demographic' if c in DEM else 'continuous ratio comparator' if c == 'TBAR' else 'outcome' if c == TARGET else 'identifier' if c == ID else 'excluded: not whitelisted'
        rows.append(dict(field=c, dtype=str(d[c].dtype), missing=int(d[c].isna().sum()), unique=d[c].nunique(), role=role, groups=groups))
    pd.DataFrame(rows).to_csv(out / 'data_audit.csv', index=False)
    (out / 'cohort.json').write_text(json.dumps(dict(n=len(d), columns=len(d.columns), events=int(d[TARGET].sum()), missing=int(d.isna().sum().sum()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2))
    desc = []
    for c in dict.fromkeys(sum(GROUPS.values(), [])):
        for label, sub in [('all', d), ('survivors', d[d[TARGET] == 0]), ('deaths', d[d[TARGET] == 1])]:
            s = sub[c].dropna()
            if c in CAT + UNCERTAIN:
                for k, v in s.value_counts().items():
                    desc.append(dict(variable=c, group=label, level=k, n=v, percent=100 * v / len(s)))
            else:
                desc.append(dict(variable=c, group=label, n=len(s), mean=s.mean(), sd=s.std(), median=s.median(), q1=s.quantile(0.25), q3=s.quantile(0.75)))
    pd.DataFrame(desc).to_csv(out / 'baseline.csv', index=False)
    return d

def build(g, m):
    cs = GROUPS[g]
    cat = [c for c in cs if c in CAT]
    binary = [c for c in cs if c in UNCERTAIN]
    num = [c for c in cs if c not in cat + binary]
    tr = []
    if num:
        tr.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num))
    if binary:
        tr.append(('binary', SimpleImputer(strategy='most_frequent'), binary))
    if cat:
        tr.append(('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))]), cat))
    if m == 'Ridge':
        clf = LogisticRegression(max_iter=2000)
        grid = {'clf__C': [0.01, 0.1, 1, 10]}
    elif m == 'Lasso':
        clf = LogisticRegression(solver='liblinear', l1_ratio=1.0, max_iter=2000, random_state=SEED)
        grid = {'clf__C': [0.01, 0.1, 1, 10]}
    elif m == 'Logistic':
        clf = LogisticRegression(C=float('inf'), max_iter=3000)
        grid = {}
    elif m == 'RF':
        clf = RandomForestClassifier(n_estimators=40, random_state=SEED, n_jobs=1)
        grid = {'clf__max_depth': [3, 6], 'clf__min_samples_leaf': [5, 15]}
    elif m == 'HistGB':
        clf = HistGradientBoostingClassifier(max_iter=40, early_stopping=False, random_state=SEED)
        grid = {'clf__max_leaf_nodes': [3, 7], 'clf__l2_regularization': [1, 10]}
    return (Pipeline([('pre', ColumnTransformer(tr)), ('clf', clf)]), grid)

def fit(d, g, m, seed):
    p, grid = build(g, m)
    X = d[GROUPS[g]]
    y = d[TARGET]
    if not grid:
        return (p.fit(X, y), {})
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, d[ID]))
    for a, b in folds:
        assert not set(d.iloc[a][ID]) & set(d.iloc[b][ID])
        assert y.iloc[a].nunique() == y.iloc[b].nunique() == 2
    s = GridSearchCV(p, grid, scoring='neg_log_loss', cv=folds, n_jobs=1, error_score='raise').fit(X, y)
    return (s.best_estimator_, s.best_params_)

def pred(p, d, g):
    return p.predict_proba(d[GROUPS[g]])[:, 1]

def metrics(y, p):
    p = np.clip(p, 1e-08, 1 - 1e-08)
    z = special.logit(p)
    inter = minimize_scalar(lambda a: np.mean(np.logaddexp(0, z + a) - y * (z + a)), bounds=(-20, 20), method='bounded').x
    slope = LogisticRegression(C=1000000.0, max_iter=2000).fit(z.reshape(-1, 1), y).coef_[0, 0] if np.std(z) > 1e-09 else np.nan
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p), log_loss(y, p), inter, slope])

def net(y, p):
    return np.array([np.mean((p >= t) & (y == 1)) - np.mean((p >= t) & (y == 0)) * t / (1 - t) for t in T])

def boot(b, d):
    sample = d.iloc[np.random.default_rng(SEED + 10000 + b).integers(len(d), size=len(d))].reset_index(drop=True)
    rows = []
    with threadpool_limits(1):
        for g, m in SPECS:
            p, h = fit(sample, g, m, SEED + b + 10000)
            a = pred(p, sample, g)
            t = pred(p, d, g)
            ys = sample[TARGET].to_numpy()
            y = d[TARGET].to_numpy()
            rows.append((metrics(ys, a), metrics(y, t), net(ys, a) - net(y, t), json.dumps(h)))
    return rows

def plots(fig, s, oof, y):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

    def save(f, n):
        f.tight_layout()
        f.savefig(fig / (n + '.pdf'))
        f.savefig(fig / (n + '.png'), dpi=300)
        plt.close(f)
    a = s[s.metric == 'AUROC']
    f, ax = plt.subplots(figsize=(7, 5))
    v = np.arange(len(a))
    ax.scatter(a.apparent, v, label='Apparent')
    ax.scatter(a.corrected, v, marker='D', label='Optimism corrected')
    ax.set_yticks(v, a.group + ' / ' + a.model)
    ax.invert_yaxis()
    ax.set_xlabel('AUROC')
    ax.legend()
    save(f, 'auc_comparison')
    for kind in ['roc', 'pr', 'calibration', 'dca']:
        f, ax = plt.subplots(figsize=(6, 4.6))
        for k in ['F3_Ridge', 'SOFA_Logistic', 'Simple_Ridge', 'TBAR_Logistic']:
            p = oof[k]
            if kind == 'roc':
                x, z, _ = roc_curve(y, p)
                ax.plot(x, z, label=f'{k} ({roc_auc_score(y, p):.3f})')
            elif kind == 'pr':
                z, x, _ = precision_recall_curve(y, p)
                ax.plot(x, z, label=f'{k} ({average_precision_score(y, p):.3f})')
            elif kind == 'calibration':
                z, x = calibration_curve(y, p, n_bins=5, strategy='quantile')
                ax.plot(x, z, 'o-', label=k)
            else:
                ax.plot(T, net(y, p), label=k)
        if kind in ['roc', 'calibration']:
            ax.plot([0, 1], [0, 1], '--', color='gray')
        if kind == 'pr':
            ax.axhline(y.mean(), ls='--', color='gray')
        if kind == 'dca':
            ax.plot(T, y.mean() - (1 - y.mean()) * T / (1 - T), '--', label='Treat all')
            ax.axhline(0, ls=':', label='Treat none')
        ax.set(xlabel={'roc': 'False positive rate', 'pr': 'Recall', 'calibration': 'Mean predicted risk', 'dca': 'Threshold probability'}[kind], ylabel={'roc': 'True positive rate', 'pr': 'Precision', 'calibration': 'Observed event rate', 'dca': 'Net benefit'}[kind], title='Nested five-fold out-of-fold estimates')
        ax.legend(fontsize=8)
        save(f, kind)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'data/pone.0323330.s001.xlsx')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    B = 1 if a.smoke else a.bootstrap
    assert B > 0
    a.out = a.out or ROOT / ('smoke_test' if a.smoke else 'final_run')
    out = a.out / 'results'
    fig = a.out / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    d = audit(a.data, out)
    y = d[TARGET].to_numpy()
    ap = {}
    an = {}
    oof = {}
    coef = []
    params = []
    (out / 'run_metadata.json').write_text(json.dumps(dict(seed=SEED, bootstrap=B, smoke=a.smoke, python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, specs=SPECS), indent=2))
    with threadpool_limits(1):
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(d, y, d[ID]))
        for g, m in SPECS:
            print('Full and OOF', g, m, flush=True)
            q, h = fit(d, g, m, SEED)
            pr = pred(q, d, g)
            ap[g, m] = metrics(y, pr)
            an[g, m] = net(y, pr)
            params.append(dict(group=g, model=m, parameters=json.dumps(h)))
            if m in ['Ridge', 'Lasso', 'Logistic']:
                clf = q.named_steps['clf']
                names = q.named_steps['pre'].get_feature_names_out()
                coef.extend((dict(group=g, model=m, feature=n, coefficient=v) for n, v in zip(names, clf.coef_[0])))
                coef.append(dict(group=g, model=m, feature='intercept', coefficient=clf.intercept_[0]))
                if (g, m) == ('F3', 'Ridge'):
                    sc = q.named_steps['pre'].named_transformers_['numeric'].named_steps['scale']
                    pd.DataFrame(dict(feature=[c for c in GROUPS[g] if c not in CAT], mean=sc.mean_, sd=sc.scale_)).to_csv(out / 'primary_scaling.csv', index=False)
            z = np.full(len(d), np.nan)
            for i, (tr, te) in enumerate(folds):
                r, _ = fit(d.iloc[tr], g, m, SEED + i + 1)
                z[te] = pred(r, d.iloc[te], g)
            assert np.isfinite(z).all()
            oof[g + '_' + m] = z
    pd.DataFrame(coef).to_csv(out / 'coefficients.csv', index=False)
    pd.DataFrame(params).to_csv(out / 'hyperparameters.csv', index=False)
    print('Bootstrap', B, flush=True)
    records = Parallel(n_jobs=a.jobs, verbose=5)((delayed(boot)(b, d) for b in range(B)))
    summary = []
    raw = []
    dc = []
    for j, (g, m) in enumerate(SPECS):
        tr = np.array([r[j][0] for r in records])
        te = np.array([r[j][1] for r in records])
        opt = tr - te
        corr = ap[g, m] - np.nanmean(opt, axis=0)
        oo = metrics(y, oof[g + '_' + m])
        for k, metric in enumerate(METRICS):
            summary.append(dict(group=g, model=m, metric=metric, apparent=ap[g, m][k], corrected=corr[k], nested_oof=oo[k], mean_optimism=np.nanmean(opt[:, k]), valid_bootstrap=int(np.isfinite(opt[:, k]).sum())))
            for b in range(B):
                raw.append(dict(iteration=b, group=g, model=m, metric=metric, train=tr[b, k], test=te[b, k], parameters=records[b][j][3]))
        cn = an[g, m] - np.mean([r[j][2] for r in records], axis=0)
        for t, aa, c, o in zip(T, an[g, m], cn, net(y, oof[g + '_' + m])):
            dc.append(dict(group=g, model=m, threshold=t, apparent=aa, corrected=c, nested_oof=o))
    s = pd.DataFrame(summary)
    s.to_csv(out / 'performance.csv', index=False)
    pd.DataFrame(raw).to_csv(out / 'bootstrap_metrics.csv', index=False)
    pd.DataFrame(dc).to_csv(out / 'decision_curve.csv', index=False)
    wide = s.pivot(index=['group', 'model'], columns='metric', values='corrected')
    wide.to_latex(out / 'performance_corrected.tex', float_format='%.3f', escape=True)
    plots(fig, s, oof, y)
    print(wide.to_string())
    print('DONE', flush=True)
if __name__ == '__main__':
    main()
