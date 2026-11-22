"""Summarise rebalancing control runs (mean +/- SD over seeds) next to the original
unweighted results in results/torch_<protocol>_<seed>.json."""
import glob, json, numpy as np

CLS = ['N', 'S', 'V', 'F']


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def row(runs):
    def ms(v):
        v = np.array(v)
        return f'{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}'
    out = {'n': len(runs), 'acc': ms([r['accuracy'] * 100 for r in runs]).replace('.000', ''),
           'macroF1': ms([r['macro_F1_NSVF'] for r in runs])}
    for c in CLS:
        out[f'Se_{c}'] = ms([r['per_class'][c]['Se'] for r in runs])
        out[f'PPV_{c}'] = ms([r['per_class'][c]['PPV'] for r in runs])
    return out


rows = []
for p in 'AB':
    for b, pat in [('none (original)', f'results/torch_{p}_*.json'),
                   ('class weighting', f'results/rebalance/ctrl_{p}_weight_*.json'),
                   ('oversampling', f'results/rebalance/ctrl_{p}_oversample_*.json')]:
        runs = load(pat)
        if runs:
            rows.append((p, b, row(runs)))

keys = ['n', 'acc', 'macroF1'] + [f'Se_{c}' for c in CLS] + [f'PPV_{c}' for c in CLS]
print('protocol | balance | ' + ' | '.join(keys))
for p, b, r in rows:
    print(f'{p} | {b} | ' + ' | '.join(str(r[k]) for k in keys))
json.dump([{'protocol': p, 'balance': b, **r} for p, b, r in rows],
          open('results/rebalance/summary.json', 'w'), indent=1)
