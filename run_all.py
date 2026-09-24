"""Run both protocols across five seeds, then summarise as mean +/- SD.

    python run_all.py --data .            # runs everything, then prints tables
    python run_all.py --data . --summary  # summarise existing result files only

Each individual run writes torch_<protocol>_<seed>.json, so the script can be
stopped and restarted; completed runs are skipped.
"""
import argparse, itertools, json, os, subprocess, sys
import numpy as np

CLS = ['N', 'S', 'V', 'F', 'Q']
PROTO = {'A': 'beat-level', 'B': 'inter-patient'}


def summarise(data):
    rows = {}
    for p in 'AB':
        files = [f'{data}/torch_{p}_{s}.json' for s in range(5)]
        got = [json.load(open(f)) for f in files if os.path.exists(f)]
        if not got:
            continue
        rows[p] = got
        print(f'\n== Protocol {p} ({PROTO[p]}) — {len(got)} seed(s), '
              f'{got[0]["n_train"]} train / {got[0]["n_test"]} test beats')
        acc = np.array([g['accuracy'] for g in got]) * 100
        mf1 = np.array([g['macro_F1_NSVF'] for g in got])
        cm0 = np.array(got[0]['confusion'])
        base = cm0.sum(1).max() / cm0.sum() * 100
        print(f'   accuracy        {acc.mean():6.2f} +/- {acc.std(ddof=1) if len(got) > 1 else 0:.2f} %'
              f'   (majority-class baseline {base:.1f} %)')
        print(f'   macro-F1 NSVF   {mf1.mean():6.3f} +/- {mf1.std(ddof=1) if len(got) > 1 else 0:.3f}')
        print(f'   {"class":6s}{"n":>8s}{"Se":>16s}{"PPV":>16s}{"F1":>16s}')
        for c in CLS:
            def ms(k):
                v = np.array([g['per_class'][c][k] for g in got])
                sd = v.std(ddof=1) if len(got) > 1 else 0.0
                return f'{v.mean():.3f} +/- {sd:.3f}'
            print(f'   {c:6s}{got[0]["per_class"][c]["n"]:>8d}{ms("Se"):>16s}'
                  f'{ms("PPV"):>16s}{ms("F1"):>16s}')
    if 'A' in rows and 'B' in rows:
        a = np.mean([g['macro_F1_NSVF'] for g in rows['A']])
        b = np.mean([g['macro_F1_NSVF'] for g in rows['B']])
        print(f'\n== Macro-F1 drop, A -> B: {a:.3f} -> {b:.3f}  ({(a - b) / a * 100:.1f}% relative)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='.')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()

    if not a.summary:
        for p, s in itertools.product('AB', range(a.seeds)):
            out = f'{a.data}/torch_{p}_{s}.json'
            if os.path.exists(out):
                print(f'skip {p} seed {s} (already done)')
                continue
            print(f'--- running protocol {p}, seed {s}', flush=True)
            subprocess.run([sys.executable, 'torch_train.py', '--protocol', p,
                            '--seed', str(s), '--data', a.data], check=True)
    summarise(a.data)


if __name__ == '__main__':
    main()
