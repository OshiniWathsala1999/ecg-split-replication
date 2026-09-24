"""Build a beat-image dataset from raw MIT-BIH signals, keeping record IDs.

Each beat is rendered as a thin black trace on white, 120x120 grayscale,
matching the style of the images used in the original honours thesis.
"""
import glob, os, numpy as np, wfdb
from PIL import Image, ImageDraw

SRC = '/tmp/mitdb/MITBIH-master/mitbih'
OUT = '/tmp/work'
PACED = {'102', '104', '107', '217'}          # excluded per AAMI practice
AAMI = {**{s: 0 for s in 'NLRej'},            # N  normal / bundle branch / escape
        **{s: 1 for s in 'AaJS'},             # S  supraventricular ectopic
        **{s: 2 for s in 'VE'},               # V  ventricular ectopic
        'F': 3,                               # F  fusion
        **{s: 4 for s in 'Q/f'}}              # Q  unknown / paced
PRE, POST = 90, 162                           # 0.25 s before, 0.45 s after R (360 Hz)
SIZE, PAD = 120, 8

def render(seg):
    lo, hi = seg.min(), seg.max()
    y = (seg - lo) / (hi - lo + 1e-8)
    xs = np.linspace(PAD, SIZE - 1 - PAD, len(seg))
    ys = (SIZE - 1 - PAD) - y * (SIZE - 1 - 2 * PAD)
    img = Image.new('L', (SIZE, SIZE), 255)
    ImageDraw.Draw(img).line(list(zip(xs, ys)), fill=0, width=1)
    return np.asarray(img, dtype=np.uint8)

recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(SRC + '/*.hea'))
recs = [r for r in recs if r not in PACED]

# First pass: count beats so the memmap can be sized exactly
plan = []
for rid in recs:
    rec = wfdb.rdrecord(f'{SRC}/{rid}')
    ch = rec.sig_name.index('MLII')           # record 114 has leads swapped
    sig = rec.p_signal[:, ch]
    ann = wfdb.rdann(f'{SRC}/{rid}', 'atr')
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI and s - PRE >= 0 and s + POST <= len(sig):
            plan.append((rid, s, AAMI[sym]))
    print(rid, len(plan), flush=True)

n = len(plan)
X = np.lib.format.open_memmap(f'{OUT}/X.npy', mode='w+', dtype=np.uint8, shape=(n, SIZE, SIZE))
y = np.empty(n, np.int8); rid_arr = np.empty(n, 'U3')

cache = {}
for i, (rid, s, lab) in enumerate(plan):
    if rid not in cache:
        cache.clear()
        rec = wfdb.rdrecord(f'{SRC}/{rid}')
        cache[rid] = rec.p_signal[:, rec.sig_name.index('MLII')]
    X[i] = render(cache[rid][s - PRE:s + POST])
    y[i] = lab; rid_arr[i] = rid
X.flush()
np.save(f'{OUT}/y.npy', y); np.save(f'{OUT}/rid.npy', rid_arr)
print('total', n, 'class counts', np.bincount(y, minlength=5))
