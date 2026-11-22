"""Rebalancing control: does class weighting or oversampling recover minority-class
performance under the inter-patient split?

Identical to torch_train.py (same model, optimiser, learning rate, batch size, epochs,
images and splits) except for one change in how the training loss sees the classes:

  --balance none        unweighted cross-entropy (the original configuration)
  --balance weight      class-weighted cross-entropy, w_c = n / (K * n_c)
  --balance oversample  WeightedRandomSampler, each beat drawn with probability
                        proportional to 1 / n_c; epoch length unchanged

Q has very few training beats (8 in DS1), so its weight is capped at the largest
weight among N, S, V and F. Q is excluded from macro-F1 as in the main analysis.

    python rebalance_control.py --protocol B --balance weight --seed 0
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_train import ConvNet_1, Beats, metrics, DS1, DS2


def class_weights(y_tr):
    n = np.bincount(y_tr, minlength=5).astype(float)
    w = len(y_tr) / (5 * np.maximum(n, 1))
    w[4] = min(w[4], w[:4].max())
    return w


def main(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    X = np.load(f'{a.data}/X.npy', mmap_mode='r')
    y = np.load(f'{a.data}/y.npy').astype(np.int64)
    rid = np.load(f'{a.data}/rid.npy')

    if a.protocol == 'A':
        perm = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        tr, te = perm[:cut], perm[cut:]
    else:
        tr = np.where(np.isin(rid, DS1))[0]
        te = np.where(np.isin(rid, DS2))[0]

    w = class_weights(y[tr])
    ds = Beats(X, y, tr)
    if a.balance == 'oversample':
        sw = torch.as_tensor(w[y[tr]], dtype=torch.double)
        g = torch.Generator().manual_seed(a.seed)
        loader = DataLoader(ds, batch_size=32, drop_last=True,
                            sampler=WeightedRandomSampler(sw, len(tr), replacement=True, generator=g))
    else:
        loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    model = ConvNet_1()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32)
                                if a.balance == 'weight' else None)

    t0 = time.time()
    model.train()
    for ep in range(5):
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        print(f'epoch {ep + 1} loss {loss.item():.3f} {time.time() - t0:.0f}s', flush=True)

    model.eval()
    cm = np.zeros((5, 5), int)
    with torch.no_grad():
        for xb, yb in DataLoader(Beats(X, y, te), batch_size=512):
            np.add.at(cm, (yb.numpy(), model(xb).argmax(1).numpy()), 1)

    res = metrics(cm)
    res.update({'protocol': a.protocol, 'balance': a.balance, 'seed': a.seed,
                'class_weights': w.tolist(), 'n_train': int(len(tr)), 'n_test': int(len(te))})
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f'{a.out}/ctrl_{a.protocol}_{a.balance}_{a.seed}.json', 'w'), indent=1)
    print(a.protocol, a.balance, a.seed, 'acc', round(res['accuracy'], 4),
          'macroF1', round(res['macro_F1_NSVF'], 4), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', choices=['A', 'B'], required=True)
    p.add_argument('--balance', choices=['none', 'weight', 'oversample'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data', default='.')
    p.add_argument('--out', default='results/rebalance')
    main(p.parse_args())
