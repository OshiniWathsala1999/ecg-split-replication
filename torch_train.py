"""PyTorch reference implementation (canonical version of the experiments).

Reproduces both splitting protocols with the original honours-project
architecture and hyperparameters. Expects the beat-image dataset produced by
build_dataset.py.

    python torch_train.py --protocol A --seed 0
    python torch_train.py --protocol B --seed 0
"""
import argparse, json, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DS1 = '101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230'.split()
DS2 = '100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234'.split()
CLS = ['N', 'S', 'V', 'F', 'Q']


class ConvNet_1(nn.Module):
    """Unchanged from the original honours project (model1.py)."""

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.layer_2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4 * 30 * 30, 5)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.layer_1(x)))
        out = self.maxpool2(self.relu2(self.layer_2(out)))
        return self.fc1(self.drop_out(out.reshape(out.size(0), -1)))


class Beats(Dataset):
    def __init__(self, X, y, idx):
        self.X, self.y, self.idx = X, y, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(np.asarray(self.X[j], np.float32) / 255.0)
        return ((x - 0.5) / 0.5).unsqueeze(0), int(self.y[j])


def metrics(cm):
    out, f1s = {'accuracy': float(np.trace(cm) / cm.sum()), 'per_class': {}}, []
    for c in range(5):
        tp = cm[c, c]
        se = tp / max(cm[c].sum(), 1)
        pp = tp / max(cm[:, c].sum(), 1)
        f1 = 2 * se * pp / max(se + pp, 1e-12)
        out['per_class'][CLS[c]] = {'n': int(cm[c].sum()), 'Se': float(se),
                                    'PPV': float(pp), 'F1': float(f1)}
        if c < 4:
            f1s.append(f1)
    out['macro_F1_NSVF'] = float(np.mean(f1s))
    out['confusion'] = cm.tolist()
    return out


def main(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    X = np.load(f'{a.data}/X.npy', mmap_mode='r')
    y = np.load(f'{a.data}/y.npy').astype(np.int64)
    rid = np.load(f'{a.data}/rid.npy')

    if a.protocol == 'A':                       # beat-level random split
        perm = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        tr, te = perm[:cut], perm[cut:]
    else:                                       # inter-patient DS1 -> DS2
        tr = np.where(np.isin(rid, DS1))[0]
        te = np.where(np.isin(rid, DS2))[0]

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvNet_1().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(Beats(X, y, tr), batch_size=32, shuffle=True, drop_last=True)

    t0 = time.time()
    model.train()
    for ep in range(5):
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb.to(dev)), yb.to(dev))
            loss.backward()
            opt.step()
        print(f'epoch {ep + 1} loss {loss.item():.3f} {time.time() - t0:.0f}s', flush=True)

    model.eval()
    cm = np.zeros((5, 5), int)
    with torch.no_grad():
        for xb, yb in DataLoader(Beats(X, y, te), batch_size=512):
            np.add.at(cm, (yb.numpy(), model(xb.to(dev)).argmax(1).cpu().numpy()), 1)

    res = metrics(cm)
    res.update({'protocol': a.protocol, 'seed': a.seed, 'framework': 'pytorch',
                'n_train': int(len(tr)), 'n_test': int(len(te))})
    json.dump(res, open(f'{a.data}/torch_{a.protocol}_{a.seed}.json', 'w'), indent=1)
    print(a.protocol, a.seed, 'acc', round(res['accuracy'], 4),
          'macroF1', round(res['macro_F1_NSVF'], 4))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', choices=['A', 'B'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data', default='.')
    main(p.parse_args())
