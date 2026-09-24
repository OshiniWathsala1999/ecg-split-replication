"""Replicate the honours-thesis ConvNet_1 under two data-splitting protocols.

Architecture and hyperparameters match the thesis and model1.py exactly:
conv(1->4,3x3)-ReLU-maxpool2 -> conv(4->4,3x3)-ReLU-maxpool2 -> dropout(0.5)
-> linear(3600->5); SGD lr=0.001 momentum=0.9, batch 32, 5 epochs,
inputs normalised to mean 0.5 / std 0.5. PyTorch default initialisation.
"""
import json, sys, time, numpy as np, jax, jax.numpy as jnp, optax

W = '/tmp/work'
DS1 = '101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230'.split()
DS2 = '100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234'.split()
CLS = ['N', 'S', 'V', 'F', 'Q']

def init(key):
    def u(k, shape, fan_in):
        b = 1.0 / np.sqrt(fan_in); return jax.random.uniform(k, shape, minval=-b, maxval=b)
    k = jax.random.split(key, 6)
    return {'c1w': u(k[0], (3, 3, 1, 4), 9),    'c1b': u(k[1], (4,), 9),
            'c2w': u(k[2], (3, 3, 4, 4), 36),   'c2b': u(k[3], (4,), 36),
            'fw':  u(k[4], (3600, 5), 3600),    'fb':  u(k[5], (5,), 3600)}

def forward(p, x, key, train):
    dn = ('NHWC', 'HWIO', 'NHWC')
    pool = lambda h: jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    h = jax.lax.conv_general_dilated(x, p['c1w'], (1, 1), 'SAME', dimension_numbers=dn) + p['c1b']
    h = pool(jax.nn.relu(h))
    h = jax.lax.conv_general_dilated(h, p['c2w'], (1, 1), 'SAME', dimension_numbers=dn) + p['c2b']
    h = pool(jax.nn.relu(h)).reshape(x.shape[0], -1)
    if train:
        keep = jax.random.bernoulli(key, 0.5, h.shape); h = jnp.where(keep, h / 0.5, 0.0)
    return h @ p['fw'] + p['fb']

opt = optax.sgd(0.001, momentum=0.9)

@jax.jit
def step(p, s, x, y, key):
    def loss(p):
        return optax.softmax_cross_entropy_with_integer_labels(forward(p, x, key, True), y).mean()
    l, g = jax.value_and_grad(loss)(p)
    u, s = opt.update(g, s, p)
    return optax.apply_updates(p, u), s, l

@jax.jit
def predict(p, x):
    return jnp.argmax(forward(p, x, None, False), -1)

def prep(X, idx):
    return ((np.asarray(X[np.sort(idx)], np.float32) / 255.0 - 0.5) / 0.5)[..., None], np.sort(idx)

def metrics(yt, yp):
    cm = np.zeros((5, 5), int)
    np.add.at(cm, (yt, yp), 1)
    out = {'accuracy': float(np.trace(cm) / cm.sum()), 'confusion': cm.tolist(), 'per_class': {}}
    f1s = []
    for c in range(5):
        tp = cm[c, c]; se = tp / max(cm[c].sum(), 1); pp = tp / max(cm[:, c].sum(), 1)
        f1 = 2 * se * pp / max(se + pp, 1e-12)
        out['per_class'][CLS[c]] = {'n': int(cm[c].sum()), 'Se': float(se), 'PPV': float(pp), 'F1': float(f1)}
        if c < 4: f1s.append(f1)
    out['macro_F1_NSVF'] = float(np.mean(f1s))
    return out

def run(protocol, seed):
    X = np.load(f'{W}/X.npy', mmap_mode='r'); y = np.load(f'{W}/y.npy').astype(np.int32)
    rid = np.load(f'{W}/rid.npy'); rng = np.random.default_rng(seed)
    if protocol == 'A':                                  # beat-level random split, as in the thesis
        perm = rng.permutation(len(y)); cut = int(0.8 * len(y))
        tr, te = perm[:cut], perm[cut:]
    else:                                                # inter-patient DS1 -> DS2
        tr = np.where(np.isin(rid, DS1))[0]; te = np.where(np.isin(rid, DS2))[0]
    key = jax.random.PRNGKey(seed); p = init(key); s = opt.init(p)
    t0 = time.time()
    for ep in range(5):
        order = rng.permutation(tr)
        for i in range(0, len(order) - 31, 32):
            xb, ib = prep(X, order[i:i + 32])
            key, k = jax.random.split(key)
            p, s, l = step(p, s, jnp.asarray(xb), jnp.asarray(y[ib]), k)
        print(f'{protocol} seed{seed} epoch{ep + 1} loss {float(l):.3f} {time.time() - t0:.0f}s', flush=True)
    preds = []
    te = np.sort(te)
    for i in range(0, len(te), 1024):
        xb, _ = prep(X, te[i:i + 1024]); preds.append(np.asarray(predict(p, jnp.asarray(xb))))
    res = metrics(y[te], np.concatenate(preds))
    res.update({'protocol': protocol, 'seed': seed, 'n_train': int(len(tr)), 'n_test': int(len(te)),
                'train_seconds': round(time.time() - t0)})
    json.dump(res, open(f'{W}/result_{protocol}_{seed}.json', 'w'), indent=1)
    print(protocol, seed, 'acc', round(res['accuracy'], 4), 'macroF1', round(res['macro_F1_NSVF'], 4), flush=True)

if __name__ == '__main__':
    for proto in sys.argv[1].split(','):
        for sd in map(int, sys.argv[2].split(',')):
            run(proto, sd)
