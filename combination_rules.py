import numpy as np

FS_THRESHOLD = 0.64
TAUS = [round(0.30 + 0.05 * i, 2) for i in range(9)]
SLICES = [6, 12]
TOLERANCES = [1.0, 1.5]
POOL = ['SNaive', 'ETS', 'ARIMA', 'SARIMA', 'LSTM']

def primary_D(fs):
    return 1 if fs > FS_THRESHOLD else 0

def mape(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)

def combine(F, M, fs, ft):
    o = dict(F)
    S, L = F['SARIMA'], F['LSTM']
    avg = .5 * S + .5 * L
    o['Avg'] = avg
    w = fs / (fs + ft) if fs + ft > 0 else .5
    o['Hybrid'] = w * S + (1 - w) * L
    for t in TAUS:
        o[f'Fs@{t}'] = avg if fs >= t else L
    P = np.vstack([F[k] for k in POOL])
    o['EQ5'] = P.mean(0)
    o['Med5'] = np.median(P, axis=0)
    o['Trim5'] = np.sort(P, axis=0)[1:4].mean(0)
    o['EQ3'] = (F['SNaive'] + S + L) / 3
    for v in SLICES:
        m = M[v]
        for k in TOLERANCES:
            o[f'VE_v{v}_k{k}'] = avg if m['SARIMA'] <= k * m['LSTM'] else L
        o[f'Sel2_v{v}'] = o[min(['SARIMA', 'LSTM'], key=lambda z: m[z])]
        o[f'Sel3_v{v}'] = o[min(['SARIMA', 'LSTM', 'Avg'], key=lambda z: m[z])]
        o[f'SelAll_v{v}'] = o[min(POOL + ['Avg'], key=lambda z: m[z])]
        ws = (1 / m['SARIMA']) / (1 / m['SARIMA'] + 1 / m['LSTM'])
        o[f'InvW_v{v}'] = ws * S + (1 - ws) * L
        iv = np.array([1 / m[k] for k in POOL])
        o[f'InvW5_v{v}'] = (iv / iv.sum()) @ P
        order = sorted(POOL, key=lambda z: m[z])
        o[f'DropW5_v{v}'] = np.mean([F[k] for k in order[:4]], axis=0)
        o[f'Top2_v{v}'] = np.mean([F[k] for k in order[:2]], axis=0)
        best = min(m[k] for k in POOL)
        o[f'Screen5_v{v}'] = np.mean([F[k] for k in POOL if m[k] <= 1.5 * best], axis=0)
    return o

def block_bootstrap_weights(n_months=23, block=3, replicates=5000, seed=20261106):
    rng = np.random.default_rng(seed)
    starts = n_months - block + 1
    k = int(np.ceil(n_months / block))
    W = np.zeros((replicates, n_months))
    for b in range(replicates):
        for s in rng.integers(0, starts, k):
            W[b, s:s + block] += 1
    return W
