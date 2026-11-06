import os, sys, random, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.preprocessing import MinMaxScaler
HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = [(36, 48), (42, 54), (48, 59)]
VSLICES = [6, 12]
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def load_series():
    p = pd.read_csv(os.path.join(HERE, 'public_series.csv'))
    S = {'Singapore': p.Singapore.astype(float).tolist(), 'Hong Kong': p.Hong_Kong.astype(float).tolist(),
         'Los Angeles': p.Los_Angeles.astype(float).tolist()}
    return S

def strengths(s):
    r = STL(pd.Series(s), period=12, robust=True).fit()
    v = np.var; vr, vsr, vtr = v(r.resid), v(r.seasonal + r.resid), v(r.trend + r.resid)
    return (max(0., 1 - vr / vsr) if vsr > 0 else 0., max(0., 1 - vr / vtr) if vtr > 0 else 0.)

def _ok(fc, tr): return np.all(np.isfinite(fc)) and np.max(np.abs(fc)) <= 5 * np.max(tr)

def sarima_select(tr, h, d_D=(1, 1)):
    best = None
    d, D = d_D
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        m = SARIMAX(tr, order=(p, d, q), seasonal_order=(P, D, Q, 12),
                                    enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
                        if not m.mle_retvals.get('converged', True):
                            continue
                        fc = m.forecast(h)
                        if _ok(fc, tr) and np.isfinite(m.aicc) and (best is None or m.aicc < best[0]):
                            best = (m.aicc, (p, d, q), (P, D, Q), fc)
                    except Exception:
                        pass
    return (best[3].tolist(), dict(order=best[1], seasonal=best[2], ic=float(best[0]))) if best else (None, None)

def arima_select(tr, h):
    best = None
    for p in range(4):
        for q in range(4):
            try:
                m = ARIMA(tr, order=(p, 1, q)).fit(); fc = m.forecast(h)
                if _ok(fc, tr) and np.isfinite(m.aicc) and (best is None or m.aicc < best[0]):
                    best = (m.aicc, (p, 1, q), fc)
            except Exception:
                pass
    return best[2].tolist(), dict(order=best[1], ic=float(best[0]))

def ets_select(tr, h):
    best = None
    for trend, damped in [(None, False), ('add', True)]:
        for seasonal in [None, 'add']:
            if seasonal and len(tr) < 24:
                continue
            try:
                m = ETSModel(tr, error='add', trend=trend, damped_trend=damped, seasonal=seasonal,
                             seasonal_periods=12 if seasonal else None).fit(disp=False, maxiter=2000)
                fc = m.forecast(h)
                if _ok(fc, tr) and np.isfinite(m.aicc) and (best is None or m.aicc < best[0]):
                    best = (m.aicc, f"A,{'Ad' if trend else 'N'},{'A' if seasonal else 'N'}", fc)
            except Exception:
                pass
    return best[2].tolist(), dict(spec=best[1], ic=float(best[0]))

def snaive(tr, h): return [float(tr[len(tr) - 12 + (i % 12)]) for i in range(h)]

def _tf():
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    return tf

def lstm_fit(tr, h, seed, units=(64, 32), dropout=0.2, dense=16, lb=12, inp='level'):
    tf = _tf()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    tr = np.asarray(tr, float)
    base = np.diff(tr) if inp == 'diff' else tr
    sc = MinMaxScaler(); z = sc.fit_transform(base.reshape(-1, 1))[:, 0]
    X = np.array([z[i:i + lb] for i in range(len(z) - lb)])[..., None]; Y = z[lb:]
    os.environ['PYTHONHASHSEED'] = str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    tf.keras.backend.clear_session()
    layers = [Input((lb, 1))]
    for i, u in enumerate(units):
        layers.append(LSTM(u, return_sequences=i < len(units) - 1))
        if dropout: layers.append(Dropout(dropout))
    if dense: layers.append(Dense(dense, activation='relu'))
    layers.append(Dense(1))
    m = Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    m.fit(X, Y, epochs=300, batch_size=8, verbose=0, validation_split=0.15,
          callbacks=[EarlyStopping(patience=30, restore_best_weights=True, monitor='val_loss')])
    win = list(z[-lb:]); out = []
    for _ in range(h):
        p = float(m(np.array(win[-lb:], dtype='float32').reshape(1, lb, 1), training=False)[0, 0])
        out.append(p); win.append(p)
    fc = sc.inverse_transform(np.array(out).reshape(-1, 1))[:, 0]
    if inp == 'diff':
        fc = tr[-1] + np.cumsum(fc)
    return fc.tolist(), int(m.count_params()), len(X)

def lstm_block(tr, h, **kw):
    fcs = []
    for s in SEEDS:
        fc, npar, ns = lstm_fit(tr, h, s, **kw); fcs.append(fc)
    return dict(mean=np.mean(fcs, axis=0).tolist(), seed_forecasts=fcs, n_params=npar, n_samples=ns)
