import sys
import numpy as np
from models import FOLDS, ets_select, arima_select, lstm_block, load_series, sarima_select, snaive, strengths
from combination_rules import SLICES, combine, mape, primary_D

def fit_all(train, h):
    fs, ft = strengths(train)
    return dict(SNaive=np.array(snaive(train, h)),
                ETS=np.array(ets_select(train, h)[0]),
                ARIMA=np.array(arima_select(train, h)[0]),
                SARIMA=np.array(sarima_select(train, h, d_D=(1, primary_D(fs)))[0]),
                LSTM=np.array(lstm_block(train, h)['mean'])), fs, ft

series, fold = sys.argv[1], int(sys.argv[2])
values = load_series()[series]
train_end, test_end = FOLDS[fold - 1]
train, actual = values[:train_end], np.array(values[train_end:test_end])
F, fs, ft = fit_all(train, len(actual))
M = {}
for v in SLICES:
    I, _, _ = fit_all(train[:-v], v)
    I['Avg'] = .5 * I['SARIMA'] + .5 * I['LSTM']
    M[v] = {k: mape(train[-v:], x) for k, x in I.items()}
rules = combine(F, M, fs, ft)
print(f'{series} fold {fold}: F_s = {fs:.3f}, D = {primary_D(fs)}')
for k in ['SNaive', 'ETS', 'ARIMA', 'SARIMA', 'LSTM', 'Avg', 'Hybrid', 'Fs@0.5', 'VE_v12_k1.0', 'VE_v12_k1.5', 'InvW_v12', 'EQ5']:
    print(f'{k:12s} {mape(actual, rules[k]):6.2f}')
