# Supplementary File S1: core code and public-series data

For "Forecast Combination for Short Container-Throughput Records: Constituent Specification, Gating and Weighting for Port Resource Planning".

Python 3.10; versions in `requirements.txt`.

- `public_series.csv`: the Singapore, Hong Kong and Los Angeles monthly series, January 2021 to November 2025.
  Sources: Maritime and Port Authority of Singapore via data.gov.sg (https://data.gov.sg/datasets/d_da030f7028200d19ffcbe4a2d71af39c),
  Hong Kong Maritime and Port Board via DATA.GOV.HK (https://data.gov.hk/en-data/dataset/hk-tlb-tlb-mpb) and
  Port of Los Angeles (https://www.portoflosangeles.org/business/statistics/container-statistics).
- `models.py`: STL strengths, seasonal naive, ETS, ARIMA, SARIMA and LSTM.
- `combination_rules.py`: the seasonal-differencing rule, combination and gating rules, MAPE and the block-bootstrap weights.
- `example_one_fold.py`: one series and fold end to end, for example `python example_one_fold.py Singapore 1`.

This file holds the core methods only, not every script behind the article's tables. Fold-level results are in Tables S4 and S5 of the Supplementary Materials. The Terminal A data are proprietary and are not included.
