# Financial ML Pipeline — Tick Data, Expanded Features, Model Comparison

An end-to-end financial machine learning pipeline built on real trade-by-trade tick data,
following the methodology of López de Prado's *Advances in Financial Machine Learning*
(AFML) — applied to IVE (iShares S&P 500 Value ETF) bid/ask tick data.

## Pipeline Stages

1. **Ingestion & Cleaning** — parses raw Kibot-format tick data (`Date, Time, Price, Bid, Ask, Size`, no header), builds a datetime index, derives traded volume and dollar value per tick
2. **Dollar Bars** — samples bars by cumulative traded dollar value rather than fixed time intervals, giving more statistically uniform bars
3. **Fractional Differentiation** — achieves stationarity while preserving memory, via non-integer-order differencing (fitted order *d\**)
4. **Triple-Barrier Labeling** — CUSUM filter for event sampling + triple-barrier method (profit-take / stop-loss / time-out) for label generation
5. **Feature Engineering (25 features)** — fractional-differentiation, microstructure (bid/ask/size-based), and additional feature families
6. **Feature Importance & Selection** — Mean Decrease Impurity (MDI) from a bagged, class-balanced, sample-weighted Random Forest
7. **Four Classifiers** — Logistic Regression, Random Forest, XGBoost, and an LSTM, trained on identical selected features and folds
8. **Purged K-Fold Cross-Validation** — purging and embargoing to prevent label leakage across the CV fold boundary (per AFML Ch. 7)
9. **Per-Event Realized-Return Statistics & Final Dashboard** — non-compounded, per-event performance stats plus a consolidated summary of every stage's output

## Data

This notebook uses **free historical intraday tick data with bid/ask** for the **IVE**
symbol from Kibot, downloaded as a `.txt` file (`Date, Time, Price, Bid, Ask, Size`,
no header):

📁 Source: [Kibot Free Historical Intraday Data](https://kibot.com/free-historical-intraday-data.html)

Due to file size, the raw tick data is **not included in this repository**. To reproduce:
1. Download the free IVE tick dataset from the Kibot link above.
2. Place the `.txt` file in a `data/` folder at the repo root.
3. Update `DATA_PATH` in the notebook's Configuration cell to point to your file.

A `DRY_RUN` toggle is available in the Configuration section to run the pipeline on a
subsample first, since tick files can run into millions of rows.

## Repository Structure

```
financial-ml-tick-pipeline/
├── code.ipynb
├── README.md
├── requirements.txt
└── data/          (not included — see Data section above)
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook code.ipynb
```

Set `DATA_PATH` to your downloaded Kibot `.txt` file and toggle `DRY_RUN = True` for a
quick first pass before running on the full tick history.

## Key Methodological Notes

- **Why dollar bars over time bars:** fixed-time sampling oversamples quiet periods and undersamples bursts of activity; dollar bars normalize for this.
- **Why fractional differentiation over integer returns:** integer differencing achieves stationarity but destroys the series' memory; fractional differencing balances both.
- **Why purged K-Fold over standard K-Fold:** standard K-Fold leaks information whenever a label's outcome window crosses a fold boundary — purging and embargoing (per López de Prado) fix this.

## Author

Yuvraj — Quantitative Options Analyst, Futures First. Built as part of an applied
quant/ML project series (Lopez de Prado AFML methodology) ahead of MFE program
applications.
