# Data

## Nasdaq Dataset
- Source: Yahoo Finance via `yfinance`
- Ticker: `^IXIC`
- Download in the notebook with: `yf.download('^IXIC', start='2010-01-01')`

## Vietnam Stock Dataset
- Source: Provided by course (`data-vn-20230228/`)
- Contains historical OHLCV data for Vietnamese companies (HOSE, HNX exchanges)
- Required columns: `TradingDate`, `Low`, `Open`, `Volume`, `High`, `Close`

> Data files are not committed to this repository due to size. Download and place them in this folder before running the notebook.
