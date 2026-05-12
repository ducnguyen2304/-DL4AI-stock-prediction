# CS313 Final Project — Time-Series Data and Application to Stock Markets

**Course:** CS313 Deep Learning for Artificial Intelligence — Spring 2026  
**Student ID:** 220080  
**Live App:** [Vietnam Stock Predictor on Hugging Face Spaces](https://huggingface.co/spaces/huynhducng/vietnam-stock-predictor)

---

## Project Overview

This project applies deep learning to financial time-series data, covering stock price prediction, trading signal identification, portfolio management, and model deployment. Two datasets are used: the **Nasdaq Composite Index** (via Yahoo Finance) and a **Vietnam stock market dataset** (HOSE/HNX exchanges).

---

## Tasks

| Task | Description | Points |
|------|-------------|--------|
| Task 1 | Nasdaq stock price prediction (multi-feature, k-th day, k-day forecast) | 15% |
| Task 2 | Vietnam stock price prediction (multi-feature, k-th day, k-day forecast) | 15% |
| Task 3 | Trading signal identification — BUY / SELL signals for Vietnam market | 20% |
| Task 4 | Portfolio management — profitable stock selection, risk scoring, optimization | 30% |
| Task 5 | Deployment — REST API (FastAPI) and SaaS web app (Streamlit + Hugging Face) | 30% extra |
| Task 6 | Report, GitHub repository, README | 20% |

---

## Repository Structure

```
├── notebooks/
│   └── 220080_project_notebook.ipynb   # Main notebook with all tasks (1–5)
├── task5_deployment/
│   ├── api.py                           # FastAPI REST service (Task 5.1)
│   ├── app.py                           # Streamlit web app (Task 5.2)
│   └── requirements.txt                 # Deployment-specific dependencies
├── reports/
│   ├── 220080-project-report.pdf        # Full project report (Task 6.1)
│   └── Final-project-DL4AI.pdf          # Assignment specification
├── data/
│   └── README.md                        # Data sources and download instructions
└── requirements.txt                     # Full project dependencies
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone git@github.com:ducnguyen2304/-DL4AI-stock-prediction.git
cd -DL4AI-stock-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the data

**Nasdaq dataset** — downloaded automatically inside the notebook via `yfinance`:
```python
import yfinance as yf
df = yf.download('^IXIC', start='2010-01-01')
```

**Vietnam dataset** — place the course-provided `data-vn-20230228/` folder inside `data/`. See [data/README.md](data/README.md) for details.

---

## Running the Notebook

Open the main notebook in Google Colab or Jupyter:

```bash
jupyter notebook notebooks/220080_project_notebook.ipynb
```

> The notebook was developed on **Google Colab**. If running locally, make sure to update any `/content/drive/...` file paths to your local paths.

Run cells sequentially from top to bottom. Each task section is clearly marked with a heading (e.g., `## Task 1`, `## Task 2.1`).

---

## Running the Deployment (Task 5)

### Task 5.1 — FastAPI REST API

```bash
cd task5_deployment
pip install -r requirements.txt
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"window": [[...30 rows of [Low, Open, Volume, High, Close]...]]}'
```

> You need the trained model files (`saved_models/`) from the notebook before running the API.

### Task 5.2 — Streamlit Web App

```bash
cd task5_deployment
streamlit run app.py
```

Or use the live deployed version: [huggingface.co/spaces/huynhducng/vietnam-stock-predictor](https://huggingface.co/spaces/huynhducng/vietnam-stock-predictor)

---

## Key Design Decisions

- **LSTM architecture** used for all prediction tasks due to its strength with sequential time-series data.
- **Multi-feature input** (Low, High, Open, Close, Volume) improves prediction accuracy over single-feature baselines.
- **Time-series-aware splits**: training/validation/test sets are split chronologically — no random shuffling — to prevent data leakage.
- **BUY/SELL signals** framed as binary classification: label = 1 if price rises/falls significantly within the next N days.
- **Portfolio optimization** uses Sharpe Ratio maximization to balance profitability and risk.

---

## Results Summary

| Model | Task | Metric |
|-------|------|--------|
| LSTM (multi-feature) | Nasdaq next-day prediction | RMSE on test set |
| LSTM (multi-feature) | Vietnam next-day prediction | RMSE on test set |
| LSTM classifier | BUY signal detection | Accuracy / F1 |
| LSTM classifier | SELL signal detection | Accuracy / F1 |
| Sharpe optimization | Portfolio (10 stocks) | Annualized Sharpe Ratio |

> Full metrics and charts are in the [project report](reports/220080-project-report.pdf).
