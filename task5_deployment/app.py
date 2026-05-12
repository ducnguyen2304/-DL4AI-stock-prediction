import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Vietnam Stock Predictor", layout="wide")
st.title("Vietnam Stock Signal Predictor")
st.markdown("Upload a Vietnam stock CSV to get **BUY / SELL** signals from the LSTM model.")

FEATURE_COLUMNS = ['Low', 'Open', 'Volume', 'High', 'Close']
TIME_STEPS = 30

@st.cache_resource
def load_models():
    buy_model  = load_model("saved_models/buy_signal_model.keras")
    sell_model = load_model("saved_models/sell_signal_model.keras")
    scaler     = joblib.load("saved_models/scaler.pkl")
    return buy_model, sell_model, scaler

buy_model, sell_model, scaler = load_models()

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Signal threshold", 0.1, 0.9, 0.5, 0.05)

uploaded = st.sidebar.file_uploader("Upload stock CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    date_col = 'TradingDate' if 'TradingDate' in df.columns else 'Date'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing columns: {missing}")
        st.stop()

    if len(df) < TIME_STEPS + 1:
        st.error(f"Need at least {TIME_STEPS + 1} rows of data.")
        st.stop()

    data   = df[FEATURE_COLUMNS].values
    scaled = scaler.transform(data)
    X = np.array([scaled[i-TIME_STEPS:i] for i in range(TIME_STEPS, len(scaled))])

    buy_probs  = buy_model.predict(X, verbose=0).flatten()
    sell_probs = sell_model.predict(X, verbose=0).flatten()

    result_df = df.iloc[TIME_STEPS:].copy().reset_index(drop=True)
    result_df['BUY_prob']  = buy_probs
    result_df['SELL_prob'] = sell_probs
    result_df['Signal']    = 'HOLD'
    result_df.loc[result_df['BUY_prob']  >= threshold, 'Signal'] = 'BUY'
    result_df.loc[result_df['SELL_prob'] >= threshold, 'Signal'] = 'SELL'

    latest = result_df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Close",     f"{latest['Close']:.2f}")
    col2.metric("BUY Probability",  f"{latest['BUY_prob']:.2%}")
    col3.metric("SELL Probability", f"{latest['SELL_prob']:.2%}")

    signal = latest['Signal']
    if signal == 'BUY':
        st.success(f"Signal: **BUY**")
    elif signal == 'SELL':
        st.error(f"Signal: **SELL**")
    else:
        st.info(f"Signal: **HOLD**")

    st.subheader("Close Price with BUY / SELL signals")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result_df[date_col], result_df['Close'], label='Close', color='gray', linewidth=1)

    buys  = result_df[result_df['Signal'] == 'BUY']
    sells = result_df[result_df['Signal'] == 'SELL']
    ax.scatter(buys[date_col],  buys['Close'],  marker='^', color='green', label='BUY',  zorder=5)
    ax.scatter(sells[date_col], sells['Close'], marker='v', color='red',   label='SELL', zorder=5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Signal Table (last 20 rows)")
    st.dataframe(
        result_df[[date_col, 'Close', 'BUY_prob', 'SELL_prob', 'Signal']].tail(20),
        use_container_width=True
    )
else:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.markdown("""
    **Expected CSV columns:** `TradingDate`, `Low`, `Open`, `Volume`, `High`, `Close`

    Example stocks: FPT, HPG, VCB, VNM from the Vietnam dataset.
    """)
