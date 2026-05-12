from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(title='Vietnam Stock Signal API')

BUY_MODEL_PATH  = 'saved_models/buy_signal_model.keras'
SELL_MODEL_PATH = 'saved_models/sell_signal_model.keras'
SCALER_PATH     = 'saved_models/scaler.pkl'

buy_model  = load_model(BUY_MODEL_PATH)
sell_model = load_model(SELL_MODEL_PATH)
scaler     = joblib.load(SCALER_PATH)

TIME_STEPS      = 30
FEATURE_COLUMNS = ['Low', 'Open', 'Volume', 'High', 'Close']

class PredictRequest(BaseModel):
    window: list[list[float]]  # 30 rows x 5 features

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(req: PredictRequest):
    window = np.array(req.window, dtype=np.float32)

    if window.shape != (TIME_STEPS, len(FEATURE_COLUMNS)):
        return {'error': f'Expected shape ({TIME_STEPS}, {len(FEATURE_COLUMNS)}), got {window.shape}'}

    scaled = scaler.transform(window)
    X      = scaled.reshape(1, TIME_STEPS, len(FEATURE_COLUMNS))

    buy_prob  = float(buy_model.predict(X,  verbose=0)[0][0])
    sell_prob = float(sell_model.predict(X, verbose=0)[0][0])

    if buy_prob >= 0.5:
        signal = 'BUY'
    elif sell_prob >= 0.5:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return {
        'buy_signal_probability':  round(buy_prob,  4),
        'sell_signal_probability': round(sell_prob, 4),
        'signal': signal,
        'model': 'vietnam_stock_lstm'
    }
