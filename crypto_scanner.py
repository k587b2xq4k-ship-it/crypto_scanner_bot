"""
crypto_scanner.py
Simple Crypto Scanner + Predictor (starter)
NOT FINANCIAL ADVICE. Use for research/paper-trading only.
"""

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
import os, datetime
from typing import Tuple
# -------------------------
# Telegram integration
# -------------------------
import os
import requests

TELEGRAM_TOKEN = os.getenv("TG_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID", "")

def send_telegram_message(msg: str):
    """Send a message to your Telegram chat via your bot token."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram token or chat ID not set. Use export TG_TOKEN and TG_CHAT_ID.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code != 200:
            print(f"Telegram send failed ({response.status_code}): {response.text}")
        else:
            print("✅ Telegram message sent successfully.")
    except Exception as e:
        print(f"Telegram send error: {e}")

# -------------------------
# Configuration
# -------------------------
EXCHANGE_ID = "binanceus"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
MODEL_PATH = "model.pkl"
LIMIT = 2000
MODEL_PATH = "rf_model.joblib"
TELEGRAM_ENABLED = False
TELEGRAM_TOKEN = os.getenv("TG_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID", "")

def now_ts():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# -------------------------
# Data Fetcher
# -------------------------
def fetch_ohlcv(exchange_id, symbol, timeframe, limit=1000) -> pd.DataFrame:
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("datetime", inplace=True)
    df = df[["open","high","low","close","volume"]].astype(float)
    return df

# -------------------------
# Indicators
# -------------------------
def add_indicators(df):
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd.iloc[:,0], macd.iloc[:,1], macd.iloc[:,2]
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["vol_14"] = df["volume"].rolling(14).mean()
    return df.dropna()

# -------------------------
# Labels
# -------------------------
def add_labels(df, horizon=1, threshold=0.0005):
    df = df.copy()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = (df["future_return"] > threshold).astype(int)
    return df.dropna()

# -------------------------
# Model
# -------------------------
FEATURES = [
    "ema_fast","ema_slow","macd","macd_signal","macd_hist",
    "rsi","atr","return_1","return_3","return_6",
    "hl_range","oc_change","vol_14"
]

def train_model(df):
    print(" Training started…")
    df = add_indicators(df)
    df = add_labels(df)
    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    dump(clf, MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")
    preds = clf.predict(Xte)
    print(f"\nAccuracy: {accuracy_score(yte,preds):.3f}")
    print(classification_report(yte,preds,zero_division=0))
    return clf    

def load_model():
    return load(MODEL_PATH)

def predict_signal(model, df):
    X = df[FEATURES].iloc[-1:].fillna(0)
    label = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0].tolist()
    return label, prob

# -------------------------
# Main
# -------------------------
def main(mode="scan"):
    print(f"[{now_ts()}] Running mode={mode}")
    df = fetch_ohlcv(EXCHANGE_ID, SYMBOL, TIMEFRAME, limit=LIMIT)
    df = add_indicators(df)

    if mode=="train":
        train_model(df)
        return

    if not os.path.exists(MODEL_PATH):
        print("No model found, training one now...")
        train_model(df)
    model = load_model()

    label, prob = predict_signal(model, df)
    if label==1:
        print(f"BUY signal → prob: {prob}")
    else:
        print(f"NO-BUY/SELL signal → prob: {prob}")
# -------------------------
# Continuous multi-symbol scan loop
# -------------------------
import time

def run_auto_scan(model, symbols, timeframe="1h", limit=200, threshold=0.65, sleep_sec=3600):
    """
    Every `sleep_sec` seconds, fetch new data and send Telegram alerts for
    symbols with buy probability above `threshold`.
    """
    log_file = "scan_log.csv"
    print(f"🔄  Auto-scanner started  (interval={sleep_sec}s)")

    while True:
        for sym in symbols:
            try:
                df = fetch_ohlcv(EXCHANGE_ID, sym, timeframe, limit=limit)
                df = add_indicators(df)
                label, prob = predict_signal(model, df)
                buy_prob = prob[1] if prob and len(prob) > 1 else 0
                ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

                # ----- Logging -----
                with open(log_file, "a") as f:
                    f.write(f"{ts},{sym},{buy_prob:.3f},{label}\n")

                # ----- Alerts -----
                if label == 1 and buy_prob >= threshold:
                    msg = f"📈 BUY {sym} prob={buy_prob:.2f}"
                    print(msg)
                    send_telegram_message(msg)
                else:
                    print(f"{ts} → {sym} no buy (prob={buy_prob:.2f})")

            except Exception as e:
                print(f"⚠️ Scan error {sym}: {e}")

        print(f"Sleeping {sleep_sec/60:.0f} min …")
        time.sleep(sleep_sec)
# -------------------------
# Auto-scanner entry point
# -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["scan", "train"], default="scan")
    args = p.parse_args()

    # If you still want manual single-scan or training capability:
    if args.mode in ["scan", "train"]:
        main(args.mode)
        exit()

    # Otherwise, run continuous scanning below
    # ----------------------------------------

    # Load the trained model
    model = joblib.load("rf_model.joblib")   # Make sure this file exists from your training step

    # Pick which coins to scan (you can add more)
    symbols = ["TEL/USDT"]

    # Start continuous scanning
    #  - timeframe="30m"  → 1-hour candles
    #  - threshold=0.65  → only alert if prob ≥ 65%
    #  - sleep_sec=1800  → wait 1 hour between scans
    run_auto_scan(model, symbols, timeframe="1h", threshold=0.65, sleep_sec=3600)

