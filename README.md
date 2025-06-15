# 📈 Stock Predictor

A machine learning-based tool to predict the **next-day stock price direction (UP/DOWN)** using a combination of **technical indicators**, **XGBoost**, and **sentiment analysis**.

---

## 🔍 Features

- ✅ Fetches live historical stock data via `yfinance`
- 📊 Calculates technical indicators:
  - 50-day and 200-day SMA
  - Bollinger Bands (Upper & Lower)
  - RSI, MACD, EMA, and others
- 🤖 Uses XGBoost for classification (UP/DOWN)
- 🧠 Integrates **news sentiment analysis** (positive/negative)
- 📉 Shows **prediction vs actual direction** comparison graph
- 📈 Visualizes price + technical indicators

---

## 🧪 Example Output

```bash
=== Prediction Results ===
Current Price: ₹3445.70
Predicted Direction: DOWN
Probability of Price Increase: 20.08%
Confidence: 79.92%
Sentiment Score: -0.61 (Negative). 

