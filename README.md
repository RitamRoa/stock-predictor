# 📈 Stock Price Predictor using XGBoost and Technical Indicators

A Python project to predict the **next day's stock price** using historical data, technical indicators, and machine learning with **XGBoost**. This tool fetches real-time data from Yahoo Finance via `yfinance`, applies a set of technical analysis features, and builds a predictive model.

---

## 🚀 Features

- Predicts **next day's closing price**
- Uses **XGBoost Regressor** for accuracy and speed
- Adds over 15 **technical indicators**:
  - Moving Averages (SMA 10, 50, 200)
  - RSI, MACD, Bollinger Bands
  - Stochastic Oscillators
  - Volume ROC and SMA
- Data automatically pulled using `yfinance`
- Clean logging and error handling
- Prediction interval and model confidence score

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
