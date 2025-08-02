# 📈 Stock Price Direction Predictor (with Sentiment Analysis)

A smart stock analysis tool that predicts the **next-day movement** (UP/DOWN) of a stock using **technical indicators**, **XGBoost classification**, and **news sentiment analysis**. It fetches live stock data, performs feature engineering, adds NLP-based sentiment insights, and produces a confident forecast of where the stock is headed next.

---

## 🚀 Features

- ✅ **Live stock data** fetched via `yfinance`
- 📉 **Technical indicators**:
  - SMA (50-day, 200-day)
  - Bollinger Bands
  - RSI, MACD, EMA
- 🤖 **XGBoost Classifier** for predicting UP/DOWN direction
- 🧠 **News Sentiment Analysis** using VADER (via NLTK)
- 📊 **Visualization** of:
  - Stock price & indicators
  - Predicted vs Actual direction (color-coded)
- 📦 Easy CLI interface — just input a stock symbol and go

---

## 🧪 Sample Prediction Output

=== Prediction Results ===
Current Price: ₹3445.70
Predicted Direction: DOWN
Probability of Price Increase: 20.08%
Confidence: 79.92%
Sentiment Score: -0.61 (Negative)

---

## 📁 Project Structure
stock-predictor/
├── main.py # Main entry point: handles data pipeline
├── indicators.py # All technical indicator functions
├── sentiment.py # Scrapes news & performs sentiment analysis
├── model.py # ML model: XGBoost training & prediction
├── visualize.py # Prediction vs actual graph & indicator plots
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ⚙️ Installation

```bash
git clone https://github.com/RitamRoa/stock-predictor.git
cd stock-predictor
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

