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

---
## for reference
.\.venv\Scripts\Activate.ps1
What does "Prediction Confidence" mean in your code?
In your project, prediction confidence is:

The average importance of all features used by the model, shown as a percentage.

🔍 Why is it called confidence?
Your model (Random Forest) tells us which features (like RSI, SMA, MACD) are important when making predictions.

If some features are clearly more important than others, it means the model is more confident in its prediction.

📊 How to read it:
High confidence (e.g., 80%) → The model relies strongly on a few useful indicators.

Low confidence (e.g., 40%) → The model is unsure, using many indicators with weak signals.

⚠️ Note:
This is not a real "probability" — it’s just a custom way to measure how clear the model’s decision is.


 1. RSI (Relative Strength Index)
What it tells:
How strong or weak a stock is based on recent price changes.

Simple meaning:

RSI near 70 = Stock may be overbought (might fall soon)

RSI near 30 = Stock may be oversold (might rise soon)

Think of it like:
A “mood meter” — Is the stock too hot or too cold?

📊 2. SMA (Simple Moving Average)
What it tells:
The average closing price over the last few days.

Examples:

SMA 20 = Average price over 20 days

SMA 50 = Average over 50 days

Why it's useful:
It helps to smooth out short-term ups and downs to see the overall trend.

Think of it like:
A “trend line” showing where the stock is heading.

📉 3. MACD (Moving Average Convergence Divergence)
What it tells:
If the momentum (speed of price change) is going up or down.

MACD parts:

MACD Line: Fast-moving average

Signal Line: Slower average

When MACD goes above signal → 📈 Buy signal

When MACD goes below signal → 📉 Sell signal

Think of it like:
A “speed monitor” — Is the stock moving faster up or down?

 1. MAE (Mean Absolute Error)
What it means: The average of all absolute errors (i.e., how far off the predictions were from the actual prices, without considering direction).

Interpretation:
"On average, the prediction is off by ₹___."

💥 2. MSE (Mean Squared Error)
What it means: Similar to MAE, but squares the errors, so bigger mistakes are punished more.
Interpretation:
Larger value = worse model. Useful when you want to penalize big errors more.

📊 3. R² (R-squared or Coefficient of Determination)
What it means: Shows how well your model explains the variability in stock prices.

Range: 0 to 1 (can be negative if model is worse than guessing)

Interpretation:

R² = 1 → Perfect prediction

R² = 0 → No better than guessing the average

R² < 0 → Worse than guessing


