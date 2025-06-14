import matplotlib.pyplot as plt

def plot_data(df, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.title(f"{symbol} - Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_true, y_pred, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f"{symbol} - Model Predictions vs Actual")
    plt.xlabel("Test Sample")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.show()
