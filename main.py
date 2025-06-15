from model_utils import (
    load_data,
    train_model,
    predict_next_day,
    get_prediction_details,
    plot_data
)
from visualizer import plot_data as visualizer_plot_data
from datetime import datetime, timedelta

def validate_indian_stock_symbol(symbol):
    exchanges = ['.NS', '.BO']
    if any(symbol.endswith(exchange) for exchange in exchanges):
        return symbol
    return f"{symbol}.NS"

def get_user_input():
    while True:
        symbol = input("Enter the stock symbol: ").strip().upper()
        if symbol:
            return symbol
        print("Please enter a valid stock symbol.")

def display_prediction(prediction_details):
    print("\n=== Prediction Results ===")
    print(f"Current Price: ₹{prediction_details['current_price']:.2f}")
    print(f"Predicted Direction: {'UP' if prediction_details['prediction'] == 1 else 'DOWN'}")
    print(f"Probability of Price Increase: {prediction_details['probability']:.2%}")
    print(f"Confidence: {prediction_details['confidence']:.2%}")

def main():
    try:
        print("=== Indian Stock Price Predictor ===")
        print("Enter stock symbol without exchange suffix (e.g., RELIANCE, TCS, INFY)")
        symbol = get_user_input()
        
        print(f"\nFetching data for {symbol}...")
        df = load_data(symbol)
        
        if df is not None:
            print("\nData loaded successfully:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Train model and get predictions
            model_data = train_model(df)
            prediction, confidence = predict_next_day(df, model_data)
            
            # Create prediction details
            prediction_details = {
                'current_price': df['Close'].iloc[-1],
                'prediction': prediction,
                'probability': confidence if prediction == 1 else 1 - confidence,
                'confidence': confidence
            }
            
            # Display prediction results
            display_prediction(prediction_details)
            
            # Plot the data with the symbol
            plot_data(df, symbol)
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Check if the stock symbol is valid. Example: RELIANCE, TCS, INFY")

if __name__ == "__main__":
    main()
