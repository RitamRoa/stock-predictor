import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import ta
from visualizer import plot_data, plot_predictions

def add_technical_indicators(df):
    try:
        print("\n=== Starting Technical Indicators Calculation ===")
        print(f"Input DataFrame Info:")
        print(f"- Shape: {df.shape}")
        print(f"- Columns: {df.columns.tolist()}")
        print(f"- Types:\n{df.dtypes}")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Debug each column's shape and type
        print("\n=== Column Analysis ===")
        for col in df.columns:
            print(f"\nAnalyzing column: {col}")
            print(f"- Type: {type(df[col])}")
            print(f"- Values type: {type(df[col].values)}")
            print(f"- Values shape: {df[col].values.shape}")
            print(f"- Values ndim: {df[col].values.ndim}")
            
            # Convert any 2D arrays to 1D Series
            if isinstance(df[col].values, np.ndarray) and df[col].values.ndim == 2:
                print(f"Converting {col} from 2D to 1D")
                df[col] = pd.Series(df[col].values.flatten(), index=df.index)
                print(f"- New shape: {df[col].values.shape}")
        
        print("\n=== Calculating Technical Indicators ===")
        
        # Calculate returns
        print("Calculating returns...")
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5'] = df['Returns'].rolling(window=5, min_periods=1).mean()
        df['Returns_20'] = df['Returns'].rolling(window=20, min_periods=1).mean()
        
        # Calculate SMAs
        print("Calculating SMAs...")
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # Calculate RSI
        print("Calculating RSI...")
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Calculate MACD
        print("Calculating MACD...")
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Calculate Bollinger Bands
        print("Calculating Bollinger Bands...")
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        print("Calculating Volume indicators...")
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_ROC'] = ta.momentum.roc(df['Volume'], window=10)
        
        # Stochastic Oscillator
        print("Calculating Stochastic Oscillator...")
        df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_Signal'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Fill NaN values with 0
        print("\nFilling NaN values...")
        df.fillna(0, inplace=True)
        
        print("\n=== Final DataFrame Info ===")
        print(f"- Shape: {df.shape}")
        print(f"- Columns: {df.columns.tolist()}")
        print(f"- Types:\n{df.dtypes}")
        
        return df
    except Exception as e:
        print(f"\n=== Error in add_technical_indicators ===")
        print(f"Error message: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"DataFrame Info at error:")
        print(f"- Shape: {df.shape}")
        print(f"- Columns: {df.columns.tolist()}")
        print(f"- Types:\n{df.dtypes}")
        raise

def load_data(symbol, period='5y', interval='1d'):
    try:
        # Add .NS suffix for Indian stocks if not already present
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
            
        print(f"\n=== Downloading Data for {symbol} ===")
        print(f"Parameters: Period={period}, Interval={interval}")
        
        df = yf.download(symbol, period=period, interval=interval)
        
        if df.empty:
            raise ValueError("No data returned. Check stock symbol.")
            
        print("\n=== Downloaded Data Info ===")
        print(f"- Shape: {df.shape}")
        print(f"- Columns: {df.columns.tolist()}")
        print(f"- Types:\n{df.dtypes}")
        
        # Flatten MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            print("\nFlattening MultiIndex columns...")
            df.columns = [f"{col[0]}" for col in df.columns]
            print(f"New columns: {df.columns.tolist()}")
        
        # Debug each column's shape and type
        print("\n=== Column Analysis ===")
        for col in df.columns:
            print(f"\nAnalyzing column: {col}")
            print(f"- Type: {type(df[col])}")
            print(f"- Values type: {type(df[col].values)}")
            print(f"- Values shape: {df[col].values.shape}")
            print(f"- Values ndim: {df[col].values.ndim}")
            
            # Ensure all columns are 1D
            if isinstance(df[col].values, np.ndarray) and df[col].values.ndim == 2:
                print(f"Converting {col} from 2D to 1D")
                df[col] = pd.Series(df[col].values.flatten(), index=df.index)
                print(f"- New shape: {df[col].values.shape}")
        
        # Add technical indicators
        print("\n=== Adding Technical Indicators ===")
        df = add_technical_indicators(df)
        
        print("\n=== Final Data Info ===")
        print(f"- Shape: {df.shape}")
        print(f"- Columns: {df.columns.tolist()}")
        print(f"- Types:\n{df.dtypes}")
        
        return df
    except Exception as e:
        print(f"\n=== Error in load_data ===")
        print(f"Error message: {str(e)}")
        print(f"Error type: {type(e)}")
        if 'df' in locals():
            print(f"DataFrame Info at error:")
            print(f"- Shape: {df.shape}")
            print(f"- Columns: {df.columns.tolist()}")
            print(f"- Types:\n{df.dtypes}")
        raise

def prepare_data(df):
    try:
        print("\n=== Preparing Data ===")
        print(f"Input shape: {df.shape}")
        
        # Create a copy of the DataFrame
        df = df.copy()
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        # Select features and target
        features = ['Returns', 'Returns_5', 'Returns_20', 'SMA_10', 'SMA_50', 'SMA_200',
                   'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower',
                   'BB_Middle', 'Volume_SMA', 'Volume_ROC', 'Stoch', 'Stoch_Signal']
        
        target = 'Close'
        
        # Create feature matrix X and target vector y
        X = df[features].values
        y = df[target].values
        
        # Initialize scalers
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        # Scale features
        X_scaled = feature_scaler.fit_transform(X)
        
        # Scale target
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences for time series
        X_sequences = []
        y_sequences = []
        sequence_length = 10
        
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:i + sequence_length])
            y_sequences.append(y_scaled[i + sequence_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split into training and testing sets
        train_size = int(len(X_sequences) * 0.8)
        X_train = X_sequences[:train_size]
        X_test = X_sequences[train_size:]
        y_train = y_sequences[:train_size]
        y_test = y_sequences[train_size:]
        
        print("\n=== Data Preparation Complete ===")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_scaler, target_scaler, features
        
    except Exception as e:
        print(f"\nError in prepare_data: {str(e)}")
        print(f"DataFrame Info at error:")
        print(f"- Shape: {df.shape}")
        print(f"- Columns: {df.columns.tolist()}")
        print(f"- Types:\n{df.dtypes}")
        print("\nChecking for infinite values:")
        for col in df.columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                print(f"Column {col} has {inf_count} infinite values")
        raise

def train_model(df):
    try:
        print("\n=== Training Model ===")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler, features = prepare_data(df)
        
        # Initialize and train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Reshape X_train for XGBoost (it expects 2D input)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        print("Training XGBoost model...")
        model.fit(X_train_reshaped, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_reshaped)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate confidence based on R² score
        confidence = max(0, min(1, r2))  # Ensure confidence is between 0 and 1
        
        print("\n=== Model Training Complete ===")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        print(f"Confidence: {confidence:.2%}")
        
        return model, feature_scaler, target_scaler, X_test, y_test
        
    except Exception as e:
        print(f"\nError in train_model: {str(e)}")
        raise

def predict_next_day(df, model, feature_scaler, target_scaler):
    try:
        print("\n=== Making Next Day Prediction ===")
        
        # Get the last sequence of data
        features = ['Returns', 'Returns_5', 'Returns_20', 'SMA_10', 'SMA_50', 'SMA_200',
                   'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower',
                   'BB_Middle', 'Volume_SMA', 'Volume_ROC', 'Stoch', 'Stoch_Signal']
        
        # Get the last 10 days of data
        last_sequence = df[features].values[-10:]
        
        # Scale the features using the feature scaler
        last_sequence_scaled = feature_scaler.transform(last_sequence)
        
        # Reshape for prediction (10, 17) -> (1, 170)
        last_sequence_reshaped = last_sequence_scaled.reshape(1, -1)
        
        # Make prediction
        prediction_scaled = model.predict(last_sequence_reshaped)
        
        # Reshape for inverse transform
        prediction_scaled = prediction_scaled.reshape(-1, 1)
        
        # Inverse transform to get actual price
        prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Calculate confidence based on model's feature importance
        feature_importance = model.feature_importances_
        confidence = np.mean(feature_importance)
        
        print(f"Predicted price: ₹{prediction:.2f}")
        print(f"Confidence: {confidence:.2%}")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"\nError in predict_next_day: {str(e)}")
        print("Data shapes at error:")
        print(f"Last sequence shape: {last_sequence.shape}")
        print(f"Last sequence scaled shape: {last_sequence_scaled.shape}")
        print(f"Last sequence reshaped shape: {last_sequence_reshaped.shape}")
        raise

def get_prediction_details(df, prediction, confidence, X_test, y_test):
    try:
        current_price = df['Close'].iloc[-1]
        expected_change = ((prediction - current_price) / current_price) * 100
        
        # Calculate prediction interval based on test set standard deviation
        std_dev = np.std(y_test)
        lower_bound = prediction - (1.96 * std_dev)
        upper_bound = prediction + (1.96 * std_dev)
        
        return {
            'current_price': current_price,
            'predicted_price': prediction,
            'expected_change': expected_change,
            'prediction_interval': (lower_bound, upper_bound),
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"\nError in get_prediction_details: {str(e)}")
        raise
