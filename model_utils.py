import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import ta
from visualizer import plot_data, plot_predictions, plot_confusion_matrix_bar
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import json

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
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Calculate CCI (Commodity Channel Index)
        print("Calculating CCI...")
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
        
        # Calculate ADX (Average Directional Index)
        print("Calculating ADX...")
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx.adx()
        df['ADX_Pos'] = adx.adx_pos()
        df['ADX_Neg'] = adx.adx_neg()
        
        # Calculate Williams %R
        print("Calculating Williams %R...")
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
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
        
        # Add sentiment features
        print("\n=== Adding Sentiment Features ===")
        df = add_sentiment_features(df, symbol)
        
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
        print(f"Debug: Input DataFrame shape: {df.shape}")
        
        # Create a copy of the DataFrame
        df = df.copy()
        print("Debug: DataFrame copied")
        
        # Create binary target: 1 if next day's Close is higher, 0 otherwise
        print("Debug: Creating target column")
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        print(f"Debug: Target column created. Value counts:\n{df['Target'].value_counts()}")
        
        # Drop the last row since we don't have the next day's price for it
        print("Debug: Dropping last row")
        df = df.iloc[:-1]
        print(f"Debug: DataFrame shape after dropping last row: {df.shape}")
        
        # Replace infinite values with NaN
        print("Debug: Replacing infinite values")
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with 0
        print("Debug: Filling NaN values")
        df = df.fillna(0)
        
        # Select features and target
        features = ['Returns', 'Returns_5', 'Returns_20', 'SMA_10', 'SMA_50', 'SMA_200',
                   'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower',
                   'BB_Middle', 'BB_Width', 'CCI', 'ADX', 'ADX_Pos', 'ADX_Neg', 'Williams_R',
                   'Volume_SMA', 'Volume_ROC', 'Stoch', 'Stoch_Signal']
        
        target = 'Target'
        print(f"Debug: Selected {len(features)} features")
        
        # Create feature matrix X and target vector y
        print("Debug: Creating feature matrix and target vector")
        X = df[features].values
        y = df[target].values
        print(f"Debug: X shape: {X.shape}")
        print(f"Debug: y shape: {y.shape}")
        
        # Initialize scaler for features only (no need to scale binary target)
        print("Debug: Initializing feature scaler")
        feature_scaler = MinMaxScaler()
        
        # Scale features
        print("Debug: Scaling features")
        X_scaled = feature_scaler.fit_transform(X)
        print(f"Debug: X_scaled shape: {X_scaled.shape}")
        
        # Create sequences for time series
        print("Debug: Creating sequences")
        X_sequences = []
        y_sequences = []
        sequence_length = 10
        
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:i + sequence_length])
            y_sequences.append(y[i + sequence_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        print(f"Debug: X_sequences shape: {X_sequences.shape}")
        print(f"Debug: y_sequences shape: {y_sequences.shape}")
        
        # Split into training and testing sets
        print("Debug: Splitting into train/test sets")
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
        
        return X_train, X_test, y_train, y_test, feature_scaler, features
        
    except Exception as e:
        print(f"\nError in prepare_data: {str(e)}")
        print("\nDetailed Debug Information:")
        print("=== Variable States at Error ===")
        if 'df' in locals():
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
        if 'X' in locals():
            print(f"X shape: {X.shape}")
        if 'y' in locals():
            print(f"y shape: {y.shape}")
        if 'X_scaled' in locals():
            print(f"X_scaled shape: {X_scaled.shape}")
        if 'X_sequences' in locals():
            print(f"X_sequences shape: {X_sequences.shape}")
        if 'y_sequences' in locals():
            print(f"y_sequences shape: {y_sequences.shape}")
        print("\n=== Stack Trace ===")
        import traceback
        traceback.print_exc()
        raise

def train_model(df):
    try:
        print("\n=== Training Model ===")
        print("Debug: Starting train_model function")
        
        # Prepare data
        print("Debug: Calling prepare_data")
        X_train, X_test, y_train, y_test, feature_scaler, features = prepare_data(df)
        print("Debug: prepare_data completed")
        print(f"Debug: X_train shape: {X_train.shape}")
        print(f"Debug: y_train shape: {y_train.shape}")
        
        # Initialize and train XGBoost classifier
        print("Debug: Initializing XGBoost classifier")
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='logloss',
            objective='binary:logistic'
        )
        
        # Reshape X_train for XGBoost (it expects 2D input)
        print("Debug: Reshaping data for XGBoost")
        print(f"Debug: Original X_train shape: {X_train.shape}")
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        print(f"Debug: Reshaped X_train_2d shape: {X_train_2d.shape}")
        
        # Ensure y_train and y_test are 1D arrays
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        print(f"Debug: y_train shape after ravel: {y_train.shape}")
        print(f"Debug: y_test shape after ravel: {y_test.shape}")
        
        print("\n=== Data Shapes ===")
        print(f"X_train_2d shape: {X_train_2d.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test_2d shape: {X_test_2d.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Train the model
        print("Debug: Starting model training")
        model.fit(X_train_2d, y_train)
        print("Debug: Model training completed")
        
        # Make predictions
        print("Debug: Making predictions")
        y_pred = model.predict(X_test_2d)
        y_pred_proba = model.predict_proba(X_test_2d)[:, 1]
        print(f"Debug: y_pred shape: {y_pred.shape}")
        print(f"Debug: y_pred_proba shape: {y_pred_proba.shape}")
        
        # Calculate classification metrics
        print("Debug: Calculating metrics")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Plot confusion matrix bar plot
        plot_confusion_matrix_bar(y_test, y_pred, df.name if hasattr(df, 'name') else 'Stock')
        
        # Calculate feature importance
        print("Debug: Calculating feature importance")
        importance = model.feature_importances_
        
        # Create feature names for the flattened sequence
        sequence_length = 10
        flattened_features = []
        for i in range(sequence_length):
            for feature in features:
                flattened_features.append(f"{feature}_t-{sequence_length-i-1}")
        
        print(f"Debug: Number of features: {len(features)}")
        print(f"Debug: Number of flattened features: {len(flattened_features)}")
        print(f"Debug: Length of importance array: {len(importance)}")
        
        feature_importance = pd.DataFrame({
            'Feature': flattened_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\n=== Feature Importance ===")
        print(feature_importance.head(10))  # Show top 10 most important features
        
        # Store the model and necessary components for prediction
        print("Debug: Creating model_data dictionary")
        model_data = {
            'model': model,
            'feature_scaler': feature_scaler,
            'features': features,
            'sequence_length': sequence_length
        }
        print("Debug: model_data created successfully")
        
        print("Debug: Returning model_data")
        return model_data
        
    except Exception as e:
        print(f"\nError in train_model: {str(e)}")
        print("\nDetailed Debug Information:")
        print("=== Variable States at Error ===")
        if 'X_train' in locals():
            print(f"X_train shape: {X_train.shape}")
            print(f"X_train type: {type(X_train)}")
        if 'y_train' in locals():
            print(f"y_train shape: {y_train.shape}")
            print(f"y_train type: {type(y_train)}")
        if 'X_train_2d' in locals():
            print(f"X_train_2d shape: {X_train_2d.shape}")
            print(f"X_train_2d type: {type(X_train_2d)}")
        if 'importance' in locals():
            print(f"importance length: {len(importance)}")
        if 'flattened_features' in locals():
            print(f"flattened_features length: {len(flattened_features)}")
        print("\n=== Stack Trace ===")
        import traceback
        traceback.print_exc()
        raise

def predict_next_day(df, model_data):
    try:
        print("\n=== Making Next Day Prediction ===")
        
        # Extract components from model_data
        model = model_data['model']
        feature_scaler = model_data['feature_scaler']
        features = model_data['features']
        sequence_length = model_data['sequence_length']
        
        # Get the last sequence of data
        last_sequence = df[features].values[-sequence_length:]
        
        # Scale the features
        last_sequence_scaled = feature_scaler.transform(last_sequence)
        
        # Reshape for prediction (flatten the sequence)
        last_sequence_2d = last_sequence_scaled.reshape(1, -1)
        
        print("\n=== Input Data Shape ===")
        print(f"Last sequence shape: {last_sequence.shape}")
        print(f"Last sequence scaled shape: {last_sequence_scaled.shape}")
        print(f"Last sequence 2D shape: {last_sequence_2d.shape}")
        
        # Get prediction probability
        prediction_proba = model.predict_proba(last_sequence_2d)[0][1]
        
        # Get binary prediction (1 if probability > 0.5, else 0)
        prediction = 1 if prediction_proba > 0.5 else 0
        
        # Calculate confidence based on prediction probability
        confidence = prediction_proba if prediction == 1 else 1 - prediction_proba
        
        print("\n=== Prediction Results ===")
        print(f"Predicted Direction: {'UP' if prediction == 1 else 'DOWN'}")
        print(f"Probability of Price Increase: {prediction_proba:.2%}")
        print(f"Confidence: {confidence:.2%}")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"\nError in predict_next_day: {str(e)}")
        print("\nDebug Information:")
        if 'last_sequence' in locals():
            print(f"Last sequence shape: {last_sequence.shape}")
        if 'last_sequence_scaled' in locals():
            print(f"Last sequence scaled shape: {last_sequence_scaled.shape}")
        if 'last_sequence_2d' in locals():
            print(f"Last sequence 2D shape: {last_sequence_2d.shape}")
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

def get_news_sentiment(symbol, days_back=30):
    try:
        print(f"\n=== Fetching News for {symbol} ===")
        
        # Remove .NS suffix if present for news search
        search_symbol = symbol.replace('.NS', '')
        
        # Get news from Yahoo Finance
        stock = yf.Ticker(symbol)
        news = stock.news
        
        if not news:
            print("No news found")
            return pd.Series(0, index=pd.date_range(end=datetime.now(), periods=days_back))
        
        # Convert news to DataFrame
        news_df = pd.DataFrame(news)
        news_df['date'] = pd.to_datetime(news_df['providerPublishTime'], unit='s')
        news_df['date'] = news_df['date'].dt.date
        
        # Calculate sentiment for each headline
        news_df['sentiment'] = news_df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        
        # Group by date and calculate average sentiment
        daily_sentiment = news_df.groupby('date')['sentiment'].mean()
        
        # Create a date range for the last n days
        date_range = pd.date_range(end=datetime.now().date(), periods=days_back)
        
        # Reindex the sentiment series to include all dates
        daily_sentiment = daily_sentiment.reindex(date_range, fill_value=0)
        
        print(f"Processed {len(news_df)} news articles")
        print(f"Date range: {daily_sentiment.index[0]} to {daily_sentiment.index[-1]}")
        
        return daily_sentiment
        
    except Exception as e:
        print(f"Error in get_news_sentiment: {str(e)}")
        return pd.Series(0, index=pd.date_range(end=datetime.now(), periods=days_back))

def add_sentiment_features(df, symbol):
    try:
        print("\n=== Adding Sentiment Features ===")
        
        # Get news sentiment
        sentiment_series = get_news_sentiment(symbol)
        
        # Convert sentiment series index to datetime
        sentiment_series.index = pd.to_datetime(sentiment_series.index)
        
        # Merge sentiment with main DataFrame
        df['News_Sentiment'] = df.index.map(lambda x: sentiment_series.get(x.date(), 0))
        
        # Add rolling sentiment features
        df['Sentiment_MA5'] = df['News_Sentiment'].rolling(window=5, min_periods=1).mean()
        df['Sentiment_MA10'] = df['News_Sentiment'].rolling(window=10, min_periods=1).mean()
        
        # Fill NaN values with 0
        df.fillna(0, inplace=True)
        
        print("Sentiment features added successfully")
        return df
        
    except Exception as e:
        print(f"Error in add_sentiment_features: {str(e)}")
        return df
