import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_data(df, symbol):
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Price and Technical Indicators
        ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
        ax1.plot(df.index, df['SMA_50'], label='50-day SMA', color='orange', alpha=0.7)
        ax1.plot(df.index, df['SMA_200'], label='200-day SMA', color='red', alpha=0.7)
        
        # Add Bollinger Bands
        ax1.plot(df.index, df['BB_Upper'], '--', color='gray', alpha=0.5, label='BB Upper')
        ax1.plot(df.index, df['BB_Lower'], '--', color='gray', alpha=0.5, label='BB Lower')
        
        # Plot 2: Actual vs Predicted Directions
        # Calculate actual direction (1 if price went up, 0 if down)
        actual_direction = (df['Close'].shift(-1) > df['Close']).astype(int)
        actual_direction = actual_direction[:-1]  # Remove last row as we don't have next day's price
        
        # Plot actual directions
        for i in range(len(actual_direction)):
            if actual_direction.iloc[i] == 1:
                ax2.bar(df.index[i], 1, color='green', alpha=0.5, label='Actual Up' if i == 0 else "")
            else:
                ax2.bar(df.index[i], 1, color='red', alpha=0.5, label='Actual Down' if i == 0 else "")
        
        # Add labels and title
        ax1.set_title(f'{symbol} Stock Price and Technical Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Actual Price Direction')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Direction')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Down', 'Up'])
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in plot_data: {str(e)}")
        raise

def plot_predictions(df, predictions, symbol):
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Price and Technical Indicators
        ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
        ax1.plot(df.index, df['SMA_50'], label='50-day SMA', color='orange', alpha=0.7)
        ax1.plot(df.index, df['SMA_200'], label='200-day SMA', color='red', alpha=0.7)
        
        # Add Bollinger Bands
        ax1.plot(df.index, df['BB_Upper'], '--', color='gray', alpha=0.5, label='BB Upper')
        ax1.plot(df.index, df['BB_Lower'], '--', color='gray', alpha=0.5, label='BB Lower')
        
        # Plot 2: Actual vs Predicted Directions
        # Calculate actual direction
        actual_direction = (df['Close'].shift(-1) > df['Close']).astype(int)
        actual_direction = actual_direction[:-1]  # Remove last row
        
        # Plot actual and predicted directions
        for i in range(len(actual_direction)):
            # Plot actual direction
            if actual_direction.iloc[i] == 1:
                ax2.bar(df.index[i], 1, color='green', alpha=0.3, label='Actual Up' if i == 0 else "")
            else:
                ax2.bar(df.index[i], 1, color='red', alpha=0.3, label='Actual Down' if i == 0 else "")
            
            # Plot predicted direction
            if predictions[i] == 1:
                ax2.bar(df.index[i], 0.5, color='green', alpha=0.7, label='Predicted Up' if i == 0 else "")
            else:
                ax2.bar(df.index[i], 0.5, color='red', alpha=0.7, label='Predicted Down' if i == 0 else "")
        
        # Add labels and title
        ax1.set_title(f'{symbol} Stock Price and Technical Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Actual vs Predicted Price Direction')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Direction')
        ax2.set_yticks([0.25, 0.75])
        ax2.set_yticklabels(['Predicted', 'Actual'])
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in plot_predictions: {str(e)}")
        raise

def plot_confusion_matrix_bar(y_true, y_pred, symbol):
    try:
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create labels for the confusion matrix
        labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        values = cm.flatten()
        
        # Create color mapping
        colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#2ecc71']  # Green for correct, Red for incorrect
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        bars = plt.bar(labels, values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Customize plot
        plt.title(f'Prediction vs Actual Direction - {symbol}', pad=20)
        plt.xlabel('Prediction Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add accuracy information
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        plt.text(0.02, 0.98, f'Accuracy: {accuracy:.2%}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in plot_confusion_matrix_bar: {str(e)}")
        raise
