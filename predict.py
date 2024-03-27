import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def predict_stock_price(stock='AAPL', start='2023-01-01', end='2023-03-01'):
    # Download stock data
    df = yf.download(stock, start=start, end=end)
    
    # Use closing price as the feature
    df['Target'] = df['Close'].shift(-1)  # Next day's closing price as the target value
    df = df[['Close', 'Target']].dropna()  # Drop rows with NaN values

    # Split into training and testing sets
    X = df[['Close']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    predictions = model.predict(X_test)

    # Plot the predicted results vs actual results
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, predictions, label='Predicted Prices', color='red')
    plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
    plt.title(f'{stock} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price USD')
    plt.legend()
    plt.show()
    
    # Predict the next day's price based on the last available data point
    last_price = df[['Close']].iloc[-1].values.reshape(-1,1)
    future_price = model.predict(last_price)
    print(f"Based on the last available price, the predicted next closing price is: {future_price[0]}")

# Call the function
predict_stock_price(stock='AAPL', start='2024-02-25', end='2024-03-27')
