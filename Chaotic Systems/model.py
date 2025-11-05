# Import libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fetch data from Yahoo Finance
ticker = '^GSPC'  # S&P 500 ticker
start_date = '2000-01-01'
end_date = '2020-01-01'    # Fetch data from 2000-2020

data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']]

# Split data into training, validation, and test sets
train_size = int(len(data) * 0.7)  # 70% for training
validation_size = int(len(data) * 0.2)  # 20% for validation

train_data = data.iloc[:train_size]
validation_data = data.iloc[train_size:train_size + validation_size]
test_data = data.iloc[train_size + validation_size:]  # remaining 10% for testing

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
validation_data_scaled = scaler.transform(validation_data)
test_data_scaled = scaler.transform(test_data)

# Function to prepare data
def prepare_data(scaled_data, time_step):
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60

# Prepare training and validation data
X_train, y_train = prepare_data(train_data_scaled, time_step)
X_val, y_val = prepare_data(validation_data_scaled, time_step)

# Prepare full dataset for test data prediction
full_data = pd.concat([train_data, validation_data, test_data])
scaled_full_data = scaler.transform(full_data)
X_full, y_full = prepare_data(scaled_full_data, time_step)
X_test = X_full[-len(test_data):]  # Only take test-sized X_test
y_test = y_full[-len(test_data):]  # Only take test-sized y_test

# Reshape inputs to 3D for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the RNN model with Dropout for regularization
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(LSTM(50))
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Making predictions
predictions = model.predict(X_test).flatten()

# Inverse transform predictions and actual test values
predictions_inverse = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate metrics
mae = mean_absolute_error(y_true, predictions_inverse)
mse = mean_squared_error(y_true, predictions_inverse)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Visualization: Show a limited amount of historical data before predictions
plt.figure(figsize=(14, 7))

# Define ranges for better readability
historical_limit = len(data) - len(test_data) - 100  # Show 100 data points before predictions start
data_range = np.arange(historical_limit, len(data))
prediction_range = np.arange(len(data) - len(predictions), len(data))

# Plot historical and predicted data
plt.plot(data_range, data.iloc[historical_limit:].values, label='Historical Data', color='blue')
plt.plot(prediction_range, predictions_inverse, label='Predicted Values', color='red')

# Add labels and legend
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


