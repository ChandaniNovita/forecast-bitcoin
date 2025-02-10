import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load the dataset
st.title("Bitcoin Price Forecasting App")

uploaded_file = st.file_uploader("Upload your BTC Dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Ensure essential columns
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("Dataset must have 'Date' and 'Close' columns.")
    else:
        # Data preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)

        # Prepare closing price for forecasting
        data = df[['Close']].values

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare training data
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]

        # Function to create sequences
        def create_sequences(data, window_size=60):
            x, y = [], []
            for i in range(window_size, len(data)):
                x.append(data[i - window_size:i, 0])
                y.append(data[i, 0])
            return np.array(x), np.array(y)

        x_train, y_train = create_sequences(train_data)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        with st.spinner("Training LSTM model..."):
            model.fit(x_train, y_train, batch_size=32, epochs=5)

        # Prepare testing data
        test_data = scaled_data[train_size - 60:]
        x_test, y_test = create_sequences(test_data)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Predict and inverse transform the results
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Plotting the results
        st.subheader("Forecast vs Actual Prices")
        train = df[:train_size]
        valid = df[train_size:]
        valid['Predictions'] = predictions

        plt.figure(figsize=(12, 6))
        plt.plot(train['Close'], label='Training Data')
        plt.plot(valid['Close'], label='Actual Prices', color='blue')
        plt.plot(valid['Predictions'], label='Predicted Prices', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD')
        plt.legend()
        st.pyplot(plt.gcf())

        st.success("Model training and forecasting complete!")
