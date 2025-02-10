import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Simulasi data historis harga Bitcoin
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
prices = 20000 + np.cumsum(np.random.normal(0, 200, 365))  # Simulasi harga Bitcoin
btc_data = pd.DataFrame({'Date': dates, 'Close': prices})

# Streamlit UI
st.title("Bitcoin Price Prediction System")
st.write("Masukkan tanggal prediksi untuk mengetahui apakah harga Bitcoin akan naik atau turun.")

# Input prediksi
prediction_date = st.date_input("Pilih tanggal prediksi (maksimal 30 hari ke depan):")
prediction_days = (prediction_date - pd.Timestamp.today().normalize()).days

if prediction_days < 0 or prediction_days > 30:
    st.warning("Tanggal prediksi harus dalam 0 hingga 30 hari ke depan.")
else:
    # Preprocessing data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(btc_data[['Close']].values)

    # Membuat data sequence untuk LSTM
    window_size = 60

    def create_sequences(data, window_size):
        x = []
        for i in range(window_size, len(data)):
            x.append(data[i - window_size:i, 0])
        return np.array(x)

    x_data = create_sequences(scaled_data, window_size)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

    # Model LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_data.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    # Compile dan training
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_data, scaled_data[window_size:], epochs=1, batch_size=32, verbose=0)

    # Prediksi harga
    last_sequence = scaled_data[-window_size:]
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
    prediction_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]

    # Menentukan aksi beli/jual
    last_price = btc_data['Close'].iloc[-1]
    action = "Beli" if prediction > last_price else "Jual"
    st.subheader(f"Prediksi Harga pada {prediction_date}: ${prediction:.2f}")
    st.write(f"Aksi yang disarankan: **{action}**")

    # Plot hasil
    future_dates = pd.date_range(start=btc_data['Date'].iloc[-1], periods=31, freq='D')[1:]
    predicted_prices = np.append(prices, prediction)

    plt.figure(figsize=(12, 6))
    plt.plot(btc_data['Date'], btc_data['Close'], label='Harga Historis')
    plt.plot(future_dates[:1], [prediction], 'ro', label='Prediksi Harga')
    plt.title('Pergerakan Harga Bitcoin')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga (USD)')
    plt.legend()
    st.pyplot(plt.gcf())
