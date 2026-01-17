import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# ------------------ 1. PAGE CONFIG ------------------
st.set_page_config(page_title="Apple Stock Trend Prediction", layout="wide")

# CUSTOM STYLING: Bold white text on black backgrounds for METRICS ONLY
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    
    /* KPI Metric Styling: Black background, White bold text */
    [data-testid="stMetricValue"] {
        background-color: black !important;
        color: white !important;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold !important;
        border: 1px solid #444;
    }
    [data-testid="stMetricLabel"] {
        background-color: black !important;
        color: white !important;
        padding: 5px 10px;
        border-radius: 5px 5px 0 0;
        font-weight: bold !important;
    }
    
    /* Table Styling: Cleaned up to remove forced black backgrounds */
    [data-testid="stTable"] {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------ 2. HEADER ------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=100)
st.title("Apple Stock Trend Prediction (2019 to 2024)")

# ------------------ 3. SIDEBAR / OPTIONS ------------------
st.sidebar.header("Settings")
theme = st.sidebar.selectbox("Select Chart Theme", ["Dark", "Light", "Default"])
if theme == "Dark":
    plt.style.use("dark_background")
elif theme == "Light":
    plt.style.use("ggplot")

# ------------------ 4. DATA LOADING ------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("AAPL.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df.loc['2019-01-01':'2024-12-31']
        return df
    except FileNotFoundError:
        st.error("AAPL.csv not found. Please ensure the file is in the same folder.")
        return None

df = load_data()

if df is not None:
    # ------------------ 5. DASHBOARD SUMMARY ------------------
    st.subheader("📊 Dashboard Summary")
    col_a, col_b = st.columns(2)

    try:
        latest_year = df.index.year.max()
        avg_close = df.loc[str(latest_year)]['Close'].mean()
        with col_a:
            st.info(f"**Apple {latest_year} Summary** - Avg Closing: ${avg_close:.2f}")
    except Exception:
        with col_a:
            st.info(f"**Overall Summary** - Avg Closing: ${df['Close'].mean():.2f}")

    with col_b:
        st.info(f"**Current Status** - Last Close: ${df['Close'].iloc[-1]:.2f}")

    # DATA TABLE (Fixed: Background color removed)
    st.write("### Data Summary (2019-2024)")
    summary_table = df['Close'].resample('YE').mean().to_frame()
    summary_table['High'] = df['High'].resample('YE').max()
    summary_table['Low'] = df['Low'].resample('YE').min()
    summary_table['Volume'] = df['Volume'].resample('YE').mean()
    
    # Rename columns
    summary_table.columns = ['CLOSING', 'HIGH', 'LOW', 'VOLUME']
    
    # Render table without the black background property
    st.dataframe(summary_table, use_container_width=True)

    # STYLED KPI METRICS (Kept black background as per your previous request)
    m1, m2, m3 = st.columns(3)
    m1.metric("CLOSING PRICE", f"${df['Close'].iloc[-1]:.2f}")
    m2.metric("HIGH", f"${df['High'].max():.2f}")
    m3.metric("LOW", f"${df['Low'].min():.2f}")

    st.markdown("---")

    # ------------------ 6. VISUALIZATIONS ------------------
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()

    st.subheader("Stock Price Analysis")
    tab1, tab2, tab3 = st.tabs(["General Trends", "Moving Averages", "Volume Analysis"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df.Close, color='skyblue', label='Closing Price')
        ax1.set_title("Closing Price Over Time")
        st.pyplot(fig1)

    with tab2:
        st.write("### Closing Price with 100MA & 200MA")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df.Close, 'b', label='Original Price', alpha=0.5)
        ax2.plot(ma100, 'r', label='100 MA')
        ax2.plot(ma200, 'g', label='200 MA')
        ax2.legend()
        st.pyplot(fig2)

    with tab3:
        year_to_filter = st.slider('Select Year for Volume Check', 2019, 2024, 2024)
        filtered_df = df[df.index.year == year_to_filter]
        st.bar_chart(filtered_df['Volume'])

    # ------------------ 7. PREDICTION LOGIC ------------------
    st.markdown("---")
    if st.button("🚀 Train and Predict Model"):
        with st.spinner("Training LSTM Model..."):
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
            
            scaler = MinMaxScaler(feature_range=(0,1))
            data_training_array = scaler.fit_transform(data_training)

            x_train, y_train = [], []
            for i in range(100, data_training_array.shape[0]):
                x_train.append(data_training_array[i-100:i])
                y_train.append(data_training_array[i, 0])
            
            x_train, y_train = np.array(x_train), np.array(y_train)

            model = Sequential([
                LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=60, activation='relu', return_sequences=True),
                Dropout(0.3),
                LSTM(units=80, activation='relu'),
                Dropout(0.4),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=10, verbose=0)

            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])
            
            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)
            
            scale_factor = 1/scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            st.subheader("📈 Prediction vs Actual")
            fig3, ax3 = plt.subplots(figsize=(12,6))
            ax3.plot(y_test, 'b', label = 'Original Price')
            ax3.plot(y_predicted, 'r', label = 'Predicted Price')
            ax3.legend()
            st.pyplot(fig3)

            # ------------------ 8. FUTURE PREDICTION ------------------
            st.subheader("🔮 Future Stock Price Prediction (2026 - 2027)")
            future_days = 730 
            last_100_days_scaled = input_data[-100:].reshape(1, 100, 1)
            future_predictions = []
            curr_step = last_100_days_scaled
            
            progress_bar = st.progress(0)
            
            for i in range(future_days):
                next_val = model.predict(curr_step, verbose=0)
                future_predictions.append(next_val[0,0])
                next_val_reshaped = next_val.reshape(1, 1, 1)
                curr_step = np.append(curr_step[:, 1:, :], next_val_reshaped, axis=1)
                progress_bar.progress((i + 1) / future_days)

            future_prices = np.array(future_predictions).reshape(-1, 1) * scale_factor
            future_dates = pd.date_range(start="2026-01-01", periods=future_days, freq='D')
            
            prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices.flatten()})
            prediction_df.set_index('Date', inplace=True)

            fig4, ax4 = plt.subplots(figsize=(12, 5))
            ax4.plot(prediction_df['Predicted_Price'], color='orange', label="Forecast (2026-2027)")
            ax4.legend()
            st.pyplot(fig4)

            csv_data = prediction_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download 2026-2027 Prediction CSV",
                data=csv_data,
                file_name='AAPL_Forecast_2026_2027.csv',
                mime='text/csv',
            )