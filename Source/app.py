# newapptest.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Load the trained ARIMA model
@st.cache_resource
def load_trained_model():
    model = joblib.load('Data/arima_model.pkl')
    return model

# Streamlit App
st.set_page_config(page_title="Oil Production Forecast", layout="centered")

st.title("🛢️ Oil Production Forecasting App")

try:
    data = pd.read_csv('Data/economic_data.csv', comment='#')

    # Parse Date column
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.dropna(subset=['Date', 'OILPRODUS'])

    # Sort by Date to ensure proper plotting and forecasting
    data = data.sort_values(by='Date')

    last_date = data['Date'].max()
    st.caption(f"📅 Last data point: {last_date.strftime('%B %Y')}")

    model = load_trained_model()

    # User Inputs
    st.subheader("🔮 Forecast Future Oil Production")
    col1, col2 = st.columns(2)

    with col1:
        short_term_steps = st.number_input(
            "Short-Term Forecast (months)",
            min_value=1,
            max_value=60,
            value=12,
            step=1
        )
    with col2:
        long_term_steps = st.number_input(
            "Long-Term Forecast (months)",
            min_value=1,
            max_value=120,
            value=24,
            step=1
        )

    if st.button("🚀 Forecast"):
        # Forecast start from the next month after the last data point
        forecast_start_date = last_date + pd.DateOffset(months=1)

        # Forecast
        short_term_forecast = model.forecast(steps=short_term_steps)
        long_term_forecast = model.forecast(steps=long_term_steps)

        # Create date ranges
        short_term_dates = pd.date_range(start=forecast_start_date, periods=short_term_steps, freq='MS')
        long_term_dates = pd.date_range(start=forecast_start_date, periods=long_term_steps, freq='MS')

        # Forecast DataFrames
        short_forecast_df = pd.DataFrame({
            "Date": short_term_dates,
            "Forecasted_Oil_Production": np.round(short_term_forecast, 2)
        })

        long_forecast_df = pd.DataFrame({
            "Date": long_term_dates,
            "Forecasted_Oil_Production": np.round(long_term_forecast, 2)
        })

        # 📈 Short-Term Forecast Plot
        st.subheader("📈 Short-Term Forecast")
        filtered_df = data[data['Date'] >= pd.to_datetime('2000-01-01')]
        fig_short = go.Figure()
        fig_short.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['OILPRODUS'],
            mode='lines',
            name='Actual Production (Historical)',
            line=dict(color='blue')
        ))
        fig_short.add_trace(go.Scatter(
            x=short_term_dates,
            y=short_term_forecast,
            mode='lines',
            name='Short-Term Forecast',
            line=dict(color='red')
        ))
        fig_short.update_layout(
            xaxis_title="Date",
            yaxis_title="Oil Production",
            legend_title="Legend",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_short, use_container_width=True)

        # 📋 Short-Term Forecast Table
        st.subheader("📋 Short-Term Forecasted Values")
        st.dataframe(short_forecast_df)

        # ⬇️ Download Short-Term
        csv_short = short_forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Short-Term Forecasted Data as CSV",
            data=csv_short,
            file_name='short_term_forecast.csv',
            mime='text/csv'
        )

        st.divider()

        # 📈 Long-Term Forecast Plot
        st.subheader("📈 Long-Term Forecast")
        fig_long = go.Figure()
        fig_long.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['OILPRODUS'],
            mode='lines',
            name='Actual Production (Historical)',
            line=dict(color='blue')
        ))
        fig_long.add_trace(go.Scatter(
            x=long_term_dates,
            y=long_term_forecast,
            mode='lines',
            name='Long-Term Forecast',
            line=dict(color='green')
        ))
        fig_long.update_layout(
            xaxis_title="Date",
            yaxis_title="Oil Production",
            legend_title="Legend",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_long, use_container_width=True)

        # 📋 Long-Term Forecast Table
        st.subheader("📋 Long-Term Forecasted Values")
        st.dataframe(long_forecast_df)

        # ⬇️ Download Long-Term
        csv_long = long_forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Long-Term Forecasted Data as CSV",
            data=csv_long,
            file_name='long_term_forecast.csv',
            mime='text/csv'
        )

except FileNotFoundError:
    st.error("❗ The file 'economic_data.csv' was not found. Please ensure it is placed in the correct path.")
except KeyError:
    st.error("❗ Column 'Date' or 'OILPRODUS' not found in economic_data.csv. Please check your file.")
