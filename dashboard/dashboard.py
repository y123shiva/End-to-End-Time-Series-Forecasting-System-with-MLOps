import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Set up project root and file paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# Data file
data_file = BASE_DIR / "data" / "financial_data.csv"

# Image files
img_daily_trend = BASE_DIR / "images" / "daily_price_trend.png"
img_rolling_avg = BASE_DIR / "images" / "price_rolling_avg.png"
img_price_change = BASE_DIR / "images" / "price_change_dist.png"

# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv(data_file, parse_dates=['Date'], dayfirst=True)
df = df.sort_values('Date')
df['7d_avg'] = df['Price'].rolling(7).mean()
df['Price_diff'] = df['Price'].diff()

# -----------------------------
# Streamlit Dashboard
# -----------------------------
st.title("Financial Price Time Series Dashboard")

# Raw data
st.subheader("Raw Data")
st.dataframe(df)

# Static images
st.subheader("Daily Price Trend")
st.image(img_daily_trend, use_column_width=True)

st.subheader("Price Trend with 7-day Rolling Average")
st.image(img_rolling_avg, use_column_width=True)

st.subheader("Distribution of Daily Price Changes")
st.image(img_price_change, use_column_width=True)

# Interactive plot
st.subheader("Interactive Price Trend")
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Price', marker='o', label='Price', ax=ax)
sns.lineplot(data=df, x='Date', y='7d_avg', label='7-day Avg', color='red', ax=ax)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
