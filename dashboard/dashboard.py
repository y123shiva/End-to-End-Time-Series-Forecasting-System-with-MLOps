import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import streamlit as st
# Load data
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "financial_data.csv")

df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
df = df.sort_values('Date')
df['7d_avg'] = df['Price'].rolling(7).mean()
df['Price_diff'] = df['Price'].diff()

st.title("Financial Price Time Series Dashboard")

# Show raw data
st.subheader("Raw Data")
st.dataframe(df)

# Display static images
st.subheader("Daily Price Trend")


# Get project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Correct image path
img_file = BASE_DIR / "images" / "daily_price_trend.png"

# Display image
st.image(img_file, use_column_width=True)

st.subheader("Price Trend with 7-day Rolling Average")
st.image("../images/price_rolling_avg.png", use_column_width=True)

st.subheader("Distribution of Daily Price Changes")
st.image("../images/price_change_dist.png", use_column_width=True)

# Interactive plot
st.subheader("Interactive Price Trend")
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Price', marker='o', label='Price', ax=ax)
sns.lineplot(data=df, x='Date', y='7d_avg', label='7-day Avg', color='red', ax=ax)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
