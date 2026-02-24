import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Project root & data path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
data_file = BASE_DIR / "data" / "financial_data.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(data_file, parse_dates=["Date"], dayfirst=True)
df = df.sort_values("Date")

# -----------------------------
# Dashboard title
# -----------------------------
st.title("📈 Interactive Financial Price Dashboard")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Filters & Options")

# Date range filter
min_date = df["Date"].min()
max_date = df["Date"].max()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date
)

# Rolling average window
rolling_window = st.sidebar.slider("7-day rolling average window", 1, 30, 7)

# Filter dataframe based on date range
df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))].copy()
df_filtered[f"{rolling_window}d_avg"] = df_filtered["Price"].rolling(rolling_window).mean()
df_filtered["Price_diff"] = df_filtered["Price"].diff()

# -----------------------------
# Show filtered raw data
# -----------------------------
st.subheader("Filtered Raw Data")
st.dataframe(df_filtered)

# -----------------------------
# Daily Price Trend
# -----------------------------
st.subheader("Daily Price Trend")
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=df_filtered, x="Date", y="Price", marker="o", ax=ax)
plt.xticks(rotation=45)
plt.title("Daily Price Trend")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# -----------------------------
# Price Trend with Rolling Average
# -----------------------------
st.subheader(f"Price Trend with {rolling_window}-day Rolling Average")
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=df_filtered, x="Date", y="Price", label="Price", marker="o", ax=ax)
sns.lineplot(data=df_filtered, x="Date", y=f"{rolling_window}d_avg", label=f"{rolling_window}-day Avg", color="red", ax=ax)
plt.xticks(rotation=45)
plt.title(f"Price Trend with {rolling_window}-day Rolling Average")
plt.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# -----------------------------
# Distribution of Daily Price Changes
# -----------------------------
st.subheader("Distribution of Daily Price Changes")
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(df_filtered["Price_diff"].dropna(), bins=30, kde=True, ax=ax)
plt.title("Distribution of Daily Price Changes")
plt.xlabel("Daily Price Change")
plt.ylabel("Frequency")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
