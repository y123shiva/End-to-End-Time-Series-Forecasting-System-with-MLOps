import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("../data/financial_data.csv", parse_dates=['Date'], dayfirst=True)
df = df.sort_values('Date')

# 1️⃣ Daily Price Trend
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Price', marker='o')
plt.title("Daily Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../images/daily_price_trend.png")
plt.close()

# 2️⃣ Price Trend with 7-day Rolling Average
df['7d_avg'] = df['Price'].rolling(7).mean()
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Price', label='Price', marker='o')
sns.lineplot(data=df, x='Date', y='7d_avg', label='7-day Rolling Avg', color='red')
plt.title("Price Trend with 7-day Rolling Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("../images/price_rolling_avg.png")
plt.close()

# 3️⃣ Distribution of Daily Price Changes
df['Price_diff'] = df['Price'].diff()
plt.figure(figsize=(8,5))
sns.histplot(df['Price_diff'].dropna(), bins=30, kde=True)
plt.title("Distribution of Daily Price Changes")
plt.xlabel("Daily Price Change")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../images/price_change_dist.png")
plt.close()

print("All images generated successfully in images/ folder!")
