import pandas as pd


def create_lags(series, lags=7):
    df = pd.DataFrame({"y": series})

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)

    return df.dropna()


def rolling_features(series, windows=(7, 14, 30)):
    df = pd.DataFrame({"y": series})

    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].rolling(w).std()

    return df


def calendar_features(index):
    df = pd.DataFrame(index=index)

    df["day"] = index.day
    df["month"] = index.month
    df["weekday"] = index.weekday
    df["weekofyear"] = index.isocalendar().week.astype(int)

    return df


def build_features(series):
    base = create_lags(series)
    roll = rolling_features(series)
    cal = calendar_features(series.index)

    df = base.join(roll, how="left").join(cal, how="left")

    return df.dropna()
