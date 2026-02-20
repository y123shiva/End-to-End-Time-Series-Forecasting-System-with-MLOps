import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df.set_index("Date").asfreq("D").ffill()
