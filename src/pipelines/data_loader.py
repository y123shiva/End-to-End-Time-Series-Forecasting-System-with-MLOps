import pandas as pd


def load_data(path: str, freq: str = "D") -> pd.DataFrame:
    """
    Load time-series CSV safely for forecasting pipelines.

    Steps:
    - parse dates
    - sort chronologically
    - drop duplicates
    - enforce fixed frequency
    - forward fill gaps

    Parameters
    ----------
    path : str
        CSV path
    freq : str
        frequency (D, W, M, etc.)

    Returns
    -------
    pd.DataFrame
        clean indexed time-series dataframe
    """

    df = pd.read_csv(path)

    # ---- validations ----
    if "Date" not in df.columns:
        raise ValueError("CSV must contain 'Date' column")

    # ---- parse date ----
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["Date"])

    # ---- sort + dedupe ----
    df = (
        df.sort_values("Date")
          .drop_duplicates("Date")
          .set_index("Date")
    )

    # ---- enforce regular frequency ----
    df = df.asfreq(freq)

    # ---- fill gaps ----
    df = df.ffill()

    return df
