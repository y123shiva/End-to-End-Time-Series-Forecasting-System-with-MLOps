import pandas as pd
from prophet import Prophet

def train_prophet(df, date_col, target_col, test_size=30):
    df = df[[date_col, target_col]].rename(
        columns={date_col: "ds", target_col: "y"}
    )

    train = df[:-test_size]
    test = df[-test_size:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=test_size)
    forecast = model.predict(future)

    y_pred = forecast["yhat"].tail(test_size).values
    y_test = test["y"].values

    return model, y_pred, y_test
