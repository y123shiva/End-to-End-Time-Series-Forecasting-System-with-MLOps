import numpy as np
import xgboost as xgb

def create_lags(series, n_lags=7):
    X, y = [], []

    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])

    return np.array(X), np.array(y)


def train_xgb(series, test_size=30):
    X, y = create_lags(series)

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred, y_test
