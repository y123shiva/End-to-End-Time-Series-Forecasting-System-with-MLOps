import numpy as np
from src.pipelines.evaluate import rmse


def rolling_cv(series, train_fn, horizon=14, step=14, min_train=90):
    """
    Walk-forward cross validation

    train_fn(train_series, test_size) -> (model, preds, y_true)
    """

    scores = []

    for end in range(min_train, len(series) - horizon, step):
        train = series[:end]
        test_size = horizon

        model, preds, y_true = train_fn(train, test_size)

        score = rmse(y_true, preds)
        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores))
