from sklearn.ensemble import IsolationForest

def detect(series):
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(series.values.reshape(-1,1))
    return preds == -1

def clean(series, mask):
    med = series.rolling(3, center=True).median()
    s = series.copy()
    s[mask] = med[mask]
    return s.ffill().bfill()
