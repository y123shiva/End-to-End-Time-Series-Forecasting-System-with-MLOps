from sklearn.ensemble import IsolationForest

def detect(series, contamination=0.05, random_state=42):
    """
    Detect outliers using Isolation Forest.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    contamination : float
        Expected proportion of outliers (0 to 0.5)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray of booleans indicating outliers
    """
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(series.values.reshape(-1, 1))
    return preds == -1

def clean(series, mask):
    """
    Replace outliers with rolling median.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to clean
    mask : np.ndarray
        Boolean array indicating outliers
        
    Returns:
    --------
    pd.Series with outliers replaced and gaps filled
    """
    med = series.rolling(3, center=True).median()
    s = series.copy()
    s[mask] = med[mask]
    return s.ffill().bfill()
