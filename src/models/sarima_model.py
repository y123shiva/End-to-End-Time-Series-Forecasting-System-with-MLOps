from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(train, order=(1,1,1), seasonal_order=(1,1,1,7), 
                 enforce_stationarity=False, enforce_invertibility=False):
    """
    Train SARIMA model.
    
    Parameters:
    -----------
    train : pd.Series
        Training time series data
    order : tuple
        (p, d, q) parameters
    seasonal_order : tuple
        (P, D, Q, s) seasonal parameters
    enforce_stationarity : bool
        Whether to enforce stationarity
    enforce_invertibility : bool
        Whether to enforce invertibility
        
    Returns:
    --------
    SARIMAX fitted model
    """
    return SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility
    ).fit(disp=False)
