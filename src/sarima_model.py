from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(train):
    return SARIMAX(train,
                   order=(1,1,1),
                   seasonal_order=(1,1,1,7),
                   enforce_stationarity=False,
                   enforce_invertibility=False
                  ).fit(disp=False)
