from functools import lru_cache

def lru_forecast_cache(maxsize=128):
    """
    LRU cache decorator for forecast functions.
    """
    return lru_cache(maxsize=maxsize)
