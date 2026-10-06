from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from features import get_pl

def get_models():
    ms = {
        "Dummy": Pipeline([("pl", get_pl()), ("m", DummyRegressor(strategy="mean"))]),
        "Ridge": Pipeline([("pl", get_pl()), ("m", Ridge(random_state=42))]),
        "RF": Pipeline([("pl", get_pl()), ("m", RandomForestRegressor(n_estimators=100, random_state=42))]),
        "HGB": Pipeline([("pl", get_pl()), ("m", HistGradientBoostingRegressor(random_state=42))])
    }
    return ms
