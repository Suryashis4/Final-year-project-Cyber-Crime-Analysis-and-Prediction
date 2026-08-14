"""
Regression model registry for Future Crime Hotspot Prediction.

Follows the same Pipeline + StandardScaler structure as classification models
in react_api.py.
"""

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

REGRESSION_MODEL_NAMES = [
    "Linear Regression",
    "SVR",
    "Decision Tree Regressor",
    "Voting Ensemble Regressor",
]


def _scaled(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def regression_models():
    """Return all regression models for hotspot count forecasting."""
    linear_regression = _scaled(LinearRegression())
    svr = _scaled(SVR(kernel="rbf", C=10.0))
    cart = _scaled(DecisionTreeRegressor(max_depth=10, random_state=42))

    models = {
        "Linear Regression": linear_regression,
        "SVR": svr,
        "Decision Tree Regressor": cart,
    }

    # Separate pipeline instances so voting ensemble does not share fitted state
    models["Voting Ensemble Regressor"] = VotingRegressor(
        estimators=[
            ("lr", _scaled(LinearRegression())),
            ("svr", _scaled(SVR(kernel="rbf", C=10.0))),
            ("dt", _scaled(DecisionTreeRegressor(max_depth=10, random_state=42))),
        ]
    )
    return models
