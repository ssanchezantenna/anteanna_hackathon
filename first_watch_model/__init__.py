"""First Watch prediction model package.

Predicts the most likely first title a new streaming subscriber will watch,
outputted as a probability distribution over titles.
"""

from first_watch_model.baseline import PopularityBaseline
from first_watch_model.classifier import ServiceClassifier
from first_watch_model.ensemble import EnsemblePredictor
from first_watch_model.features import TitleEncoder, prepare_features
from first_watch_model.predict import predict_from_df

__all__ = [
    "PopularityBaseline",
    "ServiceClassifier",
    "EnsemblePredictor",
    "TitleEncoder",
    "prepare_features",
    "predict_from_df",
]
