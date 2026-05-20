from __future__ import annotations

import pandas as pd

from .utils import clip_nonnegative


def predict_with_model(model: object, X: pd.DataFrame) -> pd.Series:
    predictions = model.predict(X)
    return pd.Series(clip_nonnegative(predictions), index=X.index)


def batch_predict(models: dict[str, object], X: pd.DataFrame) -> dict[str, pd.Series]:
    return {name: predict_with_model(model, X) for name, model in models.items()}
