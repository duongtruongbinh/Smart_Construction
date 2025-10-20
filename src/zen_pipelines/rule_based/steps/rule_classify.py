from zenml import step
import pandas as pd
from src.predictors.rule_based import RuleBasedPredictor

@step(enable_cache=False)
def rule_classify(
    train_hist: pd.DataFrame,
    df1s: pd.DataFrame,
    on_pct: float = 75,
    off_pct: float = 60,
    w_acc: float = 0.7,
    w_gyro: float = 0.3,
    debounce_sec: float = 10.0,
) -> pd.DataFrame:
    params = RuleBasedPredictor.fit_thresholds(
        train_hist, on_pct=on_pct, off_pct=off_pct,
        w_acc=w_acc, w_gyro=w_gyro, debounce_sec=debounce_sec, fs_ds=1.0
    )
    clf = RuleBasedPredictor(params)
    score = clf.predict_score(df1s)
    state = clf.predict(df1s)
    out = df1s.copy()
    out["activity_score"] = score
    out["active_cont"] = state
    return out