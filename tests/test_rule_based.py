import pandas as pd
from src.predictors.rule_based import RuleBasedPredictor, RuleParams

def test_predict_shapes():
    df = pd.DataFrame({
        "acc_norm": [0.1, 0.2, 1.0, 2.0, 0.1],
        "gyro_norm": [0.05, 0.1, 0.9, 1.5, 0.05],
    })
    params = RuleParams(acc_med=0.1, acc_mad=0.05, gyro_med=0.05, gyro_mad=0.02,
                        on_th=3.0, off_th=1.5, w_acc=0.7, w_gyro=0.3, debounce_sec=2.0, fs_ds=1.0)
    clf = RuleBasedPredictor(params)
    s = clf.predict(df)
    assert len(s) == len(df)
    assert set(s.unique()).issubset({0, 1})