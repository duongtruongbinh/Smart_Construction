from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd

@dataclass
class RuleParams:
    acc_med: float
    acc_mad: float
    gyro_med: float
    gyro_mad: float
    on_th: float
    off_th: float
    w_acc: float = 0.7
    w_gyro: float = 0.3
    debounce_sec: float = 10.0
    fs_ds: float = 1.0

class RuleBasedPredictor:
    """Robust z-score + hysteresis + debounce rule-based classifier."""
    def __init__(self, params: RuleParams | None = None):
        self.params = params

    @staticmethod
    def fit_thresholds(hist: pd.DataFrame,
                       on_pct: float = 75,
                       off_pct: float = 60,
                       w_acc: float = 0.7,
                       w_gyro: float = 0.3,
                       debounce_sec: float = 10.0,
                       fs_ds: float = 1.0) -> RuleParams:
        acc_med  = float(np.median(hist["acc_norm"].values))
        gyro_med = float(np.median(hist["gyro_norm"].values))
        acc_mad  = float(np.median(np.abs(hist["acc_norm"].values - acc_med))) + 1e-6
        gyro_mad = float(np.median(np.abs(hist["gyro_norm"].values - gyro_med))) + 1e-6
        z_acc  = np.abs((hist["acc_norm"].values  - acc_med)  / acc_mad)
        z_gyro = np.abs((hist["gyro_norm"].values - gyro_med) / gyro_mad)
        score  = w_acc * z_acc + w_gyro * z_gyro
        on_th  = float(np.percentile(score, on_pct))
        off_th = float(np.percentile(score, off_pct))
        return RuleParams(acc_med, acc_mad, gyro_med, gyro_mad, on_th, off_th,
                          w_acc, w_gyro, debounce_sec, fs_ds)

    def predict_score(self, df1s: pd.DataFrame) -> pd.Series:
        p = self.params
        z_acc  = np.abs((df1s["acc_norm"].values  - p.acc_med)  / p.acc_mad)
        z_gyro = np.abs((df1s["gyro_norm"].values - p.gyro_med) / p.gyro_mad)
        return pd.Series(p.w_acc * z_acc + p.w_gyro * z_gyro, index=df1s.index, name="activity_score")

    @staticmethod
    def _hysteresis(scores: np.ndarray, on_th: float, off_th: float, fs: float, debounce_sec: float) -> list[int]:
        debounce = int(round(debounce_sec * fs))
        states, active, below = [], False, 0
        for s in scores:
            if not active and s > on_th:
                active, below = True, 0
            elif active and s < off_th:
                below += 1
                if below >= debounce:
                    active, below = False, 0
            else:
                below = 0 if active else below
            states.append(1 if active else 0)
        return states

    def predict(self, df1s: pd.DataFrame) -> pd.Series:
        score = self.predict_score(df1s)
        p = self.params
        states = self._hysteresis(score.values, p.on_th, p.off_th, p.fs_ds, p.debounce_sec)
        return pd.Series(states, index=df1s.index, name="active_cont")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.params.__dict__, f)

    @staticmethod
    def load(path: str) -> "RuleBasedPredictor":
        with open(path, "r") as f:
            d = json.load(f)
        return RuleBasedPredictor(RuleParams(**d))