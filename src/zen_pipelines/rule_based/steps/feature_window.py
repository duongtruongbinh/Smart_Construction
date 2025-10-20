from zenml import step
import pandas as pd

@step
def feature_window(df1s: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    out = df1s.copy()
    for col in ["acc_norm","gyro_norm"]:
        out[f"{col}_mean_{w}s"] = out[col].rolling(window=w, min_periods=1).mean()
        out[f"{col}_std_{w}s"] = out[col].rolling(window=w, min_periods=1).std().fillna(0.0)
    return out