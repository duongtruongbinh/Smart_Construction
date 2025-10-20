import numpy as np
import pandas as pd

def prep_time(df: pd.DataFrame, utc_col: str = "Timestamp", tz: str = "Asia/Ho_Chi_Minh") -> pd.DataFrame:
    df = df.copy()
    df[utc_col] = pd.to_datetime(df[utc_col], errors="coerce", utc=True)
    df = df.sort_values(utc_col).reset_index(drop=True)
    df["Timestamp_local"] = df[utc_col].dt.tz_convert(tz)
    df["ts"] = df["Timestamp_local"].dt.tz_localize(None)  # naive VN time
    return df

def compute_norms(df: pd.DataFrame,
                  accel_cols=("Accel_x","Accel_y","Accel_z"),
                  gyro_cols=("Gyro_x","Gyro_y","Gyro_z")) -> pd.DataFrame:
    acc = df.loc[:, accel_cols].to_numpy()
    gyro = df.loc[:, gyro_cols].to_numpy()
    df = df.copy()
    df["acc_norm"]  = np.linalg.norm(acc, axis=1)
    df["gyro_norm"] = np.linalg.norm(gyro, axis=1)
    return df

def clean_and_resample(df: pd.DataFrame, resample_rule: str = "1s",
                       cols_keep: list[str] | None = None) -> pd.DataFrame:
    if cols_keep is None:
        cols_keep = ["acc_norm", "gyro_norm"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["ts"] + cols_keep).copy()
    out = (df.set_index("ts")[cols_keep]
             .resample(resample_rule).median()
             .dropna().reset_index())
    return out