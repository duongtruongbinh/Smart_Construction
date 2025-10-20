from zenml import step
import pandas as pd
from src.core.utils import compute_norms, clean_and_resample

@step
def resample_1s(df: pd.DataFrame, rule: str = "1s") -> pd.DataFrame:
    df = compute_norms(df)
    df1s = clean_and_resample(
        df,
        resample_rule=rule,
        cols_keep=["acc_norm","gyro_norm","Accel_x","Accel_y","Accel_z","Gyro_x","Gyro_y","Gyro_z"],
    )
    return df1s