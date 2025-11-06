from zenml import step
import pandas as pd

# from src.core.utils import prep_time


@step
def normalize_ts(df: pd.DataFrame, tz: str = "Asia/Ho_Chi_Minh") -> pd.DataFrame:
    df = df.copy()
    # 1) Parse UTC tz-aware từ cột Timestamp dạng '...Z'
    ts_utc = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    # 2) Đổi sang giờ VN, 3) bỏ tz để thao tác đơn giản
    df["ts"] = ts_utc.dt.tz_convert(tz).dt.tz_localize(None)
    # Sắp xếp, bỏ hàng lỗi thời gian
    df = df.sort_values("ts").dropna(subset=["ts"]).reset_index(drop=True)
    return df
