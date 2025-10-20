from zenml import step
import pandas as pd

@step
def ingest(csv_paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)