from zenml import step
import pandas as pd
from pathlib import Path


@step
def persist_outputs(
    df: pd.DataFrame, out_dir: str = "data/outputs", out_name: str = "classified.csv"
) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / out_name)
    df.to_csv(out_path, index=False)
    return out_path
