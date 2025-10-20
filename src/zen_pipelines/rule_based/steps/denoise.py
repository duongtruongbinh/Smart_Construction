from zenml import step
import pandas as pd

@step
def denoise(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: add light clipping/filtering if necessary
    return df