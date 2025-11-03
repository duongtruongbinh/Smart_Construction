import pandas as pd
from typing import Annotated, Dict, Any
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def ingest_step(
    data_path: str,
    cfg: Dict[str, Any]
) -> Annotated[pd.DataFrame, "raw_df"]:
    read_cfg = cfg.get("read_csv", {})
    validate_cfg = cfg.get("validate", {})

    logger.info("Ingest: reading CSV at %s", data_path)

    kwargs = {
        "sep": read_cfg.get("sep", ","),
        "header": read_cfg.get("header", 0),
        "parse_dates": read_cfg.get("parse_dates")
    }

    df = pd.read_csv(data_path, **{k: v for k, v in kwargs.items() if v is not None})

    required_cols = validate_cfg.get("required_columns", [])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing columns: {missing}")

    logger.info("Ingest: shape=%s columns=%s", df.shape, list(df.columns))
    return df

