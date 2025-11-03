from zenml import pipeline
from typing import Dict, Any
from steps import ingest_step

@pipeline
def training_pipeline(data_path: str, ingest_cfg: Dict[str, Any]):
    raw_df = ingest_step(data_path=data_path, cfg=ingest_cfg)
    return raw_df
