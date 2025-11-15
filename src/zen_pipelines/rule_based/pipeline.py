from zenml import pipeline
from typing import Dict, Any
from .steps.ingest_data import ingest
from .steps.normalize import normalize_ts
from .steps.denoise import denoise
from .steps.resample import resample_1s
from .steps.feature_window import feature_window
from .steps.rule_classify import rule_classify
from .steps.inference import persist_outputs

@pipeline(name="smart_construction_rule_based")
def rule_based_pipeline(train_csvs: str, infer_csvs: str, ingest_cfg: Dict[str, Any]):
    df_train_raw = ingest(data_path=train_csvs, cfg=ingest_cfg)
    df_train_norm = normalize_ts(df_train_raw)
    df_train_clean = denoise(df_train_norm)
    df_train_1s = resample_1s(df_train_clean)
    df_train_feat = feature_window(df_train_1s)

    df_raw = ingest(data_path=infer_csvs, cfg=ingest_cfg)
    df_norm = normalize_ts(df_raw)
    df_clean = denoise(df_norm)
    df_1s = resample_1s(df_clean)
    df_feat = feature_window(df_1s)

    df_scored = rule_classify(train_hist=df_train_1s, df1s=df_feat)
    _ = persist_outputs(df_scored)