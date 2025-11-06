from zenml import pipeline
from .steps.ingest_data import ingest
from .steps.normalize import normalize_ts
from .steps.denoise import denoise
from .steps.resample import resample_1s
from .steps.feature_window import feature_window
from .steps.rule_classify import rule_classify
from .steps.inference import persist_outputs


@pipeline(name="smart_construction_rule_based")
def rule_based_pipeline(train_csvs: list[str], infer_csvs: list[str]):
    df_train_raw = ingest(train_csvs)
    df_train_norm = normalize_ts(df_train_raw)
    df_train_clean = denoise(df_train_norm)
    df_train_1s = resample_1s(df_train_clean)
    # _df_train_feat = feature_window(   df_train_1s )  # <- đổi tên để biểu thị cố ý không dùng

    df_raw = ingest(infer_csvs)
    df_norm = normalize_ts(df_raw)
    df_clean = denoise(df_norm)
    df_1s = resample_1s(df_clean)
    df_feat = feature_window(df_1s)

    df_scored = rule_classify(train_hist=df_train_1s, df1s=df_feat)
    _ = persist_outputs(df_scored)
