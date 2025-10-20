from src.zen_pipelines.rule_based.pipeline import rule_based_pipeline

if __name__ == "__main__":
    # Edit these to your local files
    train_days = ["data/01_raw/20250804_accelerometer_data.csv"]  # historical for thresholds
    infer_days = ["data/01_raw/20250808_accelerometer_data.csv"]           # target day

    run = rule_based_pipeline(train_csvs=train_days, infer_csvs=infer_days)