from src.zen_pipelines.rule_based.pipeline import rule_based_pipeline

if __name__ == "__main__":
    # Edit these to your local files
    train_days = [
        "/home/huy/Smart_Construction/data/20250815_sensor_data.csv"
    ]  # historical for thresholds
    infer_days = [
        "/home/huy/Smart_Construction/data/20250816_sensor_data.csv"
    ]  # target day

    run = rule_based_pipeline(train_csvs=train_days, infer_csvs=infer_days)
