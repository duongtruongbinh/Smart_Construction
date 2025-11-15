import os
import sys
import click
import yaml
from src.zen_pipelines.rule_based.pipeline import rule_based_pipeline

@click.command()
@click.option("--config-path", default="configs/ingest.yaml", help="Path to ingest.yaml")
@click.option("--data-path", required=True, help="Path to data file CSV")
def main(config_path: str, data_path: str):
    # Edit these to your local files
    train_days = "data/20250815_sensor_data.csv"  # historical for thresholds
    infer_days = "data/20250815_sensor_data.csv"           # target day

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isabs(config_path):
        config_path = os.path.join(root_dir, config_path)
    if not os.path.exists(config_path):
        click.echo(f"❌ Config file not found: {config_path}", err=True)
        sys.exit(1)

    if not os.path.isabs(data_path):
        data_path = os.path.join(root_dir, data_path)
    if not os.path.exists(data_path):
        click.echo(f"❌ Data file not found: {data_path}", err=True)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        ingest_cfg = yaml.safe_load(f) or {}

    rule_based_pipeline(train_csvs=train_days, infer_csvs=infer_days, ingest_cfg=ingest_cfg)

if __name__ == "__main__":
    main()