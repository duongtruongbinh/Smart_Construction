# Smart Construction

This pack contains a minimal ZenML pipeline to run data ingestion from accelerometer data.

## Quickstart

### 1. Tạo môi trường ảo với Python 3.12

```bash
python3.12 -m venv .venv
# hoặc: uv venv --python 3.12
```

### 2. Kích hoạt môi trường ảo

```bash
source .venv/bin/activate  # macOS/Linux
# Windows: .venv\Scripts\activate
```

### 3. Cài đặt dependencies bằng uv

```bash
uv pip install -r requirements.txt
```

### 4. Khởi tạo ZenML

```bash
zenml init
```

### 5. Mở ZenML dashboard (optional)

```bash
zenml login --local
```

Dashboard sẽ mở tại http://localhost:8237

### 6. Chạy pipeline

```bash
PYTHONPATH=. python apps/run_training_pipeline.py \
       --data-path data/raw/20250803_accelerometer_data.csv \
       --config-path configs/ingest.yaml
```

### 7. Kiểm tra kết quả trên dashboard

---

## 📁 Project Structure

```bash
smart-construction/
    ├── data/
    │   └── raw/
    │       └── 20250803_accelerometer_data.csv
    ├── apps/
    │   └── run_training_pipeline.py
    ├── configs/
    │   └── ingest.yaml                 # Config duy nhất dùng cho tuần này
    ├── steps/
    │   └── ingest.py                   # Bước ingest chi tiết
    ├── pipelines/
    │   └── training_pipeline.py        # Gọi ingest duy nhất
    ├── README.md
    ├── requirements.txt
    └── .gitignore
```

---

## Configs

- Chỉ dùng `configs/ingest.yaml` cho tuần này. Không hardcode trong code.

### configs/ingest.yaml

- `required_columns`: Các cột bắt buộc trong CSV
- `read_csv`: Tham số cho pd.read_csv
- `validate`: Cấu hình validation

---

## Layout

- `steps/ingest.py`: Ingest step với validation
- `pipelines/training_pipeline.py`: Orchestrated pipeline
- `apps/run_training_pipeline.py`: CLI runner với click
- `configs/ingest.yaml`: Configuration file
