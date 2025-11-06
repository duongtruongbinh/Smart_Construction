import json
import numpy as np
import pandas as pd
from pathlib import Path


class ActivityThresholdModel:
    def __init__(
        self,
        accel_cols=("Accel_x", "Accel_y", "Accel_z"),
        gyro_cols=("Gyro_x", "Gyro_y", "Gyro_z"),
        tz="Asia/Ho_Chi_Minh",
        resample_rule="1s",
        eps=1e-6,
    ):
        self.accel_cols = accel_cols
        self.gyro_cols = gyro_cols
        self.tz = tz
        self.resample_rule = resample_rule
        self.eps = eps
        # Tham số sau khi train
        self.acc_med = None
        self.gyro_med = None
        self.acc_mad = None
        self.gyro_mad = None
        self.on_th = None
        self.off_th = None
        self.weights = (0.7, 0.3)  # (w_acc, w_gyro)
        self.debounce_sec = 10.0
        self.fs_ds = 1.0  # sẽ khớp với resample_rule = '1s'

    @staticmethod
    def _prep_time(df, utc_col="Timestamp", tz="Asia/Ho_Chi_Minh"):
        df = df.copy()
        df[utc_col] = pd.to_datetime(df[utc_col], errors="coerce", utc=True)
        df = df.sort_values(utc_col).reset_index(drop=True)
        df["Timestamp_local"] = df[utc_col].dt.tz_convert(tz)
        df["Timestamp_vn"] = df["Timestamp_local"].dt.tz_localize(None)
        return df

    def _compute_norms(self, df):
        acc = df.loc[:, self.accel_cols].to_numpy()
        gyro = df.loc[:, self.gyro_cols].to_numpy()
        df["acc_norm"] = np.linalg.norm(acc, axis=1)
        df["gyro_norm"] = np.linalg.norm(gyro, axis=1)
        return df

    def _clean_and_resample(self, df, cols_keep):
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["Timestamp_vn"] + list(cols_keep)).copy()
        df_rs = (
            df.set_index("Timestamp_vn")[cols_keep]
            .resample(self.resample_rule)
            .median()
            .dropna()
            .reset_index()
        )
        return df_rs

    def _activity_score(self, acc_norm, gyro_norm):
        # robust deviations
        z_acc = np.abs((acc_norm - self.acc_med) / (self.acc_mad + self.eps))
        z_gyro = np.abs((gyro_norm - self.gyro_med) / (self.gyro_mad + self.eps))
        w_acc, w_gyro = self.weights
        return w_acc * z_acc + w_gyro * z_gyro

    @staticmethod
    def _hysteresis_states(scores, on_th, off_th, fs=1.0, debounce_sec=10.0):
        debounce_samples = int(round(debounce_sec * fs))
        states = []
        active = False
        below_count = 0
        for s in scores:
            if not active:
                if s > on_th:
                    active = True
                    below_count = 0
            else:
                if s < off_th:
                    below_count += 1
                    if below_count >= debounce_samples:
                        active = False
                        below_count = 0
                else:
                    below_count = 0
            states.append(1 if active else 0)
        return states

    def fit(
        self, csv_paths, on_pct=75, off_pct=60, weights=(0.7, 0.3), debounce_sec=10.0
    ):
        # csv_paths: str | list[str]
        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]

        frames = []
        for p in csv_paths:
            df = pd.read_csv(p)
            df = self._prep_time(df, utc_col="Timestamp", tz=self.tz)
            df = self._compute_norms(df)
            cols_keep = (
                list(self.accel_cols) + list(self.gyro_cols) + ["acc_norm", "gyro_norm"]
            )
            df_rs = self._clean_and_resample(df, cols_keep=cols_keep)
            frames.append(df_rs[["Timestamp_vn", "acc_norm", "gyro_norm"]])
        hist = pd.concat(frames, ignore_index=True)

        # Ước lượng median & MAD toàn bộ lịch sử train
        self.acc_med = float(np.median(hist["acc_norm"].values))
        self.gyro_med = float(np.median(hist["gyro_norm"].values))
        self.acc_mad = (
            float(np.median(np.abs(hist["acc_norm"].values - self.acc_med))) + self.eps
        )
        self.gyro_mad = (
            float(np.median(np.abs(hist["gyro_norm"].values - self.gyro_med)))
            + self.eps
        )
        self.weights = weights
        self.debounce_sec = debounce_sec

        # Tạo activity_score trên dữ liệu train để chọn ngưỡng percentile
        train_scores = self._activity_score(
            hist["acc_norm"].values, hist["gyro_norm"].values
        )
        self.on_th = float(np.percentile(train_scores, on_pct))
        self.off_th = float(np.percentile(train_scores, off_pct))
        return {
            "acc_med": self.acc_med,
            "acc_mad": self.acc_mad,
            "gyro_med": self.gyro_med,
            "gyro_mad": self.gyro_mad,
            "on_th": self.on_th,
            "off_th": self.off_th,
            "weights": self.weights,
            "debounce_sec": self.debounce_sec,
            "fs_ds": self.fs_ds,
            "resample_rule": self.resample_rule,
            "tz": self.tz,
        }

    def save(self, path):
        params = {
            "acc_med": self.acc_med,
            "acc_mad": self.acc_mad,
            "gyro_med": self.gyro_med,
            "gyro_mad": self.gyro_mad,
            "on_th": self.on_th,
            "off_th": self.off_th,
            "weights": self.weights,
            "debounce_sec": self.debounce_sec,
            "fs_ds": self.fs_ds,
            "resample_rule": self.resample_rule,
            "tz": self.tz,
            "accel_cols": self.accel_cols,
            "gyro_cols": self.gyro_cols,
            "eps": self.eps,
        }
        with open(path, "w") as f:
            json.dump(params, f)

    def load(self, path):
        with open(path, "r") as f:
            params = json.load(f)
        self.acc_med = params["acc_med"]
        self.acc_mad = params["acc_mad"]
        self.gyro_med = params["gyro_med"]
        self.gyro_mad = params["gyro_mad"]
        self.on_th = params["on_th"]
        self.off_th = params["off_th"]
        self.weights = tuple(params["weights"])
        self.debounce_sec = params["debounce_sec"]
        self.fs_ds = params["fs_ds"]
        self.resample_rule = params["resample_rule"]
        self.tz = params["tz"]
        self.accel_cols = tuple(params["accel_cols"])
        self.gyro_cols = tuple(params["gyro_cols"])
        self.eps = params["eps"]
        return params

    def infer(self, csv_in, csv_out=None):
        # Trả về df_ds đã thêm activity_score và active_cont (resample 1s)
        assert all(
            v is not None
            for v in [
                self.acc_med,
                self.acc_mad,
                self.gyro_med,
                self.gyro_mad,
                self.on_th,
                self.off_th,
            ]
        ), "Model chưa fit hoặc load tham số."
        df = pd.read_csv(csv_in)
        df = self._prep_time(df, utc_col="Timestamp", tz=self.tz)
        df = self._compute_norms(df)
        cols_keep = (
            list(self.accel_cols)
            + list(self.gyro_cols)
            + ["acc_norm", "gyro_norm", "Timestamp"]
        )
        df_ds = self._clean_and_resample(df, cols_keep=cols_keep)

        # Tính activity_score theo tham số đã train
        df_ds["activity_score"] = self._activity_score(
            df_ds["acc_norm"].values, df_ds["gyro_norm"].values
        )
        # Hysteresis + debounce
        df_ds["active_cont"] = self._hysteresis_states(
            df_ds["activity_score"].values,
            self.on_th,
            self.off_th,
            fs=self.fs_ds,
            debounce_sec=self.debounce_sec,
        )
        if csv_out:
            df_ds.to_csv(csv_out, index=False)
        return df_ds


class ActivityVisualizer:
    def __init__(
        self, params_json=None, window_sec=300, html_out="excavator_activity_view.html"
    ):
        """
        params_json: đường dẫn JSON tham số đã train (để đọc on_th/off_th). Nếu None, sẽ không vẽ Span.
        window_sec: chiều rộng cửa sổ hiển thị khi Play (giây), mặc định 5 phút.
        html_out: tên file HTML xuất.
        """
        self.params = None
        self.on_th = None
        self.off_th = None
        self.window_sec = int(window_sec)
        self.html_out = html_out
        self.fs_ds = 1.0  # giả định CSV đã resample 1s khi inference
        if params_json:
            with open(params_json, "r") as f:
                self.params = json.load(f)
            # Đọc ngưỡng đã train (nếu có)
            self.on_th = self.params.get("on_th", None)
            self.off_th = self.params.get("off_th", None)
            self.fs_ds = float(self.params.get("fs_ds", 1.0))

    def _load_and_clean(self, csv_classified):
        # Đọc và làm sạch tối thiểu
        df = pd.read_csv(csv_classified, parse_dates=["Timestamp_vn"])
        # Xác định các cột có thể có
        base_cols = ["Timestamp_vn", "activity_score", "active_cont"]
        optional_cols = [
            "Accel_x",
            "Accel_y",
            "Accel_z",
            "Gyro_x",
            "Gyro_y",
            "Gyro_z",
            "acc_norm",
            "gyro_norm",
        ]
        present_optional = [c for c in optional_cols if c in df.columns]
        use_cols = base_cols + present_optional

        # Thay Inf -> NaN rồi drop
        df[use_cols] = df[use_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(
            subset=["Timestamp_vn", "activity_score", "active_cont"]
        ).reset_index(drop=True)

        # Đảm bảo kiểu
        df["active_cont"] = df["active_cont"].astype(int)

        return df, use_cols

    def show(self, csv_classified):
        df, use_cols = self._load_and_clean(csv_classified)
        source = ColumnDataSource(df)

        # Controls
        play_toggle = Toggle(label="Play", button_type="success", active=False)
        speed_slider = Slider(
            title="Tốc độ phát (x)", start=0.25, end=4.0, step=0.25, value=1.0
        )
        index_slider = Slider(
            title="Vị trí (mốc thời gian)", start=0, end=len(df) - 1, step=1, value=0
        )
        status_text = Div(text="Paused")

        # Plot 1: Activity score + Span ngưỡng (nếu có)
        p1 = figure(
            x_axis_type="datetime",
            width=1100,
            height=260,
            title="Activity Score (giờ VN)",
        )
        p1.line(
            "Timestamp_vn",
            "activity_score",
            source=source,
            line_width=1.5,
            color="navy",
        )
        if self.on_th is not None:
            p1.add_layout(
                Span(
                    location=self.on_th,
                    dimension="width",
                    line_color="green",
                    line_dash="dashed",
                    line_width=1,
                )
            )
        if self.off_th is not None:
            p1.add_layout(
                Span(
                    location=self.off_th,
                    dimension="width",
                    line_color="red",
                    line_dash="dashed",
                    line_width=1,
                )
            )
        p1.output_backend = "webgl"

        # Plot 2: Trạng thái 0/1
        p2 = figure(
            x_axis_type="datetime",
            width=1100,
            height=120,
            title="Trạng thái (Active: 1, Idle: 0)",
            x_range=p1.x_range,
            y_range=(-0.2, 1.2),
        )
        p2.step(
            "Timestamp_vn",
            "active_cont",
            source=source,
            line_width=2,
            color="green",
            mode="after",
        )
        p2.output_backend = "webgl"

        # Plot 3: Accel các trục (nếu có)
        p3 = None
        if all(c in use_cols for c in ["Accel_x", "Accel_y", "Accel_z"]):
            p3 = figure(
                x_axis_type="datetime",
                width=1100,
                height=220,
                title="Accel theo trục (m/s²)",
                x_range=p1.x_range,
            )
            p3.line(
                "Timestamp_vn",
                "Accel_x",
                source=source,
                color="#1f77b4",
                alpha=0.9,
                legend_label="Accel_x",
                line_width=1,
            )
            p3.line(
                "Timestamp_vn",
                "Accel_y",
                source=source,
                color="#ff7f0e",
                alpha=0.9,
                legend_label="Accel_y",
                line_width=1,
            )
            p3.line(
                "Timestamp_vn",
                "Accel_z",
                source=source,
                color="#2ca02c",
                alpha=0.9,
                legend_label="Accel_z",
                line_width=1,
            )
            p3.legend.location = "top_left"
            p3.output_backend = "webgl"

        # Plot 4: Gyro các trục (nếu có)
        p4 = None
        if all(c in use_cols for c in ["Gyro_x", "Gyro_y", "Gyro_z"]):
            p4 = figure(
                x_axis_type="datetime",
                width=1100,
                height=220,
                title="Gyro theo trục (°/s)",
                x_range=p1.x_range,
            )
            p4.line(
                "Timestamp_vn",
                "Gyro_x",
                source=source,
                color="#d62728",
                alpha=0.9,
                legend_label="Gyro_x",
                line_width=1,
            )
            p4.line(
                "Timestamp_vn",
                "Gyro_y",
                source=source,
                color="#9467bd",
                alpha=0.9,
                legend_label="Gyro_y",
                line_width=1,
            )
            p4.line(
                "Timestamp_vn",
                "Gyro_z",
                source=source,
                color="#8c564b",
                alpha=0.9,
                legend_label="Gyro_z",
                line_width=1,
            )
            p4.legend.location = "top_left"
            p4.output_backend = "webgl"

        # Plot 5: Norm Accel/Gyro (nếu có)
        p5 = None
        has_acc_norm = "acc_norm" in use_cols
        has_gyro_norm = "gyro_norm" in use_cols
        if has_acc_norm or has_gyro_norm:
            p5 = figure(
                x_axis_type="datetime",
                width=1100,
                height=200,
                title="Norm Accel / Gyro",
                x_range=p1.x_range,
            )
            if has_acc_norm:
                p5.line(
                    "Timestamp_vn",
                    "acc_norm",
                    source=source,
                    color="black",
                    alpha=0.8,
                    legend_label="acc_norm",
                    line_width=1,
                )
            if has_gyro_norm:
                p5.line(
                    "Timestamp_vn",
                    "gyro_norm",
                    source=source,
                    color="purple",
                    alpha=0.8,
                    legend_label="gyro_norm",
                    line_width=1,
                )
            p5.legend.location = "top_left"
            p5.output_backend = "webgl"

        # Cửa sổ xem
        window_samples = int(self.window_sec * self.fs_ds)

        # Thiết lập x_range ban đầu
        start_ts = df["Timestamp_vn"].iloc[0]
        end_idx = min(len(df) - 1, window_samples)
        end_ts = df["Timestamp_vn"].iloc[end_idx]
        for p in [p for p in (p1, p2, p3, p4, p5) if p is not None]:
            p.x_range.start = start_ts
            p.x_range.end = end_ts

        # Callback Play (JS) — giữ nguyên hành vi “replay”
        callback = CustomJS(
            args=dict(
                source=source,
                xr1=p1.x_range,
                xr2=p2.x_range,
                idx_slider=index_slider,
                spd_slider=speed_slider,
                play=play_toggle,
                status=status_text,
                window_samples=window_samples,
            ),
            code="""
            if (!play.active) { status.text = "Paused"; return; }
            status.text = "Playing x" + spd_slider.value.toFixed(2);

            var step = Math.max(1, Math.floor(spd_slider.value));
            var i = idx_slider.value + step;
            var n = source.data['Timestamp_vn'].length - 1;

            if (i >= n) { i = n; play.active = false; status.text = "Reached end"; }
            idx_slider.value = i;

            var ts  = source.data['Timestamp_vn'][i];
            var j0  = Math.max(0, i - window_samples);
            var ts0 = source.data['Timestamp_vn'][j0];

            xr1.start = ts0; xr1.end = ts;
            xr2.start = ts0; xr2.end = ts;
        """,
        )

        ticker = CustomJS(
            args=dict(cb=callback, play=play_toggle),
            code="""
            if (!window.__bokeh_player_interval__) {
                window.__bokeh_player_interval__ = setInterval(function(){ cb.execute(); }, 200);
            }
        """,
        )

        # Kéo slider bằng tay thì cập nhật cửa sổ
        index_slider.js_on_change(
            "value",
            CustomJS(
                args=dict(
                    source=source,
                    xr1=p1.x_range,
                    xr2=p2.x_range,
                    window_samples=window_samples,
                ),
                code="""
            var i = cb_obj.value;
            var j0 = Math.max(0, i - window_samples);
            var ts  = source.data['Timestamp_vn'][i];
            var ts0 = source.data['Timestamp_vn'][j0];
            xr1.start = ts0; xr1.end = ts;
            xr2.start = ts0; xr2.end = ts;
        """,
            ),
        )

        play_toggle.js_on_click(ticker)
        speed_slider.js_on_change("value", callback)

        controls = row(play_toggle, speed_slider, index_slider, status_text)

        # Layout linh hoạt theo cột có trong CSV
        figs = [p for p in (p1, p2, p3, p4, p5) if p is not None]
        layout = column(controls, *figs)

        output_file(self.html_out)
        show(layout)


if __name__ == "__main__":
    # Train
    model = ActivityThresholdModel()
    params = model.fit(
        csv_paths=["20250803_accelerometer_data.csv"],  # có thể là danh sách nhiều ngày
        on_pct=75,
        off_pct=60,
        weights=(0.7, 0.3),
        debounce_sec=10.0,
    )
    model.save("activity_params.json")
    print("Đã train:", params)

    # Inference trên ngày khác
    model2 = ActivityThresholdModel()
    model2.load("activity_params.json")
    df_cls = model2.infer(
        "20250808_accelerometer_data.csv", csv_out="20250808_classified.csv"
    )
    print("Đã ghi file phân loại:", "20250808_classified.csv")

    viz = ActivityVisualizer(
        params_json="activity_params.json",  # để vẽ on_th/off_th lên chart
        window_sec=300,  # cửa sổ 5 phút
        html_out="20250808_activity_view.html",  # file HTML đầu ra
    )
    viz.show("20250808_classified.csv")
