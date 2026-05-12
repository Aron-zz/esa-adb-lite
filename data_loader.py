"""CSV → numpy 数组，每份数据只读一次，后续算法复用内存。"""
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    "3_months":  None,
    "10_months": None,
    "21_months": None,
    "42_months": None,
    "84_months": None,
}

TARGET_CHANNELS = ["channel_" + str(i) for i in [
    12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,
    50,51,52,57,58,59,60,61,62,63,64,65,66,70,71,72,73,74,75,76
]]
SUBSET_CHANNELS = ["channel_41","channel_42","channel_43","channel_44","channel_45","channel_46"]


class Dataset:
    """单份数据集的完整视图：训练集 + 测试集 + 标签。"""
    __slots__ = ("name", "X_train", "y_train", "X_test", "y_test",
                 "train_channels", "test_channels", "train_timestamps", "test_timestamps")

    def __init__(self, name, X_train, X_test, y_train, y_test, train_channels, test_channels,
                 train_timestamps=None, test_timestamps=None):
        self.name = name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_channels = train_channels
        self.test_channels = test_channels
        self.train_timestamps = train_timestamps
        self.test_timestamps = test_timestamps


def _read_csv(filepath, return_timestamps=False):
    """读取预处理 CSV，返回 (数据矩阵, 标签矩阵, 通道名列表[, 时间戳])。"""
    df = pd.read_csv(filepath)
    data_cols = [c for c in df.columns if c != "timestamp" and not c.startswith("is_anomaly")]
    label_cols = [f"is_anomaly_{c}" for c in data_cols]
    X = df[data_cols].to_numpy(dtype=np.float32)
    y = df[label_cols].to_numpy(dtype=np.uint8) if all(c in df.columns for c in label_cols) else np.zeros_like(X, dtype=np.uint8)
    if return_timestamps:
        timestamps = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return X, y, data_cols, timestamps
    return X, y, data_cols


def load_official_mission1_context(test_csv_path):
    """加载 Mission1 官方 labels/anomaly_types/channels，并裁剪到测试集时间范围。"""
    root = Path(test_csv_path).resolve().parents[3] / "esa-adb" / "mission1" / "ESA-Mission1"
    labels_path = root / "labels.csv"
    anomaly_types_path = root / "anomaly_types.csv"
    channels_path = root / "channels.csv"

    labels_df = pd.read_csv(labels_path, parse_dates=["StartTime", "EndTime"])
    labels_df["StartTime"] = labels_df["StartTime"].dt.tz_localize(None)
    labels_df["EndTime"] = labels_df["EndTime"].dt.tz_localize(None)

    test_data_scores = pd.read_csv(test_csv_path, usecols=["timestamp"], parse_dates=[0])
    test_data_scores.rename(columns={"timestamp": "Timestamp"}, inplace=True)
    test_data_scores["Timestamp"] = test_data_scores["Timestamp"].dt.tz_localize(None)
    test_data_scores["Score"] = np.uint8(0)

    labels_df = labels_df[labels_df["StartTime"] >= test_data_scores["Timestamp"].min()]
    labels_df = labels_df[labels_df["EndTime"] <= test_data_scores["Timestamp"].max()]

    anomaly_types_df = pd.read_csv(anomaly_types_path)
    columns_to_copy = anomaly_types_df.columns[-4:]
    for col in columns_to_copy:
        labels_df[col] = ""
    for _, row in anomaly_types_df.iterrows():
        labels_df.loc[labels_df["ID"] == row["ID"], columns_to_copy] = row[columns_to_copy].values

    channels_df = pd.read_csv(channels_path)
    subsystems_mapping = {s: [*v] for s, v in channels_df.groupby("Subsystem")["Channel"]}

    return labels_df, test_data_scores, subsystems_mapping


def load_all(data_dir: str) -> dict:
    """加载全部 5 个分割的数据集。返回 {name: Dataset}。"""
    base = Path(data_dir) / "multivariate" / "ESA-Mission1-semi-supervised"
    test_df = _read_csv(base / "84_months.test.csv")
    test_X, test_y, test_channels = test_df

    datasets = {}
    for name in DATASETS:
        train_X, train_y, train_channels = _read_csv(base / f"{name}.train.csv")
        datasets[name] = Dataset(name, train_X, train_y, test_X, test_y,
                                 train_channels, test_channels)
    return datasets


def select_channels(dataset: Dataset, target_channels: list[str]) -> Dataset:
    """筛选指定通道。"""
    train_idx = [i for i, c in enumerate(dataset.train_channels) if c in target_channels]
    test_idx  = [i for i, c in enumerate(dataset.test_channels)  if c in target_channels]
    if not train_idx:
        train_idx = list(range(dataset.X_train.shape[1]))
    if not test_idx:
        test_idx  = list(range(dataset.X_test.shape[1]))
    return Dataset(
        name=dataset.name,
        X_train=dataset.X_train[:, train_idx],
        X_test=dataset.X_test[:, test_idx],
        y_train=dataset.y_train[:, train_idx],
        y_test=dataset.y_test[:, test_idx],
        train_channels=[dataset.train_channels[i] for i in train_idx],
        test_channels=[dataset.test_channels[i] for i in test_idx],
    )
