# Data Preparation / 数据准备

This repository does not include ESA-ADB data files. The benchmark expects preprocessed ESA-ADB Mission1 CSV files and a small set of raw metadata CSV files.

本仓库不包含 ESA-ADB 数据文件。benchmark 需要预处理后的 ESA-ADB Mission1 CSV，以及少量原始元数据 CSV。

## Data Source / 数据来源

Official ESA-ADB repository:

```text
https://github.com/kplabs-pl/ESA-ADB
```

Official dataset download is described by the ESA-ADB project. Follow their license and citation requirements.

官方数据下载方式以 ESA-ADB 项目说明为准。使用时请遵守其 license 和 citation 要求。

## Expected Layout / 期望目录结构

Place data outside this repository, next to `esa_adb_lite`:

```text
workspace/
  esa_adb_lite/
  data/
    preprocessed/
      multivariate/
        ESA-Mission1-semi-supervised/
          3_months.train.csv
          10_months.train.csv
          21_months.train.csv
          42_months.train.csv
          84_months.train.csv
          84_months.test.csv
          3_months.metadata.json
          10_months.metadata.json
          21_months.metadata.json
          42_months.metadata.json
          84_months.metadata.json
    esa-adb/
      mission1/
        ESA-Mission1/
          labels.csv
          anomaly_types.csv
          channels.csv
```

## What the Preprocessed CSVs Contain / 预处理 CSV 内容

Each preprocessed CSV is a wide time-series table:

- `timestamp`: timestamp column
- 87 value columns: telemetry and telecommand variables such as `channel_41` or `telecommand_244`
- 87 binary label columns: one `is_anomaly_*` column per value column

Values are normalized numerical features. Labels are binary:

```text
0 = normal
1 = anomalous for that channel at that timestamp
```

The benchmark reads each CSV into:

```text
X: [n_timestamps, n_channels]
y: [n_timestamps, n_channels]
```

中文说明：

- 每一行是一个时间戳。
- 数值列是归一化后的遥测/遥控变量。
- 标签列是对应通道的二值异常标签。
- 读取后得到 `X=[时间点数, 通道数]` 和 `y=[时间点数, 通道数]`。

## Option A: Use Shared Preprocessed Data / 方案 A：使用已预处理数据

For our experiments, we prepare a minimal data archive containing:

```text
data/preprocessed/
data/esa-adb/mission1/ESA-Mission1/{labels.csv, anomaly_types.csv, channels.csv}
```

This avoids repeating the official preprocessing step on AutoDL. The archive can be uploaded through cloud storage or another file-transfer service, then extracted to `/root/autodl-tmp/data`.

中文：这样可以避免在 AutoDL 上重复下载和预处理官方数据。可以通过网盘或其他文件传输服务上传压缩包，然后解压到 `/root/autodl-tmp/data`。

After extraction on AutoDL:

```bash
cd /root/esa-adb-lite
python scripts/tools/check_data.py \
  --data-path /root/autodl-tmp/data/preprocessed \
  --mission-root /root/autodl-tmp/data/esa-adb/mission1/ESA-Mission1
```

## Option B: Reproduce from the Official ESA-ADB Repository / 方案 B：从官方仓库复现预处理

Clone the official ESA-ADB repository and follow their preprocessing workflow to generate the `data/preprocessed` layout. This is the most reproducible path, but it can be time-consuming and may require the official TimeEval/Docker environment.

Use this path when you need an end-to-end reproducibility story. Use Option A when you need fast cloud experimentation from already verified preprocessed data.

中文：如果需要完整可复现性，应按官方仓库流程重新生成 `data/preprocessed`。如果目标是快速云端实验，可以使用已经验证过的预处理数据。

## GitHub Policy / GitHub 上传原则

Do not commit data files, generated benchmark results, or generated figures to this repository.

Ignored by default:

```text
results/
logs/
*.zip
__pycache__/
```

Only code, notebooks, and documentation should be pushed to GitHub.
