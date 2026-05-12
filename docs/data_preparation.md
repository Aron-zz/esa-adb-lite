# Data Preparation

This repository does not include ESA-ADB data files. The benchmark expects preprocessed ESA-ADB Mission1 CSV files and a small set of raw metadata CSV files.

## Data Source

Official ESA-ADB repository:

```text
https://github.com/kplabs-pl/ESA-ADB
```

Official dataset download is described by the ESA-ADB project. Follow their license and citation requirements.

## Expected Layout

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

## What the Preprocessed CSVs Contain

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

## Option A: Use Shared Preprocessed Data

For our experiments, we prepare a minimal data archive containing:

```text
data/preprocessed/
data/esa-adb/mission1/ESA-Mission1/{labels.csv, anomaly_types.csv, channels.csv}
```

This avoids repeating the official preprocessing step on AutoDL. The archive can be uploaded through cloud storage or another file-transfer service, then extracted to `/root/autodl-tmp/data`.

After extraction on AutoDL:

```bash
cd /root/esa-adb-lite
python scripts/tools/check_data.py \
  --data-path /root/autodl-tmp/data/preprocessed \
  --mission-root /root/autodl-tmp/data/esa-adb/mission1/ESA-Mission1
```

## Option B: Reproduce from the Official ESA-ADB Repository

Clone the official ESA-ADB repository and follow their preprocessing workflow to generate the `data/preprocessed` layout. This is the most reproducible path, but it can be time-consuming and may require the official TimeEval/Docker environment.

Use this path when you need an end-to-end reproducibility story. Use Option A when you need fast cloud experimentation from already verified preprocessed data.

## GitHub Policy

Do not commit data files, generated benchmark results, or generated figures to this repository.

Ignored by default:

```text
results/
logs/
*.zip
__pycache__/
```

Only code, notebooks, and documentation should be pushed to GitHub.
