import os
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OXTS_DIR = os.path.join(
    PROJECT_ROOT,
    "data/raw/kitti_raw/drives/2011_09_26/2011_09_26_drive_0009_extract/oxts"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "data/interim/imu/2011_09_26_drive_0009_oxts_100hz.csv"
)


OXTS_COLUMNS = [
    "lat", "lon", "alt",
    "roll", "pitch", "yaw",
    "vn", "ve", "vf", "vl", "vu",
    "ax", "ay", "az", "af", "al", "au",
    "wx", "wy", "wz", "wf", "wl", "wu",
    "posacc", "velacc",
    "navstat", "numsats",
    "posmode", "velmode", "orimode",
]


def read_timestamps(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return pd.to_datetime(lines)


def main():
    data_dir = os.path.join(OXTS_DIR, "data")
    timestamp_path = os.path.join(OXTS_DIR, "timestamps.txt")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"OXTS data folder not found: {data_dir}")

    if not os.path.exists(timestamp_path):
        raise FileNotFoundError(f"OXTS timestamps not found: {timestamp_path}")

    timestamps = read_timestamps(timestamp_path)

    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".txt")
    ])

    print("===== OXTS 100Hz Parsing =====")
    print(f"data_dir       : {data_dir}")
    print(f"timestamp_path : {timestamp_path}")
    print(f"num txt files  : {len(files)}")
    print(f"num timestamps : {len(timestamps)}")

    if len(files) != len(timestamps):
        print("[WARN] txt file count and timestamp count are different.")

    records = []
    n = min(len(files), len(timestamps))

    t0 = timestamps[0]

    for i in range(n):
        file_path = os.path.join(data_dir, files[i])

        with open(file_path, "r") as f:
            values = f.read().strip().split()

        if len(values) != len(OXTS_COLUMNS):
            raise ValueError(
                f"Unexpected OXTS column count in {files[i]}: "
                f"{len(values)} values, expected {len(OXTS_COLUMNS)}"
            )

        row = {col: float(val) for col, val in zip(OXTS_COLUMNS, values)}

        ts = timestamps[i]
        row["timestamp"] = ts
        row["time_sec"] = (ts - t0).total_seconds()
        row["oxts_file"] = files[i]

        records.append(row)

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"[SAVE] {OUTPUT_CSV}")

    print()
    print("===== head =====")
    print(df[["timestamp", "time_sec", "ax", "ay", "az", "wx", "wy", "wz"]].head())

    print()
    print("===== dt summary =====")
    dt = df["time_sec"].diff().dropna()
    print(dt.describe())

    print()
    print("===== estimated frequency =====")
    print(f"median dt: {dt.median():.6f} sec")
    print(f"estimated Hz: {1.0 / dt.median():.2f} Hz")


if __name__ == "__main__":
    main()
