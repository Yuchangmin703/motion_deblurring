import cv2
import yaml
import shutil
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATE = "2011_09_26"
DRIVE_SYNC = "2011_09_26_drive_0015_sync"
DRIVE_EXTRACT = "2011_09_26_drive_0015_extract"

RAW_IMAGE_DIR = PROJECT_ROOT / f"data/raw/kitti_raw/drives/{DATE}/{DRIVE_SYNC}/image_02/data"
RAW_TS_PATH = PROJECT_ROOT / f"data/raw/kitti_raw/drives/{DATE}/{DRIVE_SYNC}/image_02/timestamps.txt"
OXTS_DIR = PROJECT_ROOT / f"data/raw/kitti_raw/drives/{DATE}/{DRIVE_EXTRACT}/oxts"
CALIB_CAM_PATH = PROJECT_ROOT / f"data/raw/kitti_raw/calib/{DATE}/calib_cam_to_cam.txt"

MAPPING_CSV = PROJECT_ROOT / "data/interim/imu/2011_09_26_drive_0015_detection_raw_mapping.csv"

OUT_ROOT = PROJECT_ROOT / "data/processed/imu_blur/0015_run_0182_0291"
META_OUT = OUT_ROOT / "meta"

EXPOSURE_TIMES_MS = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

RUN_START = 182
RUN_END = 291

AUTO_BLUR_GAIN = True
TARGET_EXPOSURE_MS_FOR_GAIN = 30
TARGET_PERCENTILE_FOR_GAIN = 95
TARGET_P95_KERNEL_AT_MAX_EXPOSURE = 21.0

ACCEL_WEIGHT = 0.30
MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 51

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
        lines = [line.strip() for line in f if line.strip()]
    return pd.to_datetime(lines)


def read_p_rect_02(calib_path):
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P_rect_02:"):
                vals = [float(x) for x in line.replace("P_rect_02:", "").split()]
                P = np.array(vals).reshape(3, 4)
                return P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    raise ValueError("P_rect_02 not found.")


def load_oxts_index_time(oxts_dir):
    data_dir = oxts_dir / "data"
    ts_path = oxts_dir / "timestamps.txt"

    files = sorted(data_dir.glob("*.txt"))
    timestamps = read_timestamps(ts_path)

    records = []
    for i, p in enumerate(files):
        vals = p.read_text().strip().split()
        if len(vals) != len(OXTS_COLUMNS):
            raise ValueError(f"Unexpected OXTS columns in {p}: {len(vals)}")

        row = {c: float(v) for c, v in zip(OXTS_COLUMNS, vals)}
        row["oxts_file"] = p.name
        row["oxts_index"] = int(p.stem)
        row["timestamp_raw"] = timestamps[i] if i < len(timestamps) else pd.NaT
        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values("oxts_index").reset_index(drop=True)

    # 100Hz 균일 시간축
    df["time_sec"] = df["oxts_index"] * 0.01

    t0 = pd.to_datetime(df["timestamp_raw"].iloc[0])
    df["timestamp_from_index"] = t0 + pd.to_timedelta(df["time_sec"], unit="s")

    return df


def interpolate_vector(time_array, value_array, t):
    if t <= time_array[0]:
        return value_array[0].copy()
    if t >= time_array[-1]:
        return value_array[-1].copy()

    idx = np.searchsorted(time_array, t)

    t0 = time_array[idx - 1]
    t1 = time_array[idx]
    v0 = value_array[idx - 1]
    v1 = value_array[idx]

    alpha = (t - t0) / (t1 - t0)
    return (1.0 - alpha) * v0 + alpha * v1


def integrate_window(time_array, value_array, t_start, t_end):
    inside = time_array[(time_array > t_start) & (time_array < t_end)]
    ts = np.concatenate([[t_start], inside, [t_end]])

    vals = np.stack(
        [interpolate_vector(time_array, value_array, t) for t in ts],
        axis=0,
    )

    integ = np.trapz(vals, ts, axis=0)
    mean = integ / max(t_end - t_start, 1e-12)

    return integ, mean


def nearest_odd_kernel_size(x, min_size=1, max_size=51):
    k = int(round(float(x)))
    k = max(min_size, min(max_size, k))

    if k % 2 == 0:
        k += 1

    if k > max_size:
        k -= 2

    if k < min_size:
        k = min_size

    return k


def make_linear_motion_kernel(kernel_size, angle_deg):
    if kernel_size <= 1:
        return np.array([[1.0]], dtype=np.float32)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    c = kernel_size // 2
    kernel[c, :] = 1.0

    M = cv2.getRotationMatrix2D((c, c), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    s = kernel.sum()
    if s > 1e-12:
        kernel /= s
    else:
        kernel[c, c] = 1.0

    return kernel.astype(np.float32)


def write_data_yaml(dataset_dir, max_cls):
    if max_cls <= 2:
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        nc = 3
    else:
        nc = max_cls + 1
        names = {i: f"class_{i}" for i in range(nc)}

    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/val",
        "val": "images/val",
        "nc": nc,
        "names": names,
    }

    with open(dataset_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_max_class(label_paths):
    max_cls = -1
    for p in label_paths:
        if not p.exists():
            continue

        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            cls = int(float(line.split()[0]))
            max_cls = max(max_cls, cls)

    return max_cls


def compute_base_blur_values(
    subset,
    image_time_sec,
    time_array,
    gyro_array,
    accel_array,
    accel_median,
    accel_p95,
    fx,
    fy,
    exp_ms,
):
    exp_s = exp_ms / 1000.0
    half = exp_s / 2.0

    records = []

    for seq_id, row in subset.iterrows():
        raw_frame = int(row["raw_frame"])

        t_img = float(image_time_sec[raw_frame])
        t_start = t_img - half
        t_end = t_img + half

        delta_theta, _ = integrate_window(time_array, gyro_array, t_start, t_end)
        _, accel_mean = integrate_window(time_array, accel_array, t_start, t_end)

        delta_pitch = float(delta_theta[1])
        delta_yaw = float(delta_theta[2])

        du = float(fx * delta_yaw)
        dv = float(fy * delta_pitch)

        L_rot = float(np.sqrt(du ** 2 + dv ** 2))

        accel_dyn = accel_mean - accel_median
        accel_dyn_norm = float(np.linalg.norm(accel_dyn))
        accel_dyn_norm_n = float(np.clip(accel_dyn_norm / accel_p95, 0.0, 1.0))
        accel_boost = float(1.0 + ACCEL_WEIGHT * accel_dyn_norm_n)

        L_base = float(L_rot * accel_boost)

        records.append({
            "seq_id": seq_id,
            "raw_frame": raw_frame,
            "L_base": L_base,
            "L_rot": L_rot,
            "accel_boost": accel_boost,
        })

    return pd.DataFrame(records)


def build_one_condition(
    cond_dir,
    subset,
    image_time_sec,
    time_array,
    gyro_array,
    accel_array,
    accel_median,
    accel_p95,
    fx,
    fy,
    exp_ms,
    blur_gain,
    max_cls,
):
    exp_s = exp_ms / 1000.0
    half = exp_s / 2.0

    img_out_dir = cond_dir / "images/val"
    label_out_dir = cond_dir / "labels/val"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for seq_id, row in subset.iterrows():
        raw_frame = int(row["raw_frame"])

        new_stem = f"{seq_id:06d}"
        new_image_name = f"{new_stem}.png"
        new_label_name = f"{new_stem}.txt"

        raw_img_path = PROJECT_ROOT / row["raw_image"]
        det_label_path = PROJECT_ROOT / row["det_label_path"]

        t_img = float(image_time_sec[raw_frame])
        t_start = t_img - half
        t_end = t_img + half

        delta_theta, gyro_mean = integrate_window(time_array, gyro_array, t_start, t_end)
        _, accel_mean = integrate_window(time_array, accel_array, t_start, t_end)

        delta_roll = float(delta_theta[0])
        delta_pitch = float(delta_theta[1])
        delta_yaw = float(delta_theta[2])

        du = float(fx * delta_yaw)
        dv = float(fy * delta_pitch)

        L_rot = float(np.sqrt(du ** 2 + dv ** 2))

        accel_dyn = accel_mean - accel_median
        accel_dyn_norm = float(np.linalg.norm(accel_dyn))
        accel_dyn_norm_n = float(np.clip(accel_dyn_norm / accel_p95, 0.0, 1.0))
        accel_boost = float(1.0 + ACCEL_WEIGHT * accel_dyn_norm_n)

        L_base = float(L_rot * accel_boost)
        L_final = float(blur_gain * L_base)

        kernel_size = nearest_odd_kernel_size(
            L_final,
            min_size=MIN_KERNEL_SIZE,
            max_size=MAX_KERNEL_SIZE,
        )

        if abs(du) < 1e-12 and abs(dv) < 1e-12:
            angle_deg = 0.0
        else:
            angle_deg = float(np.degrees(np.arctan2(dv, du)))

        img = cv2.imread(str(raw_img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {raw_img_path}")

        kernel = make_linear_motion_kernel(kernel_size, angle_deg)
        blurred = cv2.filter2D(img, -1, kernel)

        cv2.imwrite(str(img_out_dir / new_image_name), blurred)
        shutil.copy2(det_label_path, label_out_dir / new_label_name)

        records.append({
            "seq_id": seq_id,
            "new_image": new_image_name,
            "new_label": new_label_name,
            "det_id": int(row["det_id"]),
            "det_image_original": row["det_image"],
            "det_label_original": row["det_label"],
            "split_original": row["split"],
            "raw_frame": raw_frame,
            "raw_image": row["raw_image"],
            "det_label_path": row["det_label_path"],
            "exposure_ms": exp_ms,
            "image_time_sec": t_img,
            "t_start": t_start,
            "t_end": t_end,
            "delta_roll": delta_roll,
            "delta_pitch": delta_pitch,
            "delta_yaw": delta_yaw,
            "du": du,
            "dv": dv,
            "L_rot": L_rot,
            "accel_dynamic_norm": accel_dyn_norm,
            "accel_dynamic_norm_n": accel_dyn_norm_n,
            "accel_boost": accel_boost,
            "L_base": L_base,
            "blur_gain": blur_gain,
            "L_final": L_final,
            "kernel_size": kernel_size,
            "angle_deg": angle_deg,
        })

    write_data_yaml(cond_dir, max_cls)

    return pd.DataFrame(records)


def build_original_dataset(original_dir, subset, max_cls):
    img_out_dir = original_dir / "images/val"
    label_out_dir = original_dir / "labels/val"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for seq_id, row in subset.iterrows():
        new_stem = f"{seq_id:06d}"
        new_image_name = f"{new_stem}.png"
        new_label_name = f"{new_stem}.txt"

        raw_img_path = PROJECT_ROOT / row["raw_image"]
        det_label_path = PROJECT_ROOT / row["det_label_path"]

        shutil.copy2(raw_img_path, img_out_dir / new_image_name)
        shutil.copy2(det_label_path, label_out_dir / new_label_name)

        records.append({
            "seq_id": seq_id,
            "new_image": new_image_name,
            "new_label": new_label_name,
            "det_id": int(row["det_id"]),
            "det_image_original": row["det_image"],
            "det_label_original": row["det_label"],
            "split_original": row["split"],
            "raw_frame": int(row["raw_frame"]),
            "raw_image": row["raw_image"],
            "det_label_path": row["det_label_path"],
        })

    write_data_yaml(original_dir, max_cls)

    return pd.DataFrame(records)


def main():
    print("===== Build IMU blur datasets: fixed sequential naming + auto gain =====")
    print("OUT_ROOT:", OUT_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    META_OUT.mkdir(parents=True, exist_ok=True)

    fx, fy, cx, cy = read_p_rect_02(CALIB_CAM_PATH)
    print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    oxts = load_oxts_index_time(OXTS_DIR)
    oxts.to_csv(META_OUT / "oxts_100hz_index_time.csv", index=False)

    time_array = oxts["time_sec"].values.astype(float)
    gyro_array = oxts[["wx", "wy", "wz"]].values.astype(float)
    accel_array = oxts[["ax", "ay", "az"]].values.astype(float)

    accel_median = np.median(accel_array, axis=0)
    accel_dyn_all = accel_array - accel_median
    accel_dyn_norm_all = np.linalg.norm(accel_dyn_all, axis=1)
    accel_p95 = np.percentile(accel_dyn_norm_all, 95)
    if accel_p95 < 1e-12:
        accel_p95 = 1.0

    image_ts = read_timestamps(RAW_TS_PATH)
    t0 = pd.to_datetime(oxts["timestamp_from_index"].iloc[0])
    image_time_sec = (image_ts - t0).total_seconds()

    mapping = pd.read_csv(MAPPING_CSV)

    subset = mapping[
        (mapping["raw_frame"] >= RUN_START)
        & (mapping["raw_frame"] <= RUN_END)
        & (mapping["raw_exists"] == True)
        & (mapping["label_exists"] == True)
    ].copy()

    subset = subset.sort_values("raw_frame").reset_index(drop=True)

    print("subset rows:", len(subset))
    print(subset["split"].value_counts())

    subset.to_csv(META_OUT / "selected_run_0182_0291_mapping_original_ids.csv", index=False)

    label_paths = [PROJECT_ROOT / p for p in subset["det_label_path"].tolist()]
    max_cls = get_max_class(label_paths)
    print("max class id:", max_cls)

    # 원본 생성
    original_meta = build_original_dataset(OUT_ROOT / "original", subset, max_cls)
    original_meta.to_csv(META_OUT / "original_meta.csv", index=False)

    # auto blur gain 계산
    gain_ref = compute_base_blur_values(
        subset=subset,
        image_time_sec=image_time_sec,
        time_array=time_array,
        gyro_array=gyro_array,
        accel_array=accel_array,
        accel_median=accel_median,
        accel_p95=accel_p95,
        fx=fx,
        fy=fy,
        exp_ms=TARGET_EXPOSURE_MS_FOR_GAIN,
    )

    p_val = np.percentile(gain_ref["L_base"].values, TARGET_PERCENTILE_FOR_GAIN)

    if p_val < 1e-12:
        blur_gain = 1.0
        print("[WARN] P95 base blur is almost zero. Using blur_gain=1.0")
    else:
        blur_gain = TARGET_P95_KERNEL_AT_MAX_EXPOSURE / p_val

    gain_info = {
        "target_exposure_ms": TARGET_EXPOSURE_MS_FOR_GAIN,
        "target_percentile": TARGET_PERCENTILE_FOR_GAIN,
        "target_kernel_size": TARGET_P95_KERNEL_AT_MAX_EXPOSURE,
        "base_L_percentile": p_val,
        "auto_blur_gain": blur_gain,
    }

    pd.DataFrame([gain_info]).to_csv(META_OUT / "auto_blur_gain_info.csv", index=False)

    print("===== AUTO GAIN =====")
    print(gain_info)

    all_meta = []

    for exp_ms in EXPOSURE_TIMES_MS:
        cond_name = f"exp_{exp_ms:03d}ms"
        cond_dir = OUT_ROOT / cond_name

        meta = build_one_condition(
            cond_dir=cond_dir,
            subset=subset,
            image_time_sec=image_time_sec,
            time_array=time_array,
            gyro_array=gyro_array,
            accel_array=accel_array,
            accel_median=accel_median,
            accel_p95=accel_p95,
            fx=fx,
            fy=fy,
            exp_ms=exp_ms,
            blur_gain=blur_gain,
            max_cls=max_cls,
        )

        meta["condition"] = cond_name
        meta.to_csv(META_OUT / f"{cond_name}_meta.csv", index=False)

        all_meta.append(meta)

        print()
        print("condition:", cond_name)
        print("rows:", len(meta))
        print("kernel size counts:")
        print(meta["kernel_size"].value_counts().sort_index())

    all_meta_df = pd.concat(all_meta, ignore_index=True)
    all_meta_df.to_csv(META_OUT / "all_exposure_blur_meta.csv", index=False)

    print()
    print("===== DONE =====")
    print("datasets:", OUT_ROOT)
    print("meta:", META_OUT)
    print("blur gain:", blur_gain)


if __name__ == "__main__":
    main()
