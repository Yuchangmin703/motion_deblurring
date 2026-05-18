import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data/raw/kitti_raw/drives/2011_09_26/2011_09_26_drive_0009_sync/image_02/data"
IMAGE_TS_PATH = PROJECT_ROOT / "data/raw/kitti_raw/drives/2011_09_26/2011_09_26_drive_0009_sync/image_02/timestamps.txt"

OXTS_CSV = PROJECT_ROOT / "data/interim/imu/2011_09_26_drive_0009_oxts_100hz_index_time.csv"
CALIB_CAM_PATH = PROJECT_ROOT / "data/raw/kitti_raw/calib/2011_09_26/calib_cam_to_cam.txt"

OUT_DIR = PROJECT_ROOT / "reports/figures/imu_kernel_test_10ms"
OUT_KERNEL_CSV = PROJECT_ROOT / "data/interim/imu/2011_09_26_drive_0009_kernel_percentiles_10ms.csv"
OUT_META_CSV = PROJECT_ROOT / "data/interim/imu/2011_09_26_drive_0009_all_kernels_10ms_meta.csv"

EXPOSURE_TIME = 0.010
HALF_EXPOSURE = EXPOSURE_TIME / 2.0

ACCEL_WEIGHT = 0.30

# 처음에는 1.0으로 두고 확인.
# 만약 블러가 너무 약하면 5.0, 10.0으로 올리면 됨.
BLUR_GAIN = 10.0

MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 31

PERCENTILES = [50, 70, 80, 90, 95, 99]

# 기준 이미지 번호.
# 일단 100번 정도가 무난함.
# 나중에 원하는 프레임 번호로 바꿔도 됨.
TEST_FRAME_INDEX = 100


def read_image_timestamps(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return pd.to_datetime(lines)


def read_p_rect_02(calib_path):
    with open(calib_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("P_rect_02:"):
            values = line.replace("P_rect_02:", "").strip().split()
            values = [float(v) for v in values]
            P = np.array(values).reshape(3, 4)
            fx = P[0, 0]
            fy = P[1, 1]
            cx = P[0, 2]
            cy = P[1, 2]
            return fx, fy, cx, cy

    raise ValueError("P_rect_02 not found.")


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


def nearest_odd_kernel_size(x, min_size=1, max_size=31):
    k = int(round(x))

    if k < min_size:
        k = min_size

    if k > max_size:
        k = max_size

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
    center = kernel_size // 2

    # 기본 수평선 kernel
    kernel[center, :] = 1.0

    # angle 방향으로 회전
    rot_mat = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))

    s = kernel.sum()
    if s > 1e-12:
        kernel /= s
    else:
        kernel[center, center] = 1.0

    return kernel.astype(np.float32)


def apply_motion_blur(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def make_comparison_grid(image_paths, labels, out_path):
    images = []
    for p, label in zip(image_paths, labels):
        img = cv2.imread(str(p))
        if img is None:
            continue

        # 너무 크면 보기 편하게 축소
        max_w = 620
        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # 라벨 쓰기
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (260, 34), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        images.append(canvas)

    if not images:
        return

    # 세로로 이어붙이기
    grid = np.vstack(images)
    cv2.imwrite(str(out_path), grid)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_KERNEL_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("===== IMU kernel percentile test: 10ms =====")
    print("IMAGE_DIR:", IMAGE_DIR)
    print("OXTS_CSV:", OXTS_CSV)
    print("OUT_DIR:", OUT_DIR)
    print("BLUR_GAIN:", BLUR_GAIN)

    fx, fy, cx, cy = read_p_rect_02(CALIB_CAM_PATH)

    print()
    print("===== camera intrinsics =====")
    print("fx:", fx)
    print("fy:", fy)
    print("cx:", cx)
    print("cy:", cy)

    oxts = pd.read_csv(OXTS_CSV)
    oxts["timestamp_from_index"] = pd.to_datetime(oxts["timestamp_from_index"])

    time_array = oxts["time_sec_index"].values.astype(float)
    gyro_array = oxts[["wx", "wy", "wz"]].values.astype(float)
    accel_array = oxts[["ax", "ay", "az"]].values.astype(float)

    accel_median = np.median(accel_array, axis=0)
    accel_dyn_array = accel_array - accel_median
    accel_dyn_norm_all = np.linalg.norm(accel_dyn_array, axis=1)
    accel_p95 = np.percentile(accel_dyn_norm_all, 95)

    if accel_p95 < 1e-12:
        accel_p95 = 1.0

    image_ts = read_image_timestamps(IMAGE_TS_PATH)
    t0 = oxts["timestamp_from_index"].iloc[0]
    image_time_sec = (image_ts - t0).total_seconds()

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    n = min(len(image_files), len(image_time_sec))

    records = []

    for i in range(n):
        fname = image_files[i]
        t_img = float(image_time_sec[i])

        t_start = t_img - HALF_EXPOSURE
        t_end = t_img + HALF_EXPOSURE

        if t_start < time_array[0] or t_end > time_array[-1]:
            continue

        gyro_start = interpolate_vector(time_array, gyro_array, t_start)
        gyro_end = interpolate_vector(time_array, gyro_array, t_end)

        accel_start = interpolate_vector(time_array, accel_array, t_start)
        accel_end = interpolate_vector(time_array, accel_array, t_end)

        # gyro trapezoidal integration
        delta_theta = 0.5 * (gyro_start + gyro_end) * EXPOSURE_TIME

        delta_roll = delta_theta[0]
        delta_pitch = delta_theta[1]
        delta_yaw = delta_theta[2]

        # 1차 근사
        du = fx * delta_yaw
        dv = fy * delta_pitch

        L_rot = float(np.sqrt(du ** 2 + dv ** 2))

        # acceleration boost
        accel_mean = 0.5 * (accel_start + accel_end)
        accel_dyn = accel_mean - accel_median
        accel_dyn_norm = float(np.linalg.norm(accel_dyn))
        accel_dyn_norm_n = float(np.clip(accel_dyn_norm / accel_p95, 0.0, 1.0))
        accel_boost = 1.0 + ACCEL_WEIGHT * accel_dyn_norm_n

        L_final = BLUR_GAIN * L_rot * accel_boost

        kernel_size = nearest_odd_kernel_size(
            L_final,
            min_size=MIN_KERNEL_SIZE,
            max_size=MAX_KERNEL_SIZE,
        )

        if abs(du) < 1e-12 and abs(dv) < 1e-12:
            angle_deg = 0.0
        else:
            angle_deg = float(np.degrees(np.arctan2(dv, du)))

        records.append({
            "frame": i,
            "filename": fname,
            "image_timestamp": str(image_ts[i]),
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
            "L_final": L_final,
            "kernel_size": kernel_size,
            "angle_deg": angle_deg,
        })

    meta = pd.DataFrame(records)
    meta.to_csv(OUT_META_CSV, index=False)

    print()
    print("[SAVE all meta]", OUT_META_CSV)
    print("num frames:", len(meta))

    print()
    print("===== L_final summary =====")
    print(meta[["L_rot", "accel_boost", "L_final", "kernel_size", "angle_deg"]].describe())

    print()
    print("===== kernel size counts =====")
    print(meta["kernel_size"].value_counts().sort_index())

    # percentile별 대표 kernel 선택
    selected_records = []

    for p in PERCENTILES:
        target = np.percentile(meta["L_final"].values, p)

        # target에 가장 가까운 frame 선택
        idx = (meta["L_final"] - target).abs().idxmin()
        row = meta.loc[idx].copy()
        row["percentile"] = p
        row["target_L_final"] = target
        selected_records.append(row)

    selected = pd.DataFrame(selected_records)
    selected.to_csv(OUT_KERNEL_CSV, index=False)

    print()
    print("[SAVE selected kernels]", OUT_KERNEL_CSV)
    print()
    print("===== selected kernels =====")
    print(selected[[
        "percentile",
        "frame",
        "filename",
        "L_final",
        "kernel_size",
        "angle_deg",
        "du",
        "dv",
        "accel_boost",
    ]])

    # 기준 이미지 한 장 선택
    if TEST_FRAME_INDEX >= len(image_files):
        raise ValueError("TEST_FRAME_INDEX is larger than image count.")

    test_fname = image_files[TEST_FRAME_INDEX]
    test_img_path = IMAGE_DIR / test_fname
    test_img = cv2.imread(str(test_img_path))

    if test_img is None:
        raise FileNotFoundError(f"Failed to read test image: {test_img_path}")

    original_out = OUT_DIR / "original.png"
    cv2.imwrite(str(original_out), test_img)

    output_paths = [original_out]
    labels = [f"Original frame {TEST_FRAME_INDEX}"]

    # 선택된 kernel들을 기준 이미지에 적용
    for _, row in selected.iterrows():
        p = int(row["percentile"])
        k = int(row["kernel_size"])
        angle = float(row["angle_deg"])

        kernel = make_linear_motion_kernel(k, angle)
        blurred = apply_motion_blur(test_img, kernel)

        out_path = OUT_DIR / f"blur_P{p}_k{k}_angle{angle:.1f}.png"
        cv2.imwrite(str(out_path), blurred)

        output_paths.append(out_path)
        labels.append(f"P{p} | k={k} | angle={angle:.1f}")

    grid_path = OUT_DIR / "comparison_grid.png"
    make_comparison_grid(output_paths, labels, grid_path)

    print()
    print("===== saved images =====")
    for p in output_paths:
        print(p)
    print("comparison:", grid_path)


if __name__ == "__main__":
    main()
