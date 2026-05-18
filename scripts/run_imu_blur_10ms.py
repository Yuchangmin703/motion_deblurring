import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data/raw/kitti_raw/drives/2011_09_26/2011_09_26_drive_0009_sync/image_02/data"
IMAGE_TS_PATH = PROJECT_ROOT / "data/raw/kitti_raw/drives/2011_09_26/2011_09_26_drive_0009_sync/image_02/timestamps.txt"

OXTS_CSV = PROJECT_ROOT / "data/interim/imu/2011_09_26_drive_0009_oxts_100hz_index_time.csv"

CALIB_CAM_PATH = PROJECT_ROOT / "data/raw/kitti_raw/calib/2011_09_26/calib_cam_to_cam.txt"

OUTPUT_IMAGE_DIR = PROJECT_ROOT / "data/interim/blurred/2011_09_26_drive_0009_imu_10ms/image_02/data"
OUTPUT_META_CSV = PROJECT_ROOT / "data/interim/imu/2011_09_26_drive_0009_imu_blur_10ms_meta.csv"

EXPOSURE_TIME = 0.010
HALF_EXPOSURE = EXPOSURE_TIME / 2.0

ACCEL_WEIGHT = 0.30
BLUR_GAIN = 1.0

MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 21


def read_image_timestamps(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return pd.to_datetime(lines)


def read_p_rect_02(calib_path):
    """
    KITTI calib_cam_to_cam.txt에서 P_rect_02를 읽고 fx, fy를 반환.
    """
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

    raise ValueError("P_rect_02 not found in calibration file.")


def interpolate_vector(time_array, value_array, t):
    """
    time_array: 증가하는 시간축
    value_array: shape (N, D)
    t: 보간할 시간

    return: shape (D,)
    """
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


def nearest_odd_kernel_size(x, min_size=1, max_size=21):
    """
    pixel shift 길이를 kernel size로 변환.
    OpenCV motion blur kernel은 홀수 크기가 안정적임.
    """
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
    """
    angle_deg 방향의 linear motion blur kernel 생성.
    angle_deg = 0이면 수평 방향 blur.
    """
    if kernel_size <= 1:
        return np.array([[1.0]], dtype=np.float32)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    kernel[center, :] = 1.0

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


def main():
    print("===== IMU-informed blur generation: 10ms =====")

    print("IMAGE_DIR      :", IMAGE_DIR)
    print("IMAGE_TS_PATH  :", IMAGE_TS_PATH)
    print("OXTS_CSV       :", OXTS_CSV)
    print("CALIB_CAM_PATH :", CALIB_CAM_PATH)
    print("OUTPUT_DIR     :", OUTPUT_IMAGE_DIR)

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_META_CSV.parent.mkdir(parents=True, exist_ok=True)

    # 1. calibration 읽기
    fx, fy, cx, cy = read_p_rect_02(CALIB_CAM_PATH)
    print()
    print("===== camera intrinsics from P_rect_02 =====")
    print(f"fx: {fx}")
    print(f"fy: {fy}")
    print(f"cx: {cx}")
    print(f"cy: {cy}")

    # 2. OXTS 읽기
    oxts = pd.read_csv(OXTS_CSV)
    oxts["timestamp_from_index"] = pd.to_datetime(oxts["timestamp_from_index"])

    time_array = oxts["time_sec_index"].values.astype(float)

    gyro_array = oxts[["wx", "wy", "wz"]].values.astype(float)
    accel_array = oxts[["ax", "ay", "az"]].values.astype(float)

    # dynamic acceleration 계산용 median
    accel_median = np.median(accel_array, axis=0)
    accel_dyn_array = accel_array - accel_median
    accel_dyn_norm_all = np.linalg.norm(accel_dyn_array, axis=1)

    # acceleration boost normalization 기준
    accel_p95 = np.percentile(accel_dyn_norm_all, 95)
    if accel_p95 < 1e-12:
        accel_p95 = 1.0

    # 3. image timestamp 읽기
    image_ts = read_image_timestamps(IMAGE_TS_PATH)

    t0 = oxts["timestamp_from_index"].iloc[0]
    image_time_sec = (image_ts - t0).total_seconds()

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(image_files) != len(image_time_sec):
        print("[WARN] image file count and timestamp count differ.")
        print("num images:", len(image_files))
        print("num timestamps:", len(image_time_sec))

    n = min(len(image_files), len(image_time_sec))

    records = []

    for i in range(n):
        fname = image_files[i]
        t_img = float(image_time_sec[i])

        t_start = t_img - HALF_EXPOSURE
        t_end = t_img + HALF_EXPOSURE

        # exposure window가 OXTS 범위를 벗어나면 skip
        if t_start < time_array[0] or t_end > time_array[-1]:
            print(f"[SKIP] frame {i}, exposure window outside OXTS range")
            continue

        gyro_start = interpolate_vector(time_array, gyro_array, t_start)
        gyro_end = interpolate_vector(time_array, gyro_array, t_end)

        accel_start = interpolate_vector(time_array, accel_array, t_start)
        accel_end = interpolate_vector(time_array, accel_array, t_end)

        # gyro 사다리꼴 적분
        # wx, wy, wz는 rad/s로 가정.
        delta_theta = 0.5 * (gyro_start + gyro_end) * EXPOSURE_TIME

        delta_roll = delta_theta[0]
        delta_pitch = delta_theta[1]
        delta_yaw = delta_theta[2]

        # 1차 근사:
        # yaw 변화량 -> 이미지 x방향 shift
        # pitch 변화량 -> 이미지 y방향 shift
        #
        # 부호는 blur 방향에서 큰 의미가 없고, angle 계산에만 반영됨.
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

        img_path = IMAGE_DIR / fname
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        kernel = make_linear_motion_kernel(kernel_size, angle_deg)
        blurred = apply_motion_blur(img, kernel)

        out_path = OUTPUT_IMAGE_DIR / fname
        cv2.imwrite(str(out_path), blurred)

        records.append({
            "frame": i,
            "filename": fname,
            "image_timestamp": str(image_ts[i]),
            "image_time_sec": t_img,
            "t_start": t_start,
            "t_end": t_end,

            "wx_start": gyro_start[0],
            "wy_start": gyro_start[1],
            "wz_start": gyro_start[2],
            "wx_end": gyro_end[0],
            "wy_end": gyro_end[1],
            "wz_end": gyro_end[2],

            "ax_start": accel_start[0],
            "ay_start": accel_start[1],
            "az_start": accel_start[2],
            "ax_end": accel_end[0],
            "ay_end": accel_end[1],
            "az_end": accel_end[2],

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

        if i % 50 == 0:
            print(
                f"[{i:04d}/{n}] {fname} "
                f"L={L_final:.3f}, k={kernel_size}, angle={angle_deg:.1f}"
            )

    meta = pd.DataFrame(records)
    meta.to_csv(OUTPUT_META_CSV, index=False)

    print()
    print("===== DONE =====")
    print("saved blurred images:", len(meta))
    print("output image dir:", OUTPUT_IMAGE_DIR)
    print("meta csv:", OUTPUT_META_CSV)

    print()
    print("===== kernel summary =====")
    print(meta[["L_rot", "accel_dynamic_norm", "accel_boost", "L_final", "kernel_size", "angle_deg"]].describe())

    print()
    print("===== kernel size counts =====")
    print(meta["kernel_size"].value_counts().sort_index())


if __name__ == "__main__":
    main()
