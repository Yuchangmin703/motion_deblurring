from pathlib import Path
import glob
import numpy as np
import pandas as pd


ROOT = Path.home() / "motion_deblurring"

CALIB_DIR = ROOT / "data/raw/kitti_raw/calib/2011_09_26"

DRIVE_DIR = ROOT / "data/raw/kitti_raw/drives/2011_09_26/2011_09_26_drive_0009_sync"
OXTS_DIR = DRIVE_DIR / "oxts/data"

SAVE_DIR = ROOT / "data/interim/imu"
SAVE_PATH = SAVE_DIR / "2011_09_26_drive_0009_cam_imu.csv"


def read_calib_file(filepath):
    """
    KITTI calibration txt 파일을 읽어서 dict 형태로 반환.
    예: data["R"], data["T"], data["R_rect_00"]
    """
    data = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip()

            if line == "":
                continue

            key, value = line.split(":", 1)
            key = key.strip()

            # calib_time은 숫자 배열이 아니므로 건너뜀
            if key == "calib_time":
                continue

            data[key] = np.array([float(x) for x in value.strip().split()])

    return data


def get_imu_to_cam_rotation(calib_dir):
    """
    IMU 좌표계의 벡터를 rectified camera 좌표계로 변환하는 회전행렬 계산.

    v_cam = R_rect_00 @ R_velo_to_cam @ R_imu_to_velo @ v_imu
    """
    imu2velo_data = read_calib_file(calib_dir / "calib_imu_to_velo.txt")
    R_imu_to_velo = imu2velo_data["R"].reshape(3, 3)

    velo2cam_data = read_calib_file(calib_dir / "calib_velo_to_cam.txt")
    R_velo_to_cam = velo2cam_data["R"].reshape(3, 3)

    cam2cam_data = read_calib_file(calib_dir / "calib_cam_to_cam.txt")
    R_rect_00 = cam2cam_data["R_rect_00"].reshape(3, 3)

    R_imu_to_cam = R_rect_00 @ R_velo_to_cam @ R_imu_to_velo

    return R_imu_to_cam


def read_oxts_imu(oxts_file, R_imu_to_cam):
    """
    OXTS txt 한 개에서 acc, gyro를 읽고 카메라 좌표계로 변환.
    """
    with open(oxts_file, "r") as f:
        line = f.readline().strip().split()
        data = [float(x) for x in line]

    # KITTI OXTS format 기준
    # ax, ay, az
    acc_imu = np.array([data[11], data[12], data[13]])

    # wx, wy, wz
    gyro_imu = np.array([data[17], data[18], data[19]])

    acc_cam = R_imu_to_cam @ acc_imu
    gyro_cam = R_imu_to_cam @ gyro_imu

    return acc_cam, gyro_cam


def main():
    print("===== KITTI OXTS IMU 추출 시작 =====")
    print(f"CALIB_DIR: {CALIB_DIR}")
    print(f"OXTS_DIR : {OXTS_DIR}")
    print(f"SAVE_PATH: {SAVE_PATH}")

    if not CALIB_DIR.exists():
        raise FileNotFoundError(f"calib 폴더가 없습니다: {CALIB_DIR}")

    if not OXTS_DIR.exists():
        raise FileNotFoundError(f"oxts 폴더가 없습니다: {OXTS_DIR}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    R_imu_to_cam = get_imu_to_cam_rotation(CALIB_DIR)

    oxts_files = sorted(glob.glob(str(OXTS_DIR / "*.txt")))

    if len(oxts_files) == 0:
        raise FileNotFoundError(f"oxts txt 파일이 없습니다: {OXTS_DIR}")

    print(f"총 OXTS 파일 수: {len(oxts_files)}")

    results = []

    for oxts_file in oxts_files:
        oxts_path = Path(oxts_file)
        frame_id = oxts_path.stem

        acc_cam, gyro_cam = read_oxts_imu(oxts_path, R_imu_to_cam)

        gyro_norm = float(np.linalg.norm(gyro_cam))
        acc_norm = float(np.linalg.norm(acc_cam))

        results.append({
            "Frame": frame_id,
            "Acc_X": acc_cam[0],
            "Acc_Y": acc_cam[1],
            "Acc_Z": acc_cam[2],
            "Acc_Norm": acc_norm,
            "Gyro_X": gyro_cam[0],
            "Gyro_Y": gyro_cam[1],
            "Gyro_Z": gyro_cam[2],
            "Gyro_Norm": gyro_norm,
        })

    df = pd.DataFrame(results)
    df.to_csv(SAVE_PATH, index=False)

    print("===== 변환 완료 =====")
    print(df.head())
    print()
    print("Gyro_Norm 통계:")
    print(df["Gyro_Norm"].describe())
    print()
    print(f"저장 완료: {SAVE_PATH}")


if __name__ == "__main__":
    main()
