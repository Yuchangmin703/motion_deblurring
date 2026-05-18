from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data/processed/imu_blur/0014_run_0077_0250"
OUT_ROOT = PROJECT_ROOT / "results/imu_blur_0014/predictions"
COMPARE_DIR = PROJECT_ROOT / "results/imu_blur_0014/prediction_comparisons"

WEIGHT_CANDIDATES = [
    PROJECT_ROOT / "weights/yolo11n_kitti_best.pt",
    PROJECT_ROOT / "results/train/yolo11n_kitti/weights/best.pt",
    PROJECT_ROOT / "results/train/yolo11n_kitti/weights/last.pt",
]

CLASS_NAMES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "Person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc",
}

CONDITIONS = [
    "original",
    "exp_002ms",
    "exp_004ms",
    "exp_006ms",
    "exp_008ms",
    "exp_010ms",
    "exp_012ms",
    "exp_015ms",
    "exp_020ms",
    "exp_025ms",
    "exp_030ms",
]

COMPARE_CONDITIONS = [
    "original",
    "exp_010ms",
    "exp_020ms",
    "exp_030ms",
]

CONF_THRES = 0.25
IMGSZ = 640


def find_weight():
    for p in WEIGHT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "YOLO weight not found. Expected one of:\n"
        + "\n".join(str(p) for p in WEIGHT_CANDIDATES)
    )


def read_label_classes(label_path):
    classes = set()
    if not label_path.exists():
        return classes

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls_id = int(float(line.split()[0]))
        classes.add(cls_id)

    return classes


def select_frames_covering_classes(label_dir):
    """
    GT label 기준으로 클래스가 최대한 많이 포함되도록 대표 frame 선택.
    한 장에 모든 클래스가 들어있지 않을 수 있으므로 greedy set cover 방식 사용.
    """
    label_files = sorted(label_dir.glob("*.txt"))

    frame_classes = []
    all_classes = set()

    for p in label_files:
        cls_set = read_label_classes(p)
        if len(cls_set) == 0:
            continue
        frame_id = p.stem
        frame_classes.append((frame_id, cls_set))
        all_classes |= cls_set

    selected = []
    covered = set()
    remaining = frame_classes.copy()

    while remaining and covered != all_classes:
        best = max(remaining, key=lambda x: len(x[1] - covered))
        gain = best[1] - covered

        if len(gain) == 0:
            break

        selected.append(best)
        covered |= best[1]
        remaining.remove(best)

    rows = []
    for frame_id, cls_set in selected:
        rows.append({
            "frame_id": frame_id,
            "classes": ",".join(str(c) for c in sorted(cls_set)),
            "class_names": ",".join(CLASS_NAMES.get(c, str(c)) for c in sorted(cls_set)),
        })

    class_report = pd.DataFrame(rows)

    return selected, all_classes, covered, class_report


def make_comparison_grid(frame_id):
    imgs = []

    for cond in COMPARE_CONDITIONS:
        pred_path = OUT_ROOT / cond / f"{frame_id}.png"

        if not pred_path.exists():
            print(f"[WARN] missing prediction image: {pred_path}")
            return

        img = cv2.imread(str(pred_path))
        if img is None:
            print(f"[WARN] failed to read: {pred_path}")
            return

        # 보기 좋게 라벨 달기
        label = cond.replace("exp_", "").replace("ms", " ms")
        if cond == "original":
            label = "original"

        cv2.rectangle(img, (0, 0), (260, 38), (0, 0, 0), -1)
        cv2.putText(
            img,
            label,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        imgs.append(img)

    top = np.hstack([imgs[0], imgs[1]])
    bottom = np.hstack([imgs[2], imgs[3]])
    grid = np.vstack([top, bottom])

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPARE_DIR / f"compare_{frame_id}.png"
    cv2.imwrite(str(out_path), grid)
    print("[SAVE comparison]", out_path)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    weight = find_weight()
    print("===== Using YOLO weight =====")
    print(weight)

    model = YOLO(str(weight))

    # 1. 각 exposure 조건별 전체 174장에 대해 박스 이미지 생성
    for cond in CONDITIONS:
        source_dir = DATA_ROOT / cond / "images/val"
        out_dir = OUT_ROOT / cond

        if not source_dir.exists():
            print("[SKIP] missing source:", source_dir)
            continue

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("========================================")
        print("YOLO predict:", cond)
        print("source:", source_dir)
        print("output:", out_dir)
        print("========================================")

        results = model.predict(
            source=str(source_dir),
            imgsz=IMGSZ,
            conf=CONF_THRES,
            save=False,
            verbose=False,
        )

        for r in results:
            # r.path = input image path
            in_path = Path(r.path)
            plotted = r.plot()  # BGR image with boxes

            out_path = out_dir / in_path.name
            cv2.imwrite(str(out_path), plotted)

        print("[DONE]", cond, "saved to", out_dir)

    # 2. GT label 기준으로 클래스가 최대한 다 들어가는 대표 frame 선택
    label_dir = DATA_ROOT / "original" / "labels/val"
    selected, all_classes, covered, class_report = select_frames_covering_classes(label_dir)

    print()
    print("===== GT class coverage in this 174-frame subset =====")
    print("all classes present:", sorted(all_classes), [CLASS_NAMES.get(c, str(c)) for c in sorted(all_classes)])
    print("covered by selected:", sorted(covered), [CLASS_NAMES.get(c, str(c)) for c in sorted(covered)])

    missing = set(CLASS_NAMES.keys()) - all_classes
    if missing:
        print("classes NOT present in this subset:", sorted(missing), [CLASS_NAMES.get(c, str(c)) for c in sorted(missing)])

    report_path = COMPARE_DIR / "selected_frames_for_class_coverage.csv"
    class_report.to_csv(report_path, index=False)
    print("[SAVE]", report_path)
    print(class_report)

    # 3. 선택된 대표 frame에 대해 original/10/20/30ms 비교 이미지 생성
    for frame_id, cls_set in selected:
        make_comparison_grid(frame_id)

    print()
    print("===== DONE =====")
    print("all predicted images:", OUT_ROOT)
    print("comparison images:", COMPARE_DIR)


if __name__ == "__main__":
    main()
