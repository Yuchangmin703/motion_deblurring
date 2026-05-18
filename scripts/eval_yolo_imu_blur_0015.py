from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data/processed/imu_blur/0015_run_0182_0291"

WEIGHT_CANDIDATES = [
    PROJECT_ROOT / "weights/yolo11n_kitti_best.pt",
    PROJECT_ROOT / "results/train/yolo11n_kitti/weights/best.pt",
    PROJECT_ROOT / "results/train/yolo11n_kitti/weights/last.pt",
]

OUT_DIR = PROJECT_ROOT / "results/imu_blur_0015"
PLOT_DIR = OUT_DIR / "figures"

EXPOSURE_TIMES_MS = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]


def find_weight():
    for p in WEIGHT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "YOLO weight not found. Expected one of:\n"
        + "\n".join(str(p) for p in WEIGHT_CANDIDATES)
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    weight = find_weight()
    print("===== Using YOLO weight =====")
    print(weight)

    model = YOLO(str(weight))

    conditions = [("original", 0)]
    conditions += [(f"exp_{ms:03d}ms", ms) for ms in EXPOSURE_TIMES_MS]

    summary_rows = []

    for cond, exposure_ms in conditions:
        data_yaml = DATA_ROOT / cond / "data.yaml"

        if not data_yaml.exists():
            print(f"[SKIP] missing data.yaml: {data_yaml}")
            continue

        print()
        print("========================================")
        print(f"YOLO evaluation: {cond}")
        print(f"exposure_ms: {exposure_ms}")
        print(f"data_yaml: {data_yaml}")
        print("========================================")

        metrics = model.val(
            data=str(data_yaml),
            imgsz=640,
            batch=16,
            project=str(OUT_DIR / "val_runs"),
            name=cond,
            plots=True,
            save_json=False,
            verbose=True,
        )

        box = metrics.box

        summary_rows.append({
            "condition": cond,
            "exposure_ms": exposure_ms,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("===== SAVE SUMMARY =====")
    print(summary_path)
    print(summary)

    for metric in ["precision", "recall", "mAP50", "mAP50_95"]:
        plt.figure()
        plt.plot(summary["exposure_ms"], summary[metric], marker="o")
        plt.xlabel("Exposure time (ms)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs exposure time")
        plt.grid(True)

        out_path = PLOT_DIR / f"overall_{metric}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print("[SAVE]", out_path)


if __name__ == "__main__":
    main()
