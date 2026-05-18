from pathlib import Path
from ultralytics import YOLO

ROOT = Path.home() / "motion_deblurring"

DATA_YAML = ROOT / "data/raw/kitti/kitti.yaml"
MODEL_PATH = ROOT / "notebooks/yolo11n.pt"

model = YOLO(str(MODEL_PATH))

model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    project=str(ROOT / "results/train"),
    name="yolo11n_kitti",
)
