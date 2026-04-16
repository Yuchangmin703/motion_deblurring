cat > README.md <<'EOF'
# motion_deblurring

KITTI 이미지에 motion blur를 적용하고, YOLO 기반 object detection 성능 저하를 분석하는 프로젝트입니다.

## Structure
- `notebooks/`: 실험 노트북
- `src/`: 재사용 코드
- `configs/`: 실험 설정
- `scripts/`: 실행 스크립트

## Dataset
KITTI dataset은 저장소에 포함하지 않습니다.
각자 `data/raw/kitti/` 아래에 다운로드해 사용합니다.
EOF