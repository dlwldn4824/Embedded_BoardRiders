"""표지판 YOLOv8n 학습.

- 좌우 반전 증강은 끈다 (left/right 클래스가 뒤바뀜). right 보강은 flip_left_to_right.py 로 라벨까지 바꿔서 한다.
- val 은 clean_dataset.py 가 연속 프레임 블록 단위로 나눈 것이라 train 과 거의 겹치지 않는다.

사용법:
    python sign/train.py                         # 기준선: yolov8n, 640
    python sign/train.py --imgsz 320 --name n320  # Pi 속도용 비교
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/yolo_clean_v2/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps", help="mps (Mac) / 0 (CUDA) / cpu")
    ap.add_argument("--name", default="baseline")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
        project=str(ROOT / "runs/sign"), name=args.name, exist_ok=True,
        patience=15, seed=0, workers=4,
        fliplr=0.0, flipud=0.0, mosaic=1.0, mixup=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5,
        plots=True,
    )

    best = ROOT / "runs/sign" / args.name / "weights/best.pt"
    metrics = YOLO(best).val(data=str(args.data), imgsz=args.imgsz, device=args.device, split="val", plots=False)
    names = metrics.names
    print(f"\n== {args.name}  mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}")
    print(f"{'class':12s} {'P':>6s} {'R':>6s} {'mAP50':>6s}")
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, _ = metrics.box.class_result(i)
        print(f"{names[int(c)]:12s} {p:6.3f} {r:6.3f} {ap50:6.3f}")
    print(f"가중치: {best}")


if __name__ == "__main__":
    main()
