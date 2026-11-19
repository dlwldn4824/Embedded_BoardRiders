"""표지판 YOLO 가지치기(구조적 채널 프루닝) + 지식 증류 재학습.

1. 가지치기: torch-pruning 으로 학생 모델(교사와 같은 가중치에서 시작)의 채널을 --ratio 만큼 잘라 낸다.
   잘라 낼 채널은 가중치 크기(L2)가 작은 순서다. 검출 헤드의 마지막 층(박스·클래스 출력)은 자르지 않는다.
2. 지식 증류: 잘린 학생을 원래 YOLO 손실 + 교사의 출력을 따라 하는 손실로 다시 학습한다.
   - 클래스: 학생 점수가 교사의 확률(sigmoid)을 따라가게 (BCE)
   - 박스: 학생의 박스 분포(DFL)가 교사 분포를 따라가게 (온도 T 의 KL). 교사가 물체가 있다고 본 위치에 가중치
3. 결과 best.pt 를 sign/export_int8.py 로 INT8 변환한다 (기존 모델과 같은 방법).

준비 (메인 .venv 에서):
    pip install torch-pruning

사용법 (데스크톱 GPU 권장, 데이터는 v3 와 같은 data/yolo_v3):
    python sign/prune_kd.py --teacher weights/sign_v3_8n.pt --ratio 0.4 --data data/yolo_v3/data.yaml --epochs 60 --device 0
    python sign/export_int8.py runs/sign/prune40_kd/weights/best.pt --imgsz 320 --out weights/sign_v4_prune40_320_int8.tflite
    python sign/compare_int8.py weights/sign_v3_26n_320_int8.tflite weights/sign_v4_prune40_320_int8.tflite
"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from pruned_modules import split_c2f

ROOT = Path(__file__).resolve().parent.parent


def params(m):
    return sum(p.numel() for p in m.parameters())


def prune(model, ratio, imgsz):
    """model: ultralytics DetectionModel (제자리에서 잘린다). 잘린 뒤 파라미터 수를 돌려준다."""
    import torch_pruning as tp

    model.float().eval()
    x = torch.rand(1, 3, imgsz, imgsz)
    ref = model(x)[0]
    print(f"C2f → C2fSplit {split_c2f(model)}개 (출력 차이 {float((model(x)[0] - ref).abs().max()):.2e})")
    for p in model.parameters():
        p.requires_grad_(True)
    detect = model.model[-1]
    ignored = [detect.dfl] if hasattr(detect, "dfl") else []
    for seq in list(getattr(detect, "cv2", [])) + list(getattr(detect, "cv3", [])):
        ignored.append(seq[-1])          # 박스·클래스 출력 층은 채널 수가 정해져 있다
    for name in ("one2one_cv2", "one2one_cv3"):   # YOLO26 의 end2end 헤드
        for seq in getattr(detect, name, []) or []:
            ignored.append(seq[-1])

    def flat(out):
        """eval 출력 (y, {"boxes", "scores", "feats"}) 에서 텐서만 꺼낸다 (의존성 그래프 추적용)."""
        if isinstance(out, torch.Tensor):
            return [out]
        if isinstance(out, dict):
            return [t for v in out.values() for t in flat(v)]
        if isinstance(out, (list, tuple)):
            return [t for v in out for t in flat(v)]
        return []

    example = torch.zeros(1, 3, imgsz, imgsz)
    pruner = tp.pruner.MetaPruner(
        model, example, importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=ratio, ignored_layers=ignored, output_transform=flat,   # round_to 를 쓰면 16채널 층이 반으로 잘린다
    )
    pruner.step()
    return params(model)


class KDLoss:
    """원래 YOLO 손실에 교사 출력을 따라 하는 손실을 더한다. 출력 형식은 v8DetectionLoss 와 같다."""

    def __init__(self, base, teacher, alpha_cls=1.0, alpha_box=1.0, temp=2.0):
        self.base, self.teacher = base, teacher
        self.alpha_cls, self.alpha_box, self.temp = alpha_cls, alpha_box, temp

    def __call__(self, preds, batch):
        loss, items = self.base(preds, batch)
        p = preds[1] if isinstance(preds, tuple) else preds          # 학습 중에는 dict
        p = p.get("one2many", p)
        with torch.no_grad():
            self.teacher.to(batch["img"].device)
            t = self.teacher(batch["img"])
            t = t[1] if isinstance(t, tuple) else t
            t = t.get("one2many", t)
        s_cls, t_cls = p["scores"], t["scores"].float()
        kd_cls = F.binary_cross_entropy_with_logits(s_cls, t_cls.sigmoid())
        b, n = s_cls.shape[0], s_cls.shape[-1]
        s_box = p["boxes"].view(b, 4, -1, n) / self.temp
        t_box = t["boxes"].float().view(b, 4, -1, n) / self.temp
        kl = F.kl_div(F.log_softmax(s_box, 2), F.softmax(t_box, 2), reduction="none").sum(2).mean(1)  # (b, n)
        w = t_cls.sigmoid().amax(1)                                   # 교사가 물체로 본 위치일수록 크게
        kd_box = (kl * w).sum() / w.sum().clamp(min=1e-6) * self.temp ** 2
        kd = self.alpha_cls * kd_cls + self.alpha_box * kd_box
        loss = loss.clone()
        loss[1] = loss[1] + kd * b      # 화면의 cls 손실 칸에 증류 손실이 함께 찍힌다
        return loss, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, default=ROOT / "weights/sign_v3_8n.pt")
    ap.add_argument("--student", type=Path, default=None, help="가지치기 전 학생 가중치 (기본: 교사와 같음)")
    ap.add_argument("--ratio", type=float, default=0.4, help="잘라 낼 채널 비율")
    ap.add_argument("--data", type=Path, default=ROOT / "data/yolo_v3/data.yaml")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0", help="0 (CUDA) / mps / cpu")
    ap.add_argument("--kd", type=float, default=1.0, help="증류 손실 배율 (0 이면 가지치기 + 일반 재학습)")
    ap.add_argument("--fraction", type=float, default=1.0, help="학습 데이터 일부만 (빠른 확인용)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.loss import v8DetectionLoss

    teacher = YOLO(args.teacher).model.float().eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    student = deepcopy(YOLO(args.student or args.teacher).model)
    before = params(student)
    after = prune(student, args.ratio, args.imgsz)
    print(f"가지치기: 파라미터 {before / 1e6:.2f}M → {after / 1e6:.2f}M ({after / before:.0%})")

    kd = args.kd

    class KDTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            student.nc, student.names = self.data["nc"], self.data["names"]
            student.criterion = None
            return student

        def _setup_train(self, *a, **k):
            super()._setup_train(*a, **k)
            if kd > 0:   # 학습 모델에만 증류 손실을 붙인다 (EMA·검증은 원래 손실)
                self.model.criterion = KDLoss(v8DetectionLoss(self.model), teacher, kd, kd)

        def save_model(self):
            crit = getattr(self.model, "criterion", None)
            self.model.criterion = None         # 체크포인트에 교사와 이 스크립트의 클래스를 넣지 않는다
            try:
                super().save_model()
            finally:
                self.model.criterion = crit

    name = args.name or f"prune{round(args.ratio * 100)}_{'kd' if kd > 0 else 'ft'}"
    model = YOLO(args.teacher)
    model.train(
        trainer=KDTrainer, data=str(args.data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, project=str(ROOT / "runs/sign"), name=name, exist_ok=True, fraction=args.fraction,
        patience=20, seed=0, workers=4, fliplr=0.0, flipud=0.0, mosaic=1.0, mixup=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.6, translate=0.1, scale=0.5, lr0=0.002, warmup_epochs=1, plots=False,
    )
    best = ROOT / "runs/sign" / name / "weights/best.pt"
    m = YOLO(best).val(data=str(args.data), imgsz=args.imgsz, device=args.device, plots=False, verbose=False)
    print(f"\n== {name}  파라미터 {after / 1e6:.2f}M  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
    print(f"가중치: {best}")


if __name__ == "__main__":
    main()
