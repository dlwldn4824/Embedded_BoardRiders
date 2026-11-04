"""YOLO .pt → TFLite INT8 (Pi 용). Windows 에서도 돈다.

ultralytics 8.4 의 format='tflite' 는 Linux x86 / macOS 에서만 되므로, 예전 ultralytics 와 같은 경로를 직접 밟는다.
    .pt → ONNX (ultralytics) → 박스 좌표 0~1 정규화 → onnx2tf (INT8, representative = train 이미지 일부) → metadata.json 붙이기

결과는 INT8 내부 연산 + FP32 입출력(ultralytics 의 *_int8.tflite 와 같은 형식)이라
drive/signs.py 의 YoloSignDetector 가 그대로 읽는다 (클래스 이름은 붙여 둔 metadata 에서 읽힌다).

준비 (메인 .venv 와 따로):
    python -m venv .venv-tf
    .venv-tf\\Scripts\\activate
    pip install "tensorflow==2.19.*" tf_keras onnx2tf onnx onnx_graphsurgeon sng4onnx onnxruntime onnxslim psutil ai-edge-litert pyyaml ultralytics

사용법:
    python sign/export_int8.py runs/sign/v2/weights/best.pt --out weights/sign_v2_int8.tflite
"""
import argparse
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def calib_array(img_dir, n, size, seed=0):
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(seed).shuffle(files)
    files = files[:n]
    arr = np.empty((len(files), size, size, 3), np.float32)
    for i, f in enumerate(files):
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)   # 한글 경로 대응
        arr[i] = cv2.cvtColor(letterbox(img, size), cv2.COLOR_BGR2RGB) / 255.0
    return arr


def patch_detect_normalized(imgsz):
    """Detect head 가 xywh 를 픽셀이 아니라 0~1 로 내도록 바꾼다 (예전 ultralytics tflite export 와 같은 방식).

    - ultralytics 의 tflite 백엔드는 xywh 에 이미지 크기를 곱하므로 정규화된 좌표를 기대한다.
    - INT8 은 텐서마다 스케일이 하나라, 박스(0~640)와 점수(0~1)가 같은 concat 에 들어가면 점수가 0 으로 뭉개진다.
      그래서 디코딩 단계에서부터 strides / imgsz 를 곱해 concat 전에 0~1 이 되게 한다. (dist2bbox 가 선형이라 값은 같다)
    """
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.tal import make_anchors

    def _get_decode_boxes(self, x):
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        norm = self.strides / imgsz
        return self.decode_bboxes(self.dfl(x["boxes"]) * norm, self.anchors.unsqueeze(0) * norm, x.get("angle"))

    Detect._get_decode_boxes = _get_decode_boxes


def disable_end2end(model):
    """YOLO26 처럼 NMS 없는(end2end) 모델은 one2many 출력 (1, 4+nc, N) 으로 내보낸다.

    end2end 출력은 (x1, y1, x2, y2, score, class) 가 한 텐서라, INT8 에서 클래스 번호(0~9)와 좌표가 한 스케일로 뭉개진다.
    ultralytics 도 LiteRT INT8 에서는 같은 이유로 end2end 를 끈다. NMS 는 Pi 에서 ultralytics 가 한다.
    """
    cls = type(model.model)
    if not isinstance(getattr(cls, "end2end", None), property) or not model.model.end2end:
        return False

    def _set(self, value):
        self.model[-1].end2end = False   # exporter 가 True 로 돌려놓지 못하게 막는다

    cls.end2end = property(lambda self: False, _set)
    model.model.model[-1].end2end = False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib", type=Path, default=ROOT / "data/yolo_clean_v2/images/train")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--quant-type", choices=["per-channel", "per-tensor"], default="per-tensor")  # per-channel 은 v2 에서 mAP50 0.947 로 더 낮았다
    args = ap.parse_args()

    import onnx2tf
    from ultralytics import YOLO

    model = YOLO(args.weights)
    # TensorFlow 는 Windows 에서 한글 경로를 잘 못 다루므로 임시 폴더(영문 경로)에서 작업한다.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        patch_detect_normalized(args.imgsz)
        if disable_end2end(model):
            print("end2end 모델 → one2many 출력으로 내보낸다 (NMS 는 추론 쪽에서)")
        onnx =Path(model.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True))
        shutil.copy(onnx, tmp / "model.onnx")
        onnx.unlink()   # 정규화 좌표용 ONNX 라 일반 ONNX 와 섞이지 않게 지운다

        calib = calib_array(args.calib, args.n_calib, args.imgsz)
        np.save(tmp / "calib.npy", calib)
        # onnx2tf 가 출력 비교용으로 받는 샘플 파일. 없으면 인터넷에서 받다가 실패하므로 직접 만든다.
        sample = np.stack([cv2.resize(x, (128, 128)) for x in calib[:20]]).astype(np.float32)
        np.save(tmp / "calibration_image_sample_data_20x128x128x3_float32.npy", sample)
        print(f"representative: {len(calib)}장 ({args.calib})")

        import os
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            onnx2tf.convert(
                input_onnx_file_path="model.onnx", output_folder_path="out", not_use_onnxsim=True,
                verbosity="error", output_integer_quantized_tflite=True, quant_type=args.quant_type,
                custom_input_op_name_np_data_path=[["images", "calib.npy", [[[[0, 0, 0]]]], [[[[1, 1, 1]]]]]],
            )
        finally:
            os.chdir(cwd)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp / "out/model_integer_quant.tflite", args.out)
        shutil.copy(tmp / "out/model_float32.tflite", args.out.with_name(args.out.stem.replace("int8", "fp32") + ".tflite")
                    if "int8" in args.out.stem else args.out.with_name(args.out.stem + "_fp32.tflite"))

    meta = {"description": f"{args.weights} INT8 {args.quant_type} (onnx2tf, train {args.n_calib}장 보정)", "task": "detect",
            "stride": int(max(model.model.stride)), "batch": 1, "imgsz": [args.imgsz, args.imgsz],
            "names": {int(k): v for k, v in model.names.items()}}
    for f in [args.out, *args.out.parent.glob(args.out.stem.replace("int8", "fp32") + ".tflite")]:
        with zipfile.ZipFile(f, "a", zipfile.ZIP_DEFLATED) as z:
            z.writestr("metadata.json", json.dumps(meta, ensure_ascii=False))
        print(f"→ {f} ({f.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
