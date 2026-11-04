# 표지판 v3: 데이터 보강 · YOLO 버전 비교 · 320 해상도 (2026-11-05)

[v2 결과](sign-v2-int8.md) 이후 Pi 없이 할 수 있는 개선 네 가지를 해 봤다.

## 결론

- **Pi 에서는 `weights/sign_v3_8n_int8.tflite` 를 쓴다.**
  - INT8 중에서 정확도(mAP50 0.957, 주행 기준 재현율 0.906)가 가장 좋고, 640 모델 중 가장 빠르다.
- v2 INT8 보다 나아진 점:
  - mAP50: 0.949 → 0.957
  - conf 0.5 재현율: 0.863 → 0.906
  - **red_line 재현율: 0.53 → 0.72**
- Pi 가 너무 느리면 `weights/sign_v3_26n_320_int8.tflite` 를 쓴다. 추론 시간이 약 0.6배지만, red_line 정확도가 떨어진다.

```bash
python3 -m drive.main --mode real --signs weights/sign_v3_8n_int8.tflite
```

320 모델을 쓸 때는 `drive/config.py` 의 `SIGN_IMGSZ = 320` 도 같이 바꾼다.

## 1. 무엇을 바꿨나

데이터는 `sign/make_v3.py` 로 만든다 (`data/yolo_v3/`, 절대 경로가 들어가서 git 에는 올리지 않는다).

| 항목 | 내용 |
|---|---|
| 실차 카메라 배경 151장 | `data/raw/run_20261028_191415` 에서 고른 라벨 없는 배경이다. 어두운 도로, 의자, 문틀에서 나는 오검출을 줄이려고 넣었다. 기준 모델이 아무것도 못 잡은 프레임 중에서, crosswalk 검출 ±10 프레임 안은 뺐다 |
| red_line 오버샘플링 | red_line 이 있는 train 이미지 191장을 목록에 3번씩 넣었다 |
| 밝기 증강 | `hsv_v` 0.4 → 0.6 (`sign/train.py --hsv-v 0.6`) |
| val | v2 와 같다 (359장). 모든 비교가 같은 val 이다 |

train 목록은 2106줄이다 (그중 배경 189장 = 실차 151 + 원래 데이터 38).

```bash
python sign/make_v3.py --ref runs/sign/v2/weights/best.pt
python sign/train.py --data data/yolo_v3/data.yaml --model yolov8n.pt --name v3_8n --epochs 40 --device 0 --hsv-v 0.6
#   --model yolo11n.pt / yolo26n.pt,  --imgsz 320 도 같은 방식
```

## 2. 학습 결과 (PyTorch, GPU val)

| 모델 | imgsz | mAP50 | mAP50-95 | red_line mAP50 | crosswalk mAP50 |
|---|---|---|---|---|---|
| v2 (YOLOv8n) | 640 | 0.971 | 0.749 | 0.861 | 0.924 |
| v3 YOLOv8n | 640 | 0.978 | 0.758 | 0.947 | 0.930 |
| v3 YOLO11n | 640 | 0.978 | 0.755 | 0.923 | 0.934 |
| **v3 YOLO26n** | 640 | **0.983** | **0.768** | **0.962** | **0.940** |
| v3 YOLO26n | 320 | 0.977 | 0.756 | 0.928 | 0.934 |

- 데이터 보강만으로 red_line 이 0.861 → 0.947 (v8n) 로 크게 올랐다. 나머지 8 클래스는 모두 0.95 이상이다.
- PyTorch 에서는 YOLO26n 이 가장 좋다.
- v8n 과 11n 은 40 epoch 째가 최고점이었다. epoch 를 늘리면 조금 더 오를 수 있다.

## 3. INT8 변환 후 (CPU, batch 1)

`sign/export_int8.py` 로 변환하고, `sign/compare_int8.py` 로 쟀다. 재현율은 주행 기준(conf ≥ 0.5)이다.

| 가중치 | 크기 (MB) | mAP50 | mAP50-95 | CPU ms | 재현율 전체 | crosswalk | red_line |
|---|---|---|---|---|---|---|---|
| v2 INT8 | 3.25 | 0.949 | 0.694 | 16.2 | 0.863 | 0.78 | 0.53 |
| **v3 YOLOv8n INT8** | **3.25** | **0.957** | **0.703** | **16.3** | **0.906** | **0.84** | 0.72 |
| v3 YOLO26n INT8 | 2.55 | 0.950 | 0.661 | 27.9 | 0.899 | 0.82 | **0.82** |
| v3 YOLO26n 320 INT8 | 2.53 | 0.926 | 0.627 | 9.9 | 0.875 | 0.78 | 0.78 |
| (참고) v3 YOLOv8n `.pt` | 6.52 | 0.978 | 0.758 | 25.0 | 0.920 | 0.85 | 0.88 |
| (참고) v3 YOLO26n `.pt` | 5.48 | 0.983 | 0.767 | 24.8 | 0.915 | 0.84 | 0.88 |

- ms 는 **데스크톱 CPU (Intel Core Ultra 7 265F)** 에서 잰 값이다. Pi 5 에서 다시 재야 한다.
  - 모델끼리 상대 비교로만 본다.
- **YOLO26n 은 INT8 에서 손해가 크다.**
  - mAP50-95 가 0.767 → 0.661 로 떨어진다.
  - CPU 시간도 v8n INT8 보다 1.7배 느리다 (주의(attention) 블록 쪽이 INT8 에서 잘 빨라지지 않는 것으로 보인다).
  - 그래서 PyTorch 에서는 가장 좋지만 Pi 용 기본값으로는 고르지 않았다.
  - 대신 red_line 재현율은 가장 높다 (0.82). 실차에서 정지선을 자주 놓치면 이 모델을 써 본다.
- YOLO26 은 원래 NMS 없는(end2end) 구조다.
  - 하지만 INT8 에서는 클래스 번호와 좌표가 한 텐서에 섞여 망가지므로, 일반 출력으로 내보냈다 (ultralytics 도 같은 처리를 한다).
  - NMS 는 Pi 에서 ultralytics 가 한다.
- 320 INT8 은 640 v8n INT8 보다 0.6배 시간이지만, mAP50 은 −0.03 이고 red_line mAP50 은 0.585 로 약하다.

## 4. 실차 카메라 오검출 (학습에 안 넣은 run)

`sign/eval_domain.py` 로 쟀다. 대상은 `data/raw/run_20261028_191445` (344장, 멀리 보이는 횡단보도 말고는 표지판 없음)다.

| 가중치 | 표지판 오검출 (conf ≥ 0.5) | 표지판 오검출 (conf ≥ 0.25) | crosswalk 검출 수 (conf ≥ 0.5 / 0.25) |
|---|---|---|---|
| v2 INT8 | 0 | 0 | 55 / 98 |
| v3 YOLOv8n INT8 | 0 | 0 | 3 / 10 |
| v3 YOLO26n INT8 | 0 | 1 (red_light) | 1 / 10 |
| v3 YOLO26n 320 INT8 | 0 | – | 4 / – |

- 주행 기준(conf 0.5)에서는 모든 모델이 이 run 에서 표지판 오검출 0 건이다.
- **v3 는 실차 카메라에서 멀리 보이는 횡단보도를 거의 잡지 않는다** (v2 는 잡았다).
  - 실차 배경 이미지를 넣은 영향으로 보인다.
  - crosswalk 는 주행 이벤트를 만들지 않으므로(`drive/config.py`) 주행 동작에는 영향이 없다.
  - 다만 실차에서 가까운 횡단보도까지 못 잡는다면, 실차 사진에 crosswalk 라벨을 달아 보강해야 한다.
- 이 run 에는 표지판이 없어서, **실차 카메라에서 표지판을 잘 잡는지(재현율)는 아직 모른다.**
  - 실차에서 표지판 앞 사진을 찍어(`--record`) 확인해야 한다.

## 5. 레포에 올린 가중치

| 파일 | 용도 |
|---|---|
| `weights/sign_v3_8n_int8.tflite` | **Pi 기본값** |
| `weights/sign_v3_26n_int8.tflite` | red_line 을 자주 놓칠 때 |
| `weights/sign_v3_26n_320_int8.tflite` | Pi 가 너무 느릴 때 (`SIGN_IMGSZ = 320`) |
| `weights/sign_v3_8n.pt`, `weights/sign_v3_26n.pt` | 추가 학습·재변환용 원본 |

FP32 tflite 는 크기만 크고 Pi 에서 쓸 일이 없어서 올리지 않았다. `sign/export_int8.py` 가 같이 만들어 준다.
