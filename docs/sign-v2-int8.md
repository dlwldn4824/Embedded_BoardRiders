# 표지판 v2 학습 · TFLite INT8 변환 결과 (2026-11-05)

모든 수치는 같은 val (`data/yolo_clean_v2` val, 359장)에서 imgsz 640 으로 쟀다.

## 1. v2 학습

- 명령: `python sign/train.py --name v2 --epochs 40 --device 0` (RTX 5060 Ti)
- 37 epoch 에서 조기 종료됐다 (patience 15). 가중치: `runs/sign/v2/weights/best.pt` (git 제외). INT8 은 `weights/` 에 있다.
- 좌우 반전 증강(fliplr)은 껐다.

| class | baseline mAP50 | v2 P | v2 R | v2 mAP50 |
|---|---|---|---|---|
| crosswalk | 0.894 | 0.900 | 0.829 | **0.924** |
| green_light | 0.995 | 0.980 | 1.000 | 0.995 |
| horn | 0.995 | 0.988 | 1.000 | 0.995 |
| left | 0.995 | 0.925 | 1.000 | 0.993 |
| red_light | 0.995 | 0.988 | 1.000 | 0.995 |
| red_line | 0.868 | 0.866 | 0.850 | 0.861 |
| right | 0.968 | 0.962 | 0.974 | 0.969 |
| speed_20 | 0.995 | 0.959 | 1.000 | 0.987 |
| stop | 0.995 | 0.978 | 1.000 | 0.995 |
| straight | 0.995 | 0.983 | 1.000 | 0.995 |
| **전체 mAP50 / mAP50-95** | **0.970 / 0.736** | | | **0.971 / 0.749** |

- 기준선(`weights/sign_baseline_ep25.pt`)과 mAP50 은 같은 수준이다. 박스 정확도(mAP50-95)는 +0.013, crosswalk 는 +0.03 올랐다.
- 가장 약한 클래스는 두 모델 모두 red_line 이다.

## 2. TFLite INT8 변환

```bash
# 따로 만든 .venv-tf 에서 (설치 방법은 sign/export_int8.py 맨 위 주석)
python sign/export_int8.py runs/sign/v2/weights/best.pt --out weights/sign_v2_int8.tflite
python sign/compare_int8.py runs/sign/v2/weights/best.pt weights/sign_v2_fp32.tflite weights/sign_v2_int8.tflite
```

- ultralytics 8.4 의 `format='tflite'` 는 Linux x86 / macOS 에서만 돌아간다. 그래서 같은 경로를 직접 밟았다: ONNX → onnx2tf.
- representative 데이터는 train 이미지 200장(전체 1573장 중, seed 0 무작위)이다.
- 양자화 방식은 INT8 per-tensor 이고, 입출력은 FP32 다. per-channel 도 해 봤지만 mAP50 이 0.947 로 더 낮았다.
- Detect head 가 박스를 0~1 로 내도록 바꿔서 내보낸다 (예전 ultralytics tflite 와 같은 방식).
  - 이렇게 하지 않으면 박스(0~640)와 점수(0~1)가 한 INT8 텐서에 들어가서 점수가 0 이 된다 (mAP 0).
- 클래스 이름은 tflite 끝에 `metadata.json` 으로 붙여 두었다. `YoloSignDetector` 가 그대로 읽는다.

| 가중치 | 크기 (MB) | mAP50 | mAP50-95 | CPU 추론 (ms) |
|---|---|---|---|---|
| v2 `best.pt` (PyTorch FP32) | 6.51 | 0.971 | 0.749 | 23.7 |
| `weights/sign_v2_fp32.tflite` | 12.70 | 0.966 | 0.747 | 33.9 |
| **`weights/sign_v2_int8.tflite`** | **3.25** | **0.949** | **0.694** | **16.1** |

- 추론 시간은 **데스크톱 CPU** (Intel Core Ultra 7 265F) 에서 batch 1, 전처리를 뺀 중앙값이다. **Pi 5 가 아니다.**
  - Pi 에서는 몇 배 느릴 것이다. Pi 에서 꼭 다시 재야 한다.
- INT8 은 FP32 tflite 대비 크기가 1/4 이고, mAP50 은 −0.017 이다.

클래스별 mAP50:

| class | best.pt | FP32 tflite | INT8 tflite |
|---|---|---|---|
| crosswalk | 0.924 | 0.920 | 0.877 |
| green_light | 0.995 | 0.995 | 0.995 |
| horn | 0.995 | 0.995 | 0.995 |
| left | 0.993 | 0.995 | 0.987 |
| red_light | 0.995 | 0.995 | 0.994 |
| red_line | 0.862 | 0.832 | **0.716** |
| right | 0.969 | 0.966 | 0.968 |
| speed_20 | 0.987 | 0.972 | 0.970 |
| stop | 0.995 | 0.995 | 0.995 |
| straight | 0.995 | 0.995 | 0.995 |

### 주행 기준값(conf ≥ 0.5)에서의 재현율

`SIGN_MIN_CONF = 0.5` 에서 정답 박스(IoU > 0.5)를 찾은 비율이다.

| | 전체 | crosswalk | red_line | 나머지 8 클래스 |
|---|---|---|---|---|
| best.pt | 0.883 | 0.81 | 0.62 | 0.97 ~ 1.00 |
| INT8 | 0.863 | 0.78 | **0.53** | 0.97 ~ 1.00 |

- 맞힌 검출의 conf 중앙값은 INT8 0.83, `.pt` 0.87 이다. 양자화로 점수가 크게 낮아지지는 않았다.
- **red_line(도착 정지선)** 은 conf 0.5 에서 절반쯤 놓친다.
  - 다만 주행 코드는 15프레임 안에 2번만 잡히면 되므로, 실차에서 곧바로 문제가 되지는 않을 수 있다.
  - 놓치면 `drive/config.py` 의 `SIGN_CLASS_RULES["red_line"]` 에 `"min_conf": 0.3` 을 넣어 본다.

## 3. Pi 에서 실행하기

```bash
# Pi 5 (64-bit Raspberry Pi OS), 레포 루트에서
git pull
pip install ultralytics ai-edge-litert simple-pid gpiozero opencv-python pyyaml
python3 -m drive.main --mode real --signs weights/sign_v2_int8.tflite   # v3 가 나왔다: docs/sign-v3.md 의 sign_v3_8n_int8.tflite 를 쓴다
```

- 처음 실행할 때 `표지판 검출기: weights/sign_v2_int8.tflite` 가 출력되는지 본다.
- 표지판 검출은 별도 스레드(`SignWorker`)에서 돈다. 검출이 느려도 주행 루프는 막히지 않는다. 다만 표지판 반응은 그만큼 늦어진다.
- Pi 에서 지연 시간을 잰다: `python3 sign/compare_int8.py weights/sign_v2_int8.tflite` 의 ms 값.
  - 너무 느리면 `--imgsz 320` 으로 다시 학습하는 것(`sign/train.py --imgsz 320 --name n320`)을 검토한다. 이때 `SIGN_IMGSZ` 도 같이 바꾼다.

### 실차에서 다시 맞춰야 할 값 ([drive-changes.md](drive-changes.md) 참고)

| 값 (`drive/config.py` 등) | 맞추는 방법 |
|---|---|
| `TURN_ENTRY_FRAMES`, `TURN_FRAMES` | 실제 교차로 크기와 바퀴 속도에 맞춰 좌·우회전을 반복해 본다 |
| `RIGHT_WHEEL_TRIM` (`motors.py`) | 감속 직진을 2m 시켜 보고 휘는 쪽을 보정한다 |
| `SIGN_MIN_BOX_H`, `SIGN_CLASS_RULES` | 표지판 앞 몇 cm 에서 반응할지 정해서 맞춘다 |
| 정지선 `min_y2` (현재 380) | 정지선이 차 앞에 왔을 때의 y2 를 보고 정한다 |
| `SIGN_MIN_CONF` / red_line `min_conf` | INT8 에서 red_line 이 약하다 (위 표). 놓치면 낮춘다 |

## 4. 주의

- YOLO 학습 이미지(밝은 회색 도로)와 주행 코드를 맞춘 카메라 사진(어두운 도로)은 색감이 다르다.
- val 점수는 학습 데이터와 같은 분포에서 잰 값이다. **실제 차 카메라로 표지판이 잡히는지 꼭 확인한다.**
  - 잘 안 잡히면 실차 카메라로 찍은 사진을 라벨링해서 데이터를 보강한다.
