# 데스크톱 이어받기 (2026-11-04 기준)

## 1. 데스크톱에서 할 일

```bash
git clone https://github.com/dlwldn4824/Embedded_BoardRiders.git 임베디드
cd 임베디드
python -m venv .venv
```

활성화한 뒤 패키지를 설치한다.
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

```bash
pip install ultralytics simple-pid gpiozero opencv-python pyyaml pytest
python -m pytest tests -q
```

`pytest` 는 15개가 통과해야 한다.

## 2. 데스크톱 Claude 에게 붙여넣을 말

```
이 레포(임베디드)는 광운대 임베디드 AI 최적화 과목 Raspberry Pi 5 자율주행 프로젝트야.
먼저 HANDOFF.md, README.md, docs/drive-changes.md, docs/data-report.md 를 읽고 현재 상태를 파악해줘.

이어서 할 일:
1. data/yolo_clean_v2 로 YOLOv8n 표지판 모델을 학습해 (v2).
   python sign/train.py --name v2 --epochs 40 --device 0   (NVIDIA GPU 없으면 --device cpu)
   끝나면 클래스별 P/R/mAP50 표를 정리하고, weights/sign_baseline_ep25.pt (기준선) 와 같은 val 로 비교해.
2. v2 best.pt 를 Pi 용 TFLite INT8 로 변환해 (representative 데이터는 train 이미지 일부).
   FP32 대비 크기, mAP50, CPU 추론 시간을 표로 남겨. 결과 가중치는 weights/ 에 넣고 커밋해.
3. 팀원이 Pi 에서 바로 돌릴 수 있게 실행 방법을 정리해:
   python3 -m drive.main --mode real --signs weights/<int8 파일>
   이 명령으로 돌린다. docs/drive-changes.md 의 "실차에서 다시 맞춰야 할 값" 도 같이 안내해.

주의:
- 학습 데이터(data/)는 공개 레포에 없다. 팀 내부에서 data/ 폴더로 복사해 쓴다.
- 좌우 반전 증강(fliplr)은 끄고 학습한다. right 는 flip_left_to_right.py 로 이미 합성돼 있다.
- YOLO 데이터(밝은 회색 도로)와 팀 주행 코드가 맞춰진 카메라(어두운 도로)는 색감이 다르다. 실제 차 카메라로 확인이 필요하다.
```

## 3. 지금까지 된 것
- `drive/`: 팀 규칙 기반 주행 코드에 YOLO 표지판 연결, 교차로 회전, 정지·신호등·서행·도착 동작을 추가했다. sim 과 테스트로 확인했다. → [docs/drive-changes.md](docs/drive-changes.md)
- `data/yolo_clean_v2/`: 학습용 데이터셋. 고유 이미지 1779장, train 1420 + 반전 right 153, val 359.
  - 원본 val 의 80% 가 train 과 같은 이미지였던 문제를 고쳤다.
  - 라벨이 없던 187장은 사전 라벨링한 뒤 사람이 검토했다.
  - 연속 프레임이 train/val 에 섞이지 않도록 블록 단위로 나눴다.
- `weights/sign_baseline_ep25.pt`: 기준선 모델. 검토 전 데이터로 학습했고, 25 epoch 에서 val mAP50 0.973, mAP50-95 0.737.
- **v2 학습 + TFLite INT8 (2026-11-05)** → [docs/sign-v2-int8.md](docs/sign-v2-int8.md)
  - v2: val mAP50 0.971, mAP50-95 0.749 (같은 val 에서 기준선은 0.970 / 0.736).
  - `weights/sign_v2_int8.tflite`: 3.25MB, mAP50 0.949. Pi 실행: `python3 -m drive.main --mode real --signs weights/sign_v2_int8.tflite`
  - 변환 `sign/export_int8.py`, 비교 `sign/compare_int8.py`. Windows 에서도 돈다 (`.venv-tf` 따로 만듦).
- `deploy/collect.py`: 차선 수집 코드. 차동 구동이고, 연속 조향을 기록한다. 아직 Pi 에서 돌려 보지 않았다.

## 4. 아직 안 된 것
- Pi 5 에서 INT8 추론 시간 측정 (데스크톱 CPU 에서는 16ms).
- 실차 확인: 카메라 색감, 교차로 타이밍, 표지판 반응 거리. INT8 에서 red_line 재현율이 낮다 (conf 0.5 에서 0.53).
- 차선 CNN 회귀 모델: 규칙 기반 주행 기록(`--record` + CSV `fid`)으로 라벨을 모은 뒤 진행한다.

## 5. 데이터
- 학습 데이터는 공개 레포에 넣지 않는다. 팀 내부에서 `data/` 폴더로 복사해 쓴다.
  - `data/raw/`: 원본 `yolo_dataset`, 차선 주행 run 2개(`run_20261028_*`).
  - `data/to_label_review/`: 라벨이 없던 187장과 검토를 마친 라벨.
  - `data/yolo_clean_v2/`: 바로 학습할 수 있는 데이터셋.
- `data/yolo_clean_v2/` 를 다시 만들 때:
  ```bash
  python sign/clean_dataset.py data/raw/yolo_dataset data/yolo_clean_v2 --extra data/to_label_review
  python sign/flip_left_to_right.py data/yolo_clean_v2
  ```
