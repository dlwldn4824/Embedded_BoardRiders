# 임베디드 · 보드타는 사람들

2026 광운대학교 매치업 심화과정 「임베디드 보드를 활용한 인공지능 시스템 최적화 실무」
Raspberry Pi 5 한 대로 트랙을 스스로 달리고, 표지판과 신호를 지키는 자율주행 차 프로젝트입니다.

- **팀:** 조성호(팀장) · 이지우 · 하서현 · 강준모
- **지도교수:** 이예솝

![표지](docs/slides/01.jpg)

## 한눈에 보기

| 영역 | 방법 | 결과 |
|---|---|---|
| 차선 | LAB 색공간에서 초록 경계 마스크 → 띠 3줄로 도로 중앙 계산 → 상태 판정 → P 제어 | 시뮬레이터 60초 주행, 도로 위 100% |
| 표지판 | YOLOv8n 10종 → TFLite INT8 | mAP50 0.971 (INT8 0.949), 크기 3.25 MB |
| 주행 | 상태 머신 + 표지판 동작(정지 · 신호 · 서행 · 회전 · 도착) | 교차로 좌 · 우 · 직진 3/3, 테스트 15/15 |

## 발표 자료

### 01 미션
트랙을 완주하면서 표지판과 신호를 지켜야 합니다.

![미션](docs/slides/02.jpg)

### 02 기기
라즈베리파이 5, 카메라, 좌우 바퀴 차동 구동(TB6612), 18650 배터리 2개로 구성했습니다. 카메라 입력부터 판단, 모터 제어까지 한 보드에서 처리합니다.

![기기](docs/slides/03.jpg)
![구조](docs/slides/04.jpg)

### 03 전처리
- **차선:** 카메라 영상에서 초록 경계만 남기고, 가로 띠 3줄에서 양쪽 경계를 찾아 도로 중앙을 계산합니다.
- **표지판 데이터:** 학습 전에 데이터 문제 네 가지(중복, 라벨 누락, 연속 프레임 섞임, right 부족)를 고쳤습니다.

![차선 전처리](docs/slides/05.jpg)
![데이터 문제](docs/slides/06.jpg)
![데이터셋](docs/slides/07.jpg)
![수집](docs/slides/08.jpg)
![라벨링](docs/slides/09.jpg)
![학습](docs/slides/10.jpg)

### 04 알고리즘
차선 상태를 판정하고, 표지판을 보면 정해진 동작을 합니다. 교차로 회전은 직진 → 회전 → 차선 다시 잡기 순서로, 시간을 기준으로 합니다.

![차선 상태](docs/slides/11.jpg)
![표지판 동작](docs/slides/12.jpg)
![교차로 회전](docs/slides/13.jpg)

### 05 성능

![표지판 인식 성능](docs/slides/14.jpg)
![v2 재학습](docs/slides/15.jpg)
![INT8 변환](docs/slides/16.jpg)
![클래스별 성능](docs/slides/17.jpg)
![시뮬레이터](docs/slides/18.jpg)
![주행 검증](docs/slides/19.jpg)
![정리](docs/slides/20.jpg)

## 실행

Pi 5 (64-bit Raspberry Pi OS), 레포 루트에서:

```bash
pip install ultralytics ai-edge-litert simple-pid gpiozero opencv-python pyyaml
python3 -m drive.main --mode real --signs weights/sign_v2_int8.tflite
```

노트북에서 주행 로직 확인:

```bash
python -m drive.sim --start outer
python -m pytest tests -q
```

## 구조

```
drive/     주행 코드 (Pi 실행: python3 -m drive.main, 시뮬레이터: python -m drive.sim)
sign/      표지판 YOLO 데이터 정리 · 학습 · INT8 변환
lane/      차선 데이터 점검 (차선 CNN 모델은 다음 단계)
deploy/    차선 데이터 수집 코드
weights/   표지판 모델 (sign_v2_int8.tflite 등)
data/      학습 데이터
docs/      설계 · 결과 문서, 발표 슬라이드 이미지
tests/     컨트롤러 · 표지판 필터 테스트
```

## 문서
- 다른 PC 에서 이어받기: [HANDOFF.md](HANDOFF.md)
- 표지판 v2 학습 · INT8 변환 결과: [docs/sign-v2-int8.md](docs/sign-v2-int8.md)
- 주행 코드에 더한 것: [docs/drive-changes.md](docs/drive-changes.md)
- 데이터 점검 결과: [docs/data-report.md](docs/data-report.md)
- 학습 방향 설계: [docs/training-plan.md](docs/training-plan.md)
- 차선 데이터 수집 방법: [docs/collection-guide.md](docs/collection-guide.md)
