# 차선 데이터 수집 가이드

기존 데이터의 문제를 고치기 위한 수집 방법 ([data-report.md](data-report.md) 참고).
- 조향이 45/90 두 값뿐이었음 → 게임패드 스틱으로 **연속 조향값**을 기록한다.
- 우회전이 없었음 → **시계 방향(cw) 주행**도 수집한다.
- 복구 장면이 없었음 → 복구 태그를 붙여서 따로 수집한다.
- YOLO 데이터와 색감이 달랐음 → **카메라 노출과 화이트밸런스를 고정**한다.

## 0. 처음 한 번 하는 설정

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-yaml python3-evdev python3-gpiozero python3-lgpio
```

1. `deploy/config.yaml` 은 팀 주행 코드(`drive/config.py`)와 같은 값으로 맞춰 두었다.
   - 차동 구동: 왼쪽 바퀴 PWMA 18 / AIN1 22 / AIN2 27, 오른쪽 바퀴 PWMB 23 / BIN1 25 / BIN2 24.
   - 회전 방식은 pivot (안쪽 바퀴 감속). 카메라는 180° 회전(`hflip`, `vflip`).
   - 조향 라벨 `steer` 와 함께 실제 바퀴 값 `left`, `right` 도 저장한다.
2. 바퀴를 띄운 상태에서 `--input keyboard` 로 확인한다.
   - a 를 누르면 **왼쪽 바퀴가 느려져야** 한다. 반대로 움직이면 핀 배선을 확인한다.
   - 저장된 이미지가 바로 서 있는지 확인한다.
3. 게임패드 버튼 이름을 확인한다.
   ```bash
   python3 deploy/collect.py --test-input
   ```
   눌러서 나온 이름으로 `collect.gamepad` 를 고친다.
4. **카메라 고정값 정하기:** 첫 run 을 수집한 뒤 `data/runs/<run>/meta.json` 의 `camera_locked` 값을 `config.yaml` 의 `exposure_time / analogue_gain / colour_gains` 에 적는다.
   이후 모든 차선 수집과 **YOLO 표지판 촬영도 이 값으로** 한다.

## 1. 조작

| 동작 | 게임패드 (기본) | 키보드 (`--input keyboard`) |
|---|---|---|
| 조향 | 왼쪽 스틱 좌우 | a/d 또는 ←/→ (누르는 동안 커지고, 떼면 0으로) |
| 주행 ON/OFF (cruise 속도) | RB | g |
| 녹화 시작/정지 | A | r |
| 복구 태그 ON/OFF | X | c |
| 방금 실수 (직전 2초 무효) | Y | x |
| 비상 정지 (주행·녹화 OFF) | B | space |
| 속도 ± | 십자키 위/아래 | + / - |
| 종료 | START | q |

- 주행이 OFF 이면 녹화 중이어도 저장하지 않는다 (정지 화면이 쌓이지 않게).
- 게임패드 연결이 끊기면 바로 정지한다.
- 키보드는 키 반복 속도에 따라 조향이 계단식이 될 수 있다. **본 수집은 게임패드로 할 것.**

## 2. 수집 계획 (목표 약 15,000장 = 15fps × 약 17분)

| 구분 | 방향 | 분량 | 방법 |
|---|---|---|---|
| 정상 주행 | ccw | 5분 | 차선 중앙, 일정 속도 |
| 정상 주행 | cw | 5분 | 우회전 확보 |
| 복구 주행 | ccw + cw | 3분 | 녹화를 끄고 차를 차선 가장자리/비뚤어진 자세로 둔다 → 복구 태그 ON → 녹화 ON → 중앙으로 돌아오면 녹화 OFF. **틀어지는 과정은 녹화하지 않는다** |
| 조명 변화 | 둘 다 | 각 1~2분 | 창가 쪽 블라인드 열고/닫고, 일부 조명 끄고 |
| 속도 변화 | 둘 다 | 2분 | cruise ±0.05 |

- run 하나는 2~3분 정도로 짧게 끊는다. 나중에 **run 단위로 train/val/test 를 나눈다.**
- `--note` 에 조명, 배터리 상태, 트랙 배치를 적는다.
  ```bash
  python3 deploy/collect.py --direction cw --note "창가 블라인드 열림, 배터리 80%"
  ```
- 커브를 할 때 **부드럽게 연속으로** 꺾는다. 스틱을 끝까지 쳤다가 놓는 식이면 기존 데이터와 같은 문제가 생긴다.
- 실수하면 바로 Y 를 누른다. 학습할 때 `bad.csv` 구간을 뺀다.

## 3. 수집 후 확인

```bash
python3 lane/inspect_data.py data/runs
```

다음을 확인한다.
- 조향 분포에서 좌와 우가 둘 다 있는지.
- 실제 fps가 설정값(15)에 가까운지. 낮으면 SD 카드 속도나 `jpeg_quality` 를 확인.
- `camera_locked` 값이 run 마다 같은지.

## 4. 학습할 때 처리할 것 (lane 학습 코드에서)

- `bad.csv` 구간 제외, 주행 시작 직후 1초 제외.
- 사람의 반응 지연: 조향 라벨을 `t + 0.1~0.2s` 시점 값으로 당겨서 실험.
- 직진 과다: `|steer| < 0.1` 다운샘플.
- 좌우 반전 증강 시 steer 부호 반전 (ccw/cw 균형이 맞으면 필요성은 줄어든다).
