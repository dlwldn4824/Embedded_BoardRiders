# 팀 주행 코드(drive/)에 더한 것

원본은 첫 커밋 `Import team rule-based drive code as of 2026-11-02 (unchanged)` 에 그대로 있다. `git diff 86941b5 -- drive/` 로 바뀐 부분만 볼 수 있다.

## 1. 표지판 YOLO 연결 (`signs.py`)
- `YoloSignDetector`: ultralytics 로 `.pt / .onnx / .tflite` 를 읽는다.
  ```bash
  python3 -m drive.main --mode real --signs runs/sign/baseline/weights/best.pt
  ```
  `--signs` 가 없으면 이전처럼 표지판을 보지 않는다.
- `SignInterpreter` 에 추가한 필터:
  - **연속 검출:** 같은 클래스가 15프레임 안에 2번 검출돼야 이벤트가 난다. 한 프레임짜리 오검출을 거른다.
  - **클래스별 기준:** `SIGN_CLASS_RULES` 로 클래스마다 다른 기준을 둔다. 정지선은 박스 높이 대신 화면 아래쪽(y2 ≥ 380)까지 왔는지로 판단한다.
- 클래스 이름은 YOLO 데이터셋 10 클래스에 맞췄다. 기존 `slow` 는 `speed_20` 으로 바꿨다.

## 2. 표지판 동작 (`controller.py`)
| 이벤트 | 동작 | 설정 |
|---|---|---|
| left / right | 다음 교차로에서 회전: ENTRY 직진 → SPIN 한쪽 바퀴 회전 → EXIT 차선 다시 잡기 | `TURN_ENTRY_FRAMES=40`, `TURN_FRAMES=20` |
| straight / 표지판 없음 | 교차로 감속 직진 (기존과 같음, 최대 프레임만 20 → 80) | `INTERSECTION_MAX_FRAMES` |
| stop | 3초 정지 후 출발 | `STOP_WAIT_FRAMES` |
| red_light | 멀리서 보면 기억만 하고 계속 간다. 차 바로 앞 횡단보도(crosswalk)가 같이 보이면 그 앞에서 서서, 초록불을 보거나 빨간불이 꺼질 때까지 기다린다 (빨간불을 마지막으로 본 뒤 최대 20초). 횡단보도를 못 잡으면 교차로 입구에서 선다 | `SIGN_CLASS_RULES["crosswalk"]`, `LIGHT_OFF_RUNS`, `LIGHT_MAX_WAIT_FRAMES` |
| crosswalk | 상자 아래 끝이 화면 아래쪽(y2 ≥ 350)이고 화면 가운데를 덮을 때만 이벤트. 빨간불일 때만 정지에 쓴다 | `SIGN_CLASS_RULES["crosswalk"]` |
| speed_20 | 6초 동안 속도 0.6배 (기존은 끝까지 유지) | `SPEED_FRAMES` |
| red_line | 출발 10초 뒤부터 인식, 조금 더 가서 정지 (도착) | `FINISH_ARM_FRAMES`, `FINISH_DELAY_FRAMES` |
| horn | 기존과 같음 | |

## 3. 교차로 판정 수정 (`vision.py`)
- **고친 문제:** 교차로에 들어가면 먼 띠부터 경계가 사라지고, 결국 세 띠 모두 경계가 없어진다. 원래 코드는 이것을 `LOST`(도로 잃음)로 보고 멈췄다.
- **수정:** 세 띠를 모두 정상적으로 훑었는데 양쪽 경계가 하나도 없으면 `INTERSECTION` 으로 판정한다.
- sim 결과:
  - `--start offset` 이 0.40m 에서 멈추던 것이 60초 동안 9.63m 를 주행한다.
  - 바깥 순환로(`outer`, `corner`)는 수정 전과 같다.

## 4. 기타
- `main.py` CSV 로그 첫 열에 `fid` 를 추가했다. `--record` 이미지 파일 이름(`{fid:06d}.jpg`)과 같은 번호다.
  → 규칙 기반 주행 기록을 CNN 학습 라벨(이미지 ↔ steering)로 바로 쓸 수 있다.
- `motors.py` 에 `RIGHT_WHEEL_TRIM` 을 추가했다. 교차로 동작은 시간 기반이라, 차가 한쪽으로 휘면 실패하기 때문이다.
- `sim.py --turn LEFT|RIGHT|STRAIGHT`: 방향 표지판을 이미 본 상태로 출발한다.

## 검증
- `python -m pytest tests`: 컨트롤러와 표지판 필터 테스트 15개가 통과한다.
- sim `--start cross`:
  - 좌회전, 우회전, 직진 모두 교차로를 통과하고 차선을 다시 잡는다 (`docs/data_report/sim_turns.jpg`).
  - 모터 좌우 차가 ±8% 안이면 보정 없이 통과한다. 15% 차이는 `RIGHT_WHEEL_TRIM` 으로 보정해야 한다.
- YOLO 검출기 → 이벤트: val 이미지 묶음(정지, 빨간불, 초록불, 좌회전)에서 묶음마다 이벤트가 한 번씩 났다. 검출 시간은 Mac 에서 장당 약 150ms 다.

## 실차에서 다시 맞춰야 할 값 (sim 값은 출발점일 뿐)
- `TURN_ENTRY_FRAMES`, `TURN_FRAMES`: 실제 교차로 크기와 바퀴 속도에 따라 달라진다.
- `RIGHT_WHEEL_TRIM`: 감속 직진을 2m 시켜 보고 맞춘다.
- `SIGN_MIN_BOX_H`, `SIGN_CLASS_RULES`: 표지판 앞 몇 cm 에서 반응할지에 따라 정한다.
- 정지선 `min_y2`.
- Pi 에서 `.pt` 는 느리다. INT8 `.tflite` 로 바꾼 뒤 지연 시간을 잰다.

## 주의
**YOLO 학습 이미지와 차선 주행 사진(run_20261028_*)은 색감이 다르다.**
- 주행 코드의 초록 기준값(`A_THR=95`)은 어두운 도로 사진으로 맞춘 값이다.
- YOLO 이미지(밝은 회색 도로)에 그대로 적용하면 대부분 교차로로 판정된다.
- 실제 주행할 카메라 설정이 둘 중 어느 쪽인지 확인하고, 그 설정으로 YOLO 데이터를 보강해야 한다.
