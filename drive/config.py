"""주행 코드의 모든 설정 상수.

좌표와 픽셀 값은 모두 처리 해상도(PROC_SIZE, 기본 320x240) 기준이다.
"""

# ---------------------------------------------------------------- 카메라 / 전처리
CAMERA_SIZE = (640, 480)      # 기준 해상도. 표지판 박스 기준값(SIGN_*)과 시뮬레이터 카메라가 이 단위다
# 실제로 찍는 해상도. 320x240 이면 카메라 읽기·뒤집기·YOLO 전처리 연산이 1/4 로 줄어든다.
# 차선 비전은 원래 320x240 으로 줄여 쓰고, YOLO 도 320 입력이라 정확도 차이는 없다. 640 모델을 쓸 때는 (640, 480).
CAMERA_CAPTURE_SIZE = (320, 240)
PROC_SIZE = (320, 240)        # 리사이즈 후 처리 해상도 (width, height)
FLIP_CODE = -1                # 카메라가 거꾸로 달려 있어 상하좌우 반전
BOTTOM_EXCLUDE = 20           # 화면 맨 아래 차체가 찍히는 줄 수, 계산에서 제외
FRAME_TIMEOUT_S = 0.3         # 이 시간 동안 새 프레임이 없으면 정지

# ---------------------------------------------------------------- 녹색 마스크 (LAB a*)
# a* < A_THR 이면 녹색. run_20261028_190928 측정: 녹색 55~75, 도로/바닥/횡단보도 105~130
# (도로가 113까지 내려가는 프레임이 있어 115는 너무 가깝다). 빈 구간 80~100의 가운데.
A_THR = 95
OPEN_KERNEL = 5               # 열기: 작은 녹색 노이즈 제거 (0 또는 1이면 생략)
CLOSE_KERNEL = 15             # 닫기: 녹색 위 파란 원 같은 구멍 메우기 (0 또는 1이면 생략)
DARK_MEAN_THR = 20            # 회색조 평균이 이보다 어두우면 무효 프레임 (카메라 가림)

# ---------------------------------------------------------------- 스캔 띠
# y: 띠 윗줄, h: 높이, weight: 도로 중앙 가중 평균에 쓰는 가중치 (가까울수록 크게)
# y + h 는 PROC_SIZE[1] - BOTTOM_EXCLUDE(=220) 이하여야 한다.
BANDS = [
    {"name": "near", "y": 195, "h": 12, "weight": 0.5},
    {"name": "mid",  "y": 168, "h": 12, "weight": 0.3},
    {"name": "far",  "y": 141, "h": 12, "weight": 0.2},
]
COLUMN_GREEN_RATIO = 0.5      # 띠 안에서 한 열의 녹색 비율이 이 이상이면 그 열은 녹색
MIN_RUN = 8                   # 녹색이 연속 MIN_RUN px 이상이어야 경계로 인정

# 띠별 도로 폭 초기값(px, BANDS 순서). None이면 첫 양쪽 검출에서 학습한다.
# run_20261028_190928 직선 구간(260~290번)에서 잰 값. 띠 y를 바꾸면 tune.py의 w 키 출력으로 다시 잰다.
# 값이 없으면 커브에서 출발할 때 가까운 띠가 한쪽 경계만 보여 한참 동안 쓸 수 없다.
ROAD_WIDTH_INIT = [264, 206, 143]
WIDTH_EMA_ALPHA = 0.1         # 도로 폭 이동평균 갱신 비율
MIN_ROAD_WIDTH_PX = 30        # 이보다 좁으면 도로로 보지 않음 (학습 전 첫 측정 보호)
WIDTH_RATIO = 1.6             # 폭이 평소의 이 배 이상이면 넓어짐 (가까운 띠: 교차로)
WIDTH_MIN_RATIO = 0.5         # 폭이 평소의 이 배 미만이면 잘못 잡힌 것으로 보고 무효

# ---------------------------------------------------------------- 상태 판정
STATE_CONFIRM_FRAMES = 3      # 같은 후보 상태가 N프레임 연속일 때만 상태 변경
LOST_FRAMES = 5               # 유효한 띠가 없는 프레임이 이만큼 연속이면 LOST
OFFROAD_HALF_W = 12           # 화면 중앙 아래(가까운 띠) 탐색 시작점 검사 폭의 절반
OFFROAD_GREEN_RATIO = 0.6     # 그 영역의 녹색 비율이 이 이상이면 OFF_ROAD

# ---------------------------------------------------------------- 제어
LOOP_HZ = 20                  # 주행 루프 주기 고정 (PID D항이 주기에 영향을 받음)
PID_P = 1.5                   # 1.3 에서 올림: 실차 커브 출구에서 바깥쪽에 붙었다. 좌우로 흔들리면 낮춘다
PID_I = 0.0
PID_D = 0.0
# 차를 도로 한가운데에 똑바로 놓고 python3 -m drive.main --mode dry 화면의 err 값을 읽어 넣는다 (예: err +0.12 면 0.12).
# 실차는 직선에서도 한쪽 선에 붙어 갔다. 카메라가 비껴 달린 만큼을 빼 준다
LANE_CENTER_OFFSET = 0.0
# 차선 안에서 설 자리 (+ 오른쪽, - 왼쪽, 오차 단위). sim 에서 0.04 가 약 1.5cm. 차 폭 15cm, 도로 20cm 라 양쪽 여유 2.5cm
LANE_OFFSET = 0.025             # 평소: 살짝 오른쪽, 약 1cm (표지판이 오른쪽 길가에 있다. 왼쪽으로 치우치면 표지판·교차로를 놓쳤다)
TURN_PREP_OFFSET = 0.10         # 좌/우회전이 정해지면 그쪽으로 붙는다 (sim: 좌회전 예약 시 왼쪽 약 1cm)
# 가운데 찾기: 곧은 길에서 차가 가운데를 벗어나면(추정 치우침 > START) 천천히 가며 치우침으로 조향해 먼저 가운데로 돌아온다.
# 실차 속도 sim 에서는 켜도 가운데로 오는 거리가 같고(20~40cm) 바퀴가 선을 넘는 프레임만 늘어 기본은 끈다. 실차에서 시험용
CENTER_FIRST = False
CENTER_START_ERROR = 0.15       # 가까운 띠 기준 약 2cm
CENTER_DONE_ERROR = 0.06
CENTER_GAIN = 1.5
CENTER_EXTRAP = 1.0            # 가까운·중간 띠 도로 중앙을 이은 선을 차 쪽으로 늘리는 비율 (차 위치 추정)
CENTER_K = 1.0                 # 차 위치의 치우침을 조향에 더하는 배율 (항상). 내다본 오차만 쓰면 커브에서 안쪽으로 파고들거나
                               # 바깥 테두리로 밀려 표지판·교차로를 놓쳤다. 실차 속도 sim 바깥 한 바퀴: 바퀴가 도로 안 63 -> 70%
CENTER_THROTTLE = 0.28          # MIN_THROTTLE. 이보다 낮으면 바퀴가 멈춘다
BASE_THROTTLE = 0.3            # 0.35 는 출발 때 Pi 가 꺼졌다. 커브는 SLOW_K 로 바깥 바퀴를 더 밀어 보완
SLOW_K = -0.4                 # 조향 비례 속도: BASE * (1 - SLOW_K * |steering|). 음수면 커브에서 바깥 바퀴가 빨라져 더 빨리 돈다
# 모터가 실제로 도는 최소 PWM. 바퀴를 띄우고 `python3 -m drive.motors --sweep`으로
# 실측해서 넣는다. 배터리 전압에 따라 달라진다.
MIN_THROTTLE = 0.28           # 0.25 는 실차에서 교차로 감속 중 바퀴가 멈췄다 (손으로 밀어 주면 다시 감)
MAX_THROTTLE = 0.5
INTERSECTION_THROTTLE = 0.32  # 교차로 감속 직진 속도. 0.28 은 실차에서 멈춰 섰다
INTERSECTION_MAX_FRAMES = 120  # 감속 직진 최대 프레임 수(6초), 넘으면 정지. 속도를 낮춰 4초에서 늘림

# ---------------------------------------------------------------- 모터 / 부저 (gpiozero BCM 번호)
PWMA, AIN1, AIN2 = 18, 22, 27   # A = 왼쪽 바퀴
PWMB, BIN1, BIN2 = 23, 25, 24   # B = 오른쪽 바퀴
PWM_FREQUENCY = 100             # gpiozero PWMOutputDevice 기본값
TURN_MODE = "pivot"             # "keep_inner": 조향이 PIVOT_STEER 보다 작으면 안쪽 바퀴를 INNER_MIN_THROTTLE 아래로 안 내림 (sim 에서 차이 없음, 실차 시험용)
INNER_MIN_THROTTLE = 0.27
PIVOT_STEER = 0.8
# TURN_MODE: 'pivot' 안쪽 바퀴 감속 / 'spin' 안쪽 바퀴 역회전까지
# 오른쪽 바퀴 PWM 배율. 바닥에 놓고 감속 직진(INTERSECTION_THROTTLE)을 2m 시켜 봤을 때
# 오른쪽으로 휘면 올리고 왼쪽으로 휘면 내린다 (데드밴드 때문에 1/세기비 보다 작은 값이 맞다).
# sim: 좌우 차 8% 안은 보정 없이 교차로 통과, 15% 차는 1.03~1.09 로 보정하면 통과.
RIGHT_WHEEL_TRIM = 1.0
MOTOR_WATCHDOG_S = 0.5          # 이 시간 동안 모터 명령이 없으면 스스로 정지
# 바퀴 세기를 올릴 때 1초에 이만큼까지만 올린다 (내릴 때는 바로). 출발 순간 모터가 전류를 한꺼번에 끌어
# 배터리 전압이 떨어지고 Pi 가 꺼지는 것을 줄인다. 0 이면 끈다. 1.5 면 BASE_THROTTLE 0.3 까지 약 0.2초.
MOTOR_RAMP_PER_S = 1.0          # 출발 전류를 더 줄이려고 1.5 에서 낮춤 (0.3 까지 약 0.3초)
BUZZER_PIN = 12
HORN_TONE_HZ = 440              # TonalBuzzer 기본 음역(220~880Hz) 안의 값
HORN_FRAMES = 20                # 경적 표지판 하나에 울리는 프레임 수 (1초). 0.3초는 실차에서 들리지 않았다

# ---------------------------------------------------------------- 표지판 (YOLO)
# 검출 클래스 이름 -> (이벤트 종류, 값). 클래스 이름은 sign/ 의 YOLO 데이터셋(10 클래스)과 같다.
# crosswalk 는 이벤트를 만들지 않는다 (신호등과 함께 보이므로 신호등 이벤트로 충분).
SIGN_CLASS_EVENTS = {
    "left": ("TURN", "LEFT"),
    "straight": ("TURN", "STRAIGHT"),
    "right": ("TURN", "RIGHT"),
    "horn": ("HORN", None),
    "speed_20": ("SPEED", 0.6),
    "stop": ("STOP", None),
    "red_light": ("LIGHT", "RED"),
    "green_light": ("LIGHT", "GREEN"),
    "red_line": ("FINISH", None),
}
# 박스 기준은 검출기 입력(바로 세운 640x480) 픽셀. 실제 트랙에서 반드시 다시 잰다.
SIGN_MIN_CONF = 0.5
SIGN_MIN_BOX_H = 40             # 박스가 이 높이(px) 이상일 때만 (충분히 가까울 때만) 반응
SIGN_CLASS_RULES = {            # 클래스별로 다르게 둘 값만 적는다 (min_conf, min_h, min_y2)
    "red_line": {"min_h": 0, "min_y2": 380},    # 납작한 선이라 높이 대신 화면 아래쪽까지 왔는지로 판단
    "red_light": {"min_h": 60},
    "horn": {"min_h": 25},          # 반대편 섬에 있는 경적 표지판은 40px 까지 커지기 전에 화면 밖으로 나갔다
    "green_light": {"min_h": 60},
}
SIGN_CONFIRM_HITS = 2           # 같은 클래스가 SIGN_CONFIRM_WINDOW 프레임 안에서 이만큼 검출돼야 이벤트
SIGN_CONFIRM_WINDOW = 15
HORN_CONFUSE_FRAMES = 30        # 경적이 보였던 자리에서 이 프레임 안에 보인 비슷한 크기의 회전 표지판은 경적으로 본다
SIGN_EDGE_MARGIN = 4            # 회전 표지판 상자가 화면 좌우 끝에서 이 px(640 기준) 안에 닿으면 잘린 것으로 보고 버린다
SIGN_TURN_MIN_ASPECT = 0.7      # left/right 상자의 가로/세로가 이보다 작으면(비스듬히 봄) 버린다. 학습 데이터 좌·우 표지판은 95% 가 0.73 이상
SIGN_COOLDOWN_FRAMES = 40       # 같은 클래스 이벤트를 다시 받기까지 기다릴 프레임 수
SIGN_IMGSZ = 320                # YOLO 입력 크기. Pi 는 weights/sign_v3_26n_320_int8.tflite (320). 640 모델이면 640 과 CAMERA_CAPTURE_SIZE (640, 480)
# YOLO 를 돌리는 최소 간격(초). 매 프레임 돌리면 CPU 4코어가 꽉 차 전력이 모자랄 수 있다.
# 0.25 = 1초에 최대 4번. SIGN_CONFIRM_WINDOW(15프레임=0.75초) 안에 2번 잡히려면 0.35 이하로 둔다.
SIGN_DETECT_INTERVAL_S = 0.35   # 전력 절약용으로 0.25 에서 올림. 이보다 크면 정지선 등을 놓친다
# YOLO 스레드가 쓸 CPU 코어. None 이면 제한하지 않는다. 주행 루프는 나머지 코어도 쓸 수 있다.
SIGN_CPU_CORES = (0,)           # 전력 절약용으로 1코어만. 검출이 0.35초 안에 안 끝나면 (0, 1) 로 되돌린다
# True: YOLO 를 별도 프로세스에서 돌린다 (주행 루프가 느려지지 않음). False: 예전처럼 스레드.
SIGN_USE_PROCESS = True
# 회전 중과 출발 직후 SIGN_PAUSE_START_FRAMES 동안 YOLO 를 쉰다 (모터 전류가 클 때 CPU 부하를 겹치지 않게)
SIGN_PAUSE_IN_TURN = True
SIGN_PAUSE_START_FRAMES = 0      # 출발 직후 쉴 프레임. 출발점에 신호등이 있으면 빨간불을 놓칠 수 있어 기본은 0

# ---------------------------------------------------------------- 표지판 동작 (LOOP_HZ 프레임 단위)
SPEED_FRAMES = 120              # 서행 표지판 후 감속 유지 (6초)
STOP_WAIT_FRAMES = 60           # 정지 표지판 앞 정지 (3초)
# 정지 대기 중과, 대기를 마치고 출발한 뒤 이 프레임 동안은 정지 표지판을 무시한다 (5초).
# 없으면 차 앞에 그대로 보이는 같은 표지판을 쿨다운(40) 뒤 다시 보고 영원히 선다 (sim 에서 재현).
STOP_IGNORE_FRAMES = 100
LIGHT_MAX_WAIT_FRAMES = 400     # 빨간불 최대 대기 (20초). 초록불을 못 보더라도 멈춰 있지 않게
FINISH_ARM_FRAMES = 200         # 출발 후 이 프레임 전에는 정지선을 무시 (출발 지점의 선)
FINISH_DELAY_FRAMES = 6         # 정지선을 보고 이만큼 더 간 뒤 정지 (선이 차 앞에 올 때까지)
# 교차로 회전: ENTRY 직진 -> SPIN 회전 -> EXIT 차선 다시 잡기. sim 으로 먼저 맞추고 실차에서 다시 맞춘다.
TURN_ENTRY_FRAMES = 40          # 실차 속도(약 5cm/s, 17Hz)를 넣은 sim 에서 맞춘 값. 18 은 빠른 가상 차 기준이라 실차에서 너무 일찍 돌았다
TURN_FRAMES = 25               # 최소 회전 프레임 (실차 약 35도/s, 17Hz 에서 약 50도). 12 는 25도라 대각선을 새 길로 보고 끝낼 수 있었다
TURN_DONE_CONFIRM = 3          # 새 길이 정면에 보이는 것이 이만큼 연속이어야 회전을 끝낸다
TURN_STEERING = 1.0             # pivot: 1.0 이면 안쪽 바퀴 정지
TURN_THROTTLE = 0.32          # 0.4 는 회전 순간 바깥 바퀴 전류가 커서 Pi 가 꺼졌다
TURN_STUCK_FRAMES = 8           # 회전 중 화면이 이만큼 그대로면 차가 안 도는 것 (바퀴를 끌다 멈춤)
TURN_KICK_THROTTLE = 0.42       # 그때 잠깐 미는 세기. 0.4 를 계속 쓰면 Pi 가 꺼졌으므로 짧게만
TURN_KICK_FRAMES = 6            # 약 0.35 초
TURN_EXIT_MAX_FRAMES = 30       # 회전 후 이만큼 지나도 차선을 못 잡으면 정지
TURN_SPIN_MAX_FRAMES = 60       # TURN_FRAMES 뒤에도 새 길이 정면에 안 보이면 이만큼까지 계속 돈다 (3초)
TURN_SPIN_HARD_MAX_FRAMES = 200 # 새 길이 정면 근처(오차 < TURN_EXIT_ERROR, 앞쪽 띠 LR)에 오기 전에는 직진으로 넘어가지 않고 이만큼까지 돈다
TURN_EXIT_ERROR = 0.5
TURN_BLOCKED_ENTRY_FRAMES = 3   # 직진하다 앞이 막힌 T자에서 받아 둔 방향으로 돌 때 더 들어갈 프레임 (이미 교차로 끝까지 왔다)
# 막힌 T자 끝에서 빠져나가는 회전: 교차로를 알아볼 때 먼 띠 위 초록 비율(ahead_green)이 TEND_AHEAD_GREEN 이상이면,
# 정해진 진입 프레임 대신 먼 띠가 테두리(GG)에 닿은 뒤 TEND_EXTRA_FRAMES 더 가서 돈다 (최대 TURN_ENTRY_TEND_MAX)
TEND_AHEAD_GREEN = 0.75
AHEAD_PROBE_H = 30              # 먼 띠 위 몇 줄까지 보는지 (320x240 기준)
TEND_EXTRA_FRAMES = 4
TURN_ENTRY_TEND_MAX = 90
TURN_BOOST_AFTER = 35           # 회전이 이 프레임보다 오래 걸리면 세기를 올린다 (실차: 0.32 로 3초에 30도밖에 못 돎)
TURN_BOOST_PER_FRAME = 0.005    # 프레임마다 올리는 양. 0.32 -> 0.42 까지 약 20 프레임
TURN_THROTTLE_MAX = 0.42        # 실차 로그: 0.42 로 3초 돌려도 Pi 가 꺼지지 않았다
LOST_HOLD_FRAMES = 8            # 커브를 꺾던 중(조향 >= LOST_HOLD_MIN_STEER) 도로가 안 보이면 직전 조향을 이만큼 유지 (약 0.5초)
LOST_HOLD_MIN_STEER = 0.3
TURN_DONE_ERROR = 0.35          # 회전 끝 판정: 양쪽 경계가 보이고(LANE) 도로 중앙 오차가 이보다 작을 때
# T자·옆길 교차로: 한쪽 경계는 계속 보이고 회전할 쪽만 열리는 곳 (비전은 ONE_SIDE 로 본다).
# 회전 표지판을 본 뒤 SIDE_TURN_WINDOW_FRAMES 안에, 가까운 SIDE_OPEN_BANDS 개 띠에서 회전할 쪽 경계가
# SIDE_OPEN_FRAMES 연속 사라지면 회전을 시작한다. 0 이면 끈다. 바깥 커브도 한쪽이 열려 보이므로 창을 짧게 둔다.
PENDING_STALE_FRAMES = 200      # 정한 방향의 표지판을 이만큼(약 12초) 다시 못 봤고 교차로도 없었으면, 받아 둔 다른 방향으로 바꾼다
SIDE_TURN_WINDOW_FRAMES = 150   # 표지판을 마지막으로 본 뒤 약 9초 (실차가 느려서 100 은 옆길 전에 끝났다)
SIDE_OPEN_BANDS = 2
SIDE_OPEN_FRAMES = 10            # 3 이면 왼쪽 선에 붙어 가다 잠깐 틀어져 오른쪽 경계가 화면 밖으로 나간 것도 옆길로 볼 수 있다
SIDE_CLOSED_FRAMES = 5          # 열리기 전에 양쪽 경계가 이만큼 보여야 한다 (교차로를 막 나올 때 오작동 방지)
SIDE_AFTER_CROSS_FRAMES = 60    # 교차로·회전 뒤 이만큼 지났으면 양쪽 경계 기록 없이도 열린 쪽을 옆길로 본다
SIDE_OPEN_MARGIN = (10, 30, 30) # 띠별: 없는 경계가 있어야 할 자리가 화면 안쪽으로 이만큼(px, 320 기준) 들어와야 진짜 열린 길
SIDE_OPEN_MAX_AGE = 40          # 표지판을 보기 전에 이미 열려 있던 옆길도 인정하는 기간 (2초). 더 오래 열려 있었으면 무시
SIDE_TURN_ENTRY_FRAMES = 60     # 옆길 회전을 정한 뒤 직진할 프레임. 실차(IMG_2397 28~33초): 53 에서 안쪽 섬 모서리를 2~3cm 밟음 = 약 7~10 프레임 일찍. sim 에서도 60 이 바퀴가 선을 가장 적게 넘음
SIDE_SKIP_EXTRA_FRAMES = 0      # 먼 띠로 옆길을 알아봤을 때 띠 하나마다 더 직진할 프레임. sim 에서는 늘리면 성공률이 떨어져(20: 4/6, 35: 0/6) 0. 실차에서 너무 일찍 돌면 10 씩 올려 본다
SIDE_MIN_ENTRY_FRAMES = 5       # 표지판보다 옆길이 먼저 열려 있었을 때 줄일 수 있는 진입 프레임의 최소값
CROSS_LATE_SIGN_FRAMES = 40     # 정한 방향 없이 교차로에 들어선 지 이 프레임 안에 회전 표지판이 확정되면 그 방향으로 돈다

# ---------------------------------------------------------------- 화면 / 기록
DISPLAY_SCALE = 2               # 오버레이 창 확대 배율
DISPLAY_EVERY = 1               # N프레임마다 한 번만 화면 갱신 (SSH X11 등 느린 화면용)
LOG_DIR = "logs"
SAVE_SIGN_SHOTS = True          # 회전 표지판 이벤트마다 그 순간 화면(검출 상자 표시)을 logs/sign_shots/ 에 저장

# ---------------------------------------------------------------- 키 (cv2.waitKey 값)
# 아래 방향키는 Pi(GTK)에서 84. Windows에서는 waitKey가 방향키를 주지 않으므로 space를 쓴다.
KEY_TOGGLE_RUN = (ord("s"),)
KEY_ESTOP = (84, ord(" "))
KEY_QUIT = (ord("q"),)
KEY_TOGGLE_MASK = (ord("m"),)
KEY_PLAYBACK_PAUSE = (ord("p"),)  # offline: 재생 멈춤/재개
KEY_PLAYBACK_STEP = (ord("n"),)   # offline: 멈춘 상태에서 한 프레임 진행
