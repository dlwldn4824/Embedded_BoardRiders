"""주행 코드의 모든 설정 상수.

좌표와 픽셀 값은 모두 처리 해상도(PROC_SIZE, 기본 320x240) 기준이다.
"""

# ---------------------------------------------------------------- 카메라 / 전처리
CAMERA_SIZE = (640, 480)      # MyPiCamera(width, height)
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
PID_P = 1.0
PID_I = 0.0
PID_D = 0.0
BASE_THROTTLE = 0.35
SLOW_K = 0.5                  # 조향이 클수록 감속: BASE * (1 - SLOW_K * |steering|)
# 모터가 실제로 도는 최소 PWM. 바퀴를 띄우고 `python3 -m drive.motors --sweep`으로
# 실측해서 넣는다. 배터리 전압에 따라 달라진다.
MIN_THROTTLE = 0.25
MAX_THROTTLE = 0.5
INTERSECTION_THROTTLE = 0.28  # 교차로 감속 직진 속도
INTERSECTION_MAX_FRAMES = 20  # 감속 직진 최대 프레임 수, 넘으면 정지 (YOLO 연결 전 임시)

# ---------------------------------------------------------------- 모터 / 부저 (gpiozero BCM 번호)
PWMA, AIN1, AIN2 = 18, 22, 27   # A = 왼쪽 바퀴
PWMB, BIN1, BIN2 = 23, 25, 24   # B = 오른쪽 바퀴
PWM_FREQUENCY = 100             # gpiozero PWMOutputDevice 기본값
TURN_MODE = "pivot"             # 'pivot': 안쪽 바퀴 감속 / 'spin': 안쪽 바퀴 역회전까지
MOTOR_WATCHDOG_S = 0.5          # 이 시간 동안 모터 명령이 없으면 스스로 정지
BUZZER_PIN = 12
HORN_TONE_HZ = 440              # TonalBuzzer 기본 음역(220~880Hz) 안의 값
HORN_FRAMES = 6                 # 경적 표지판 하나에 울리는 프레임 수

# ---------------------------------------------------------------- 표지판 (YOLO 연결 지점)
# 검출 클래스 이름 -> (이벤트 종류, 값). 클래스 이름은 YOLO 모델에 맞춰 바꾼다.
SIGN_CLASS_EVENTS = {
    "left": ("TURN", "LEFT"),
    "straight": ("TURN", "STRAIGHT"),
    "right": ("TURN", "RIGHT"),
    "horn": ("HORN", None),
    "slow": ("SPEED", 0.6),
}
SIGN_MIN_CONF = 0.5
SIGN_MIN_BOX_H = 40             # 박스가 이 높이(px) 이상일 때만 (충분히 가까울 때만) 반응
SIGN_COOLDOWN_FRAMES = 40       # 같은 클래스 이벤트를 다시 받기까지 기다릴 프레임 수

# ---------------------------------------------------------------- 화면 / 기록
DISPLAY_SCALE = 2               # 오버레이 창 확대 배율
DISPLAY_EVERY = 1               # N프레임마다 한 번만 화면 갱신 (SSH X11 등 느린 화면용)
LOG_DIR = "logs"

# ---------------------------------------------------------------- 키 (cv2.waitKey 값)
# 아래 방향키는 Pi(GTK)에서 84. Windows에서는 waitKey가 방향키를 주지 않으므로 space를 쓴다.
KEY_TOGGLE_RUN = (ord("s"),)
KEY_ESTOP = (84, ord(" "))
KEY_QUIT = (ord("q"),)
KEY_TOGGLE_MASK = (ord("m"),)
KEY_PLAYBACK_PAUSE = (ord("p"),)  # offline: 재생 멈춤/재개
KEY_PLAYBACK_STEP = (ord("n"),)   # offline: 멈춘 상태에서 한 프레임 진행
