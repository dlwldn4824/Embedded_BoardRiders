# 임베디드

광운대 「임베디드 보드를 활용한 인공지능 시스템 최적화 실무」 — Raspberry Pi 자율주행 프로젝트.

- 차선: PilotNet 기반 조향각 회귀 → KD → Pruning → PTQ INT8 (TFLite)
- 표지판: YOLOv8n → TFLite INT8
- 주행: FSM + 후처리 제어

- **다른 PC 에서 이어받기: [HANDOFF.md](HANDOFF.md)**
- 학습 방향 설계: [docs/training-plan.md](docs/training-plan.md)
- 데이터 점검 결과: [docs/data-report.md](docs/data-report.md)
- 차선 데이터 수집 방법: [docs/collection-guide.md](docs/collection-guide.md) (`deploy/collect.py`)
- 팀 규칙 기반 주행 코드(`drive/`)와 추가한 표지판·교차로 동작: [docs/drive-changes.md](docs/drive-changes.md)

## 구조

```
drive/     팀 규칙 기반 주행 (Pi 실행: python3 -m drive.main)
data/      수집 데이터 (git 제외, run 단위 폴더)
lane/      차선 조향 회귀 모델 학습·압축
sign/      표지판 YOLO 학습
deploy/    Pi 실행 코드 (추론, FSM, 모터)
docs/      설계 문서
tests/     컨트롤러·표지판 필터 테스트 (python -m pytest tests)
```
