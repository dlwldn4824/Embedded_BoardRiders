# 임베디드

광운대 「임베디드 보드를 활용한 인공지능 시스템 최적화 실무」 — Raspberry Pi 자율주행 프로젝트.

- 차선: PilotNet 기반 조향각 회귀 → KD → Pruning → PTQ INT8 (TFLite)
- 표지판: YOLOv8n → TFLite INT8
- 주행: FSM + 후처리 제어

- 학습 방향 설계: [docs/training-plan.md](docs/training-plan.md)
- 데이터 점검 결과: [docs/data-report.md](docs/data-report.md)
- 차선 데이터 수집 방법: [docs/collection-guide.md](docs/collection-guide.md) (`deploy/collect.py`)

## 구조

```
data/      수집 데이터 (git 제외, run 단위 폴더)
lane/      차선 조향 회귀 모델 학습·압축
sign/      표지판 YOLO 학습
deploy/    Pi 실행 코드 (추론, FSM, 모터)
docs/      설계 문서
```
