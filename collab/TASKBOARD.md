# 작업판 (Claude ↔ Codex, 10/13 보스 지시)

## 운영 규칙
- **Codex = 작업자**: 이 문서의 "Codex 할 일"에서 위부터 하나씩 진행한다. **1시간마다** 이 문서를 다시 읽고 새 지시·수정 요청을 반영한 뒤 다음 일을 한다.
- **Claude = 점검자 + 자기 작업**: 1시간마다(매시 약 :07) Codex가 한 일(git diff, 새 파일, LOG)을 점검하고, 고칠 것은 직접 고치거나 "수정 요청"에 적는다. 결과는 이 문서의 "점검 기록"과 `collab/LOG.md`에 남긴다. 그 사이에는 자기 단계를 진행한다.
- Codex는 한 항목을 끝내면 "상태"를 `완료(커밋 해시)`로 바꾸고, 결과·수치·경로를 `collab/LOG.md`에 적는다. 막히면 `막힘: 이유`라고 적고 다음 항목으로 넘어간다.
- 같은 파일을 동시에 고치지 않는다. "Claude 작업 중 파일"은 Codex가 수정 금지.
- 공통 규칙: `collab/CODEX_BRIEF_2026-10-13.md`의 규칙 절(로봇·그리퍼 명령 금지, 덮어쓰기 금지, 코드 배치, 그림 규칙, 데이터 git 금지, "보스" 호칭).

## Claude 작업 중 파일 (Codex 수정 금지)
- `calibration/plate_plans.py`, `calibration/robot_model.py`, `calibration/ros2_nodes.py`, `calibration/pipeline.py`
- `rl/basket/*` (오늘 밤 정책 재학습)

## Codex 할 일 (위에서부터)
| # | 일 | 자세한 내용 | 상태 |
|---|---|---|---|
| A | 손목 카메라 핀 쌍 검출 + 번호(위→아래) | `CODEX_BRIEF_2026-10-13.md` A | 대기 |
| B | 바구니 마커 크기 실험 측정 모드 + 10 cm 기준값 | 같은 문서 B | 대기 |
| C | failure detection 선행 연구 요약 | 같은 문서 C | 대기 |

## 수정 요청 (Claude → Codex)
- (없음)

## 점검 기록 (Claude, 1시간마다)
- 10/13 18:xx 작업판 시작.
