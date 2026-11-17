# GLP-1 보험 분석

위고비 확산에 따른 사망보험금 분석과 연구용 약제비 보장 특약 모형입니다.

## 폴더 안내
| 폴더 | 내용 |
|---|---|
| `models/start_rate` | 민서 원형·해인 6개 시나리오: 최초 치료 시작률 |
| `models/treatment` | 성환 치료 경로, 통합 전 v1.1 수정 |
| `models/rider` | 생존·유지·미시작 계약에 시작률 연결 |
| `config` | 시나리오·치료·계약 가정 |
| `data/reference` | 실행에 필요한 집계 인구·기초 사망률·근거표 |
| `tests` | 계산 검증 |
| `reports` | 작은 기준 결과 요약 |
| `docs` | 자료 출처와 변경·연결 규칙 |
| `outputs` | 실행 시 생성되는 일별 결과. Git 제외 |

## 실행 순서
저장소 루트에서 Python 3.10 이상으로 실행합니다.
```bash
python -m pip install -r requirements.txt
python -m models.start_rate.run
python -m models.rider.start_integration
python -m unittest discover -s tests -v
```
시작률 파일은 첫 명령 실행 후 자동 생성됩니다. 중복 일별 CSV, 원본 ZIP, 논문 PDF, 캐시, 이전 버전 전체는 올리지 않았습니다.
`config/start_scenarios.json`을 직접 수정하고 시작률 명령을 다시 실행합니다.

기존 루트의 사망보험금 노트북과 산출방법서는 그대로 유지했습니다. 산출방법서의 최신 구현 차이는 docs를 참고합니다.
현재 시작률·치료 모듈·최초 시작 연결까지 완료했고, 급부·보험료·수지 통합은 진행 중입니다.
