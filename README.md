# GLP-1 보험 분석

위고비 확산에 따른 사망보험금 분석과 연구용 약제비 보장 특약 모형입니다.

## 폴더 안내
| 폴더 | 내용 |
|---|---|
| `models/start_rate` | 민서 원형·해인 6개 시나리오: 최초 치료 시작률 |
| `models/treatment` | 성환 치료 경로, 통합 전 v1.1 수정 |
| `models/rider` | 계약 상태·현금흐름·보험료 산출·검증 |
| `notebooks` | 현금흐름 실행·결과 확인 |
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
python -m models.rider.cashflows
python -m models.rider.pricing
python -m models.rider.validate_pricing
python -m models.rider.profit_analysis
python -m models.rider.treatment_sensitivity
python -m models.rider.product_comparison --stage full
python -m unittest discover -s tests -v
```
시작률 파일은 첫 명령 실행 후 자동 생성됩니다. 중복 일별 CSV, 원본 ZIP, 논문 PDF, 캐시, 이전 버전 전체는 올리지 않았습니다.
`config/start_scenarios.json`을 직접 수정하고 시작률 명령을 다시 실행합니다.

기존 루트의 사망보험금 노트북은 유지했고, 산출방법서에는 목표이익률 정의와 산식만 추가했습니다. 최신 구현 차이는 docs를 참고합니다.
## 현재 완료 범위
- 4단계: 사망·대기기간·지급한도·월납·소진 반환·사업비를 연결한 10년 현금흐름.
- 5단계: 기준 보험료 산출·검증과 목표이익률 0%·5%·10% 가격 및 고정 보험료 경험 수지 비교 완료.
- 치료 행태: `base` 시작률을 고정한 지속·재개 13개 시나리오와 확정 보험료 3개의 39개 수지 비교 완료.
- 상품조건: 지급액 3개×지급한도 3개의 가격·혜택과 시작률 108행·치료 행태 234행 분리 스트레스 비교 완료.
- 월 순보험료 2,006.86원, 월 영업보험료 2,807.05원(원 단위 올림 2,808원).
- 목표이익률은 보험료 수입 현가 대비 연구용 현금흐름 이익률이며 세금·자본비용·준비금·투자손익은 제외.
- 실제 판매 보험료나 IFRS17 보험계약마진이 아닌 연구 가정의 보험료와 경험 수지 결과.

## 먼저 볼 파일
- [적용 가정·현금흐름](docs/cashflows.md)
- [현금흐름 노트북](notebooks/04_계약현금흐름.ipynb)
- [기준 보험료·산출 검증 설명](docs/pricing_validation.md)
- [목표이익 보험료·고정 보험료 수지 비교](docs/profit_analysis.md)
- [치료 지속·재개 민감도](docs/treatment_sensitivity.md)
- [지급액·지급한도 9개 상품조건 비교](docs/product_comparison.md)
- [보험료 결과 스냅샷](reports/base_premium.json)
- [검증 결과](reports/pricing_validation.json)
- [수입·지출 현재가치 표](reports/pricing_pv_breakdown.csv)

노트북은 저장소 루트 또는 notebooks 폴더에서 실행하며, 먼저 시작률 생성 명령을 실행해야 합니다. Jupyter 실행 환경은 별도 준비합니다.
reports는 현재 가정의 기준 결과이며 입력을 바꾸면 자동 갱신되지 않습니다. 최신 실행 결과는 outputs에 생성됩니다.
`pricing`은 기본적으로 기존 0% 가격을 산출합니다. 예를 들어 5% 가격만 별도로 산출하려면
`python -m models.rider.pricing --target-profit-margin 0.05`를 실행합니다.
