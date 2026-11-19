# GLP-1 보험 분석

위고비 확산에 따른 사망보험금 분석과 연구용 약제비 보장 특약 모형입니다.

> 이 저장소의 WE GO FIT은 학술제 연구용 가상 보험상품이며 실제 판매·인가 상품이 아닙니다.

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
| `docs` | 자료 출처·연결 규칙과 `product` 상품 문서 |
| `outputs` | 실행 시 생성되는 일별 결과. Git 제외 |

## 기존 보험계약 영향 분석

- [사망보장 국내분석](notebooks/impact/01_위고비_사망보장_국내연구.ipynb): 위고비 시장 확산에 따른 기대 사망보험금 변화를 분석합니다.
- [심혈관 사건급부 분석](notebooks/impact/02_위고비_심혈관보험금_국내연구.ipynb): 비치명적 심근경색·뇌졸중 기대 사건급부 변화를 분석합니다.
- [담낭수술 급부 분석](notebooks/impact/03_위고비_담낭수술보험금_국내연구.ipynb): 시장 확산과 특약 귀속에 따른 기대 담낭절제술 급부 증가를 구분합니다.
- [기존 보험금 종합영향](notebooks/impact/04_위고비_기존보험금_종합영향.ipynb): 공통 시작·치료·생존 모형에서 네 기존 담보의 저·고확산 보험금 현가 변화를 통합합니다.

네 분석은 위고비 시장 확산 시나리오에 따른 기존 보험계약의 기대 급부 변화를 다루며, 특약 가입으로 발생하는 순증 효과와는 구분합니다.

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
python -m models.rider.mvp_age_analysis --build-sale-lookup
python -m models.impact.existing_claims
python -m unittest discover -s tests -v
```
시작률 파일은 첫 명령 실행 후 자동 생성됩니다. 중복 일별 CSV, 원본 ZIP, 논문 PDF, 캐시, 이전 버전 전체는 올리지 않았습니다.
`config/start_scenarios.json`을 직접 수정하고 시작률 명령을 다시 실행합니다.

## 현재 완료 범위
- 4단계: 사망·대기기간·지급한도·월납·소진 반환·사업비를 연결한 10년형과 실제 5년형 현금흐름.
- 5단계: 기준 보험료 산출·검증과 목표이익률 0%·5%·10% 가격 및 고정 보험료 경험 수지 비교 완료.
- 치료 행태: `base` 시작률을 고정한 지속·재개 13개 시나리오와 확정 보험료 3개의 39개 수지 비교 완료.
- 상품조건: 지급액 3개×지급한도 3개의 가격·혜택과 시작률 108행·치료 행태 234행 분리 스트레스 비교 완료.
- MVP: 지급액 10만·15만 원×지급한도 6·13회×대기기간 30·90일의 8개 비교군을 확정하고 팀원 27개 표와 재현 대조 완료.
- 가입연령: 보험나이 19~69세는 10년형, 70~74세는 실제 5년 현금흐름을 사용하는 고령 확장형, 75세 이상은 가입 불가·연구 계산 전용. 최대 보장종료연령은 79세.
- 판매 기준: 목표이익률 10%만 판매 후보 보험료로 사용. 기존 0%·5%는 연구 비교와 회귀검증용으로 보존.
- 월 순보험료 2,006.86원, 월 영업보험료 2,807.05원(원 단위 올림 2,808원).
- 목표이익률은 보험료 수입 현가 대비 연구용 현금흐름 이익률이며 세금·자본비용·준비금·투자손익은 제외.
- 실제 판매 확정 보험료나 IFRS17 보험계약마진이 아닌 연구용 판매 후보 보험료와 경험 수지 결과.

## 먼저 볼 파일
- [적용 가정·현금흐름](docs/cashflows.md)
- [현금흐름 노트북](notebooks/04_계약현금흐름.ipynb)
- [기준 보험료·산출 검증 설명](docs/pricing_validation.md)
- [목표이익 보험료·고정 보험료 수지 비교](docs/profit_analysis.md)
- [치료 지속·재개 민감도](docs/treatment_sensitivity.md)
- [지급액·지급한도 9개 상품조건 비교](docs/product_comparison.md)
- [MVP 8개 상품조건 선정](docs/mvp_product_selection.md)
- [가입연령 정책·연령별 탐색](docs/issue_age_policy.md)
- [WE GO FIT 사업방법서](docs/product/WE_GO_FIT_사업방법서.md) ([DOCX](docs/product/WE_GO_FIT_사업방법서.docx), [PDF](docs/product/WE_GO_FIT_사업방법서.pdf))
- [WE GO FIT 보험료 산출방법서](docs/product/WE_GO_FIT_산출방법서.md)
- [기존 담보 보험금 종합영향](docs/existing_claims_impact.md)
- [MVP 8개 추적 결과](data/derived/mvp_product_options_summary.csv)
- [기본 가입연령 분석 결과](data/derived/issue_age_baseline_results.csv)
- [연령별 중단률 탐색 결과](data/derived/issue_age_discontinuation_exploratory_results.csv)
- [70~74세 5년형 80개 결과](data/derived/senior_extension_5year_results.csv)
- [판매가격 전 연령 조회표 896행](data/derived/sale_product_age_lookup.csv)
- [대표연령·고령 확장 요약 94행](data/derived/sale_product_age_representative_summary.csv)
- [고령 확장형 검증 결과](reports/senior_extension_5year_validation.json)
- [보험료 결과 스냅샷](reports/base_premium.json)
- [검증 결과](reports/pricing_validation.json)
- [수입·지출 현재가치 표](reports/pricing_pv_breakdown.csv)

노트북은 저장소 루트 또는 notebooks 폴더에서 실행하며, 먼저 시작률 생성 명령을 실행해야 합니다. Jupyter 실행 환경은 별도 준비합니다.
reports는 현재 가정의 기준 결과이며 입력을 바꾸면 자동 갱신되지 않습니다. 최신 실행 결과는 outputs에 생성됩니다.
`pricing` 단독 명령의 기본값 0%는 회귀검증용입니다. 판매 후보 조회표는 목표이익률 10%만 사용합니다. 예를 들어 5% 연구가격만 별도로 산출하려면
`python -m models.rider.pricing --target-profit-margin 0.05`를 실행합니다.
