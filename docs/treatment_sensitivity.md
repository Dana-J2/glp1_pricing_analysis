# 치료 지속·재개 민감도

## 분석 범위

이 분석은 `base`의 특약 있음 일별 최초 시작확률을 고정하고 치료 지속·재개 가정만 바꾼다. 가격은 기준 치료 행태로 한 번 산출한 뒤 원 단위 올림 보험료를 모든 치료 경험 시나리오에 고정한다. 결과의 이익률은 관측 실적이 아닌 **시나리오 이익률**이며, 손익 현가를 보험료 수입 현가로 나눈 연구용 지표다.

| 분석 종류 | 최초 시작률 | 지속·재개 |
|---|---|---|
| 시작률 민감도 | `market_low` 등 기존 6개 | 기준 고정 |
| 치료 행태 민감도 | `base` 고정 | `treatment_` 접두사의 13개 |

시작률의 `combined_stress`와 치료 행태의 `treatment_combined_stress`는 서로 다른 시나리오다. 이번 실행은 6×13 교차 스트레스를 포함하지 않는다.

## 설정 정의

시나리오는 `config/treatment_scenarios.json`에 있고, 각 행은 `config/treatment.json`의 새 복사본에 해당 행의 `overrides`만 적용한다.

- `stop_reduction`: 기준 일별 조건부 중단확률에 `(1-stop_reduction)`을 곱한다. 지속률에 퍼센트포인트를 더하지 않는다.
- `restart_in_window_fraction`: 중단 후 60일이 지난 뒤 90일 재개 관찰창 전체의 누적 재개확률이다. 모형이 이를 일정 일별 재개확률로 환산한다.
- `postrestart_stop_hazard_multiplier`: 호환용 이름에 `hazard`가 있지만 실제로는 재개 후 일별 조건부 중단확률에 곱한다.

기준 지속곡선은 모든 시나리오에서 같고 중단 감소 효과를 중복 반영하지 않는다. 기준 외 수준은 연구용 민감도 가정이며 문헌 신뢰구간이나 관측 상한이 아니다.

## 계산 순서

```text
기준 상품과 그 상품에서 다시 산출한 시작률
→ 기준 치료 행태로 목표이익률별 보험료 산출
→ 원 단위 올림 후 보험료 고정
→ 시작률을 유지하고 치료 행태만 변경
→ 시나리오별 현금흐름과 손익 평가
```

각 치료 경로는 한 번만 계산하고 0%·5%·10%의 고정 보험료에 재사용한다. 보험료 연동 신계약비·수금비·미경과보험료 반환은 보험료마다 다시 계산한다. 직접 합산 손익과 `G×D_s-C_s`를 대조한다.

기본 실행은 상품조건 10만원·13회·90일 및 시작률 설정의 지급액·한도·대기기간·`decision_horizon_units=13`이 일치하는지 확인한다. 따라서 이 상품의 시작률 CSV와 2,808원·2,965원·3,141원을 다른 상품에 자동 재사용하지 않는다. 공통 함수 `analyze_treatment_scenarios()`는 상품 설정과 그 상품에서 생성한 시작률 행을 인자로 받으므로 향후 상품조건 비교에서 다음 순서를 지원한다.

```text
상품조건 선택 → 시작률 재산출 → 기준 치료 행태로 가격 산출
→ 보험료 고정 → 시작률 또는 치료 행태 스트레스 평가
```

## 실행과 출력

```bash
python -m models.start_rate.run
python -m models.rider.treatment_sensitivity
```

결과는 `outputs/treatment_sensitivity/` 아래에 기존 결과와 분리해 저장한다.

- `pricing_basis.csv`: 기준 치료 행태에서 산출한 0%·5%·10% 보험료.
- `treatment_sensitivity_39_results.csv`: 치료 13개 × 고정 보험료 3개의 상세 현가.
- `treatment_sensitivity_summary.csv`: 지급 건수·손익·시나리오 이익률 요약.
- `team_reproduction.csv`: 성환 팀원 원정밀도 결과와 현행 엔진 재실행 대조.
- `validation.json`: 확률·인원·지급한도·설정 독립성·손익식 등의 검증.

팀원 참조값은 `data/reference/treatment_sensitivity_team_results.csv`에 보존했다. ZIP의 과거 현가 재가중 스크립트는 통합하지 않고 공통 현금흐름 엔진을 호출한다.
