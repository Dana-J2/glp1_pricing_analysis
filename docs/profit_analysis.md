# 목표이익 보험료와 고정 보험료 경험 수지

## 정의와 산식

목표이익률 `m`은 `(보험료 수입 현가 - 전체 지출 현가) / 보험료 수입 현가`이다. 전체 지출은 급부, 청구 심사비, 유지비, 고정·비례 신계약비, 수금비, 미경과보험료 반환금이다. 세금·자본비용·준비금·투자손익은 포함하지 않으며 IFRS17 보험계약마진이나 실제 회계이익을 뜻하지 않는다.

- `A = pv_premium_units`: 반환 차감 전 월보험료 1원당 총 납입현가 계수.
- `D = pv_net_premium_units`: 반환과 보험료 연동 사업비를 차감한 순수입현가 계수.
- `C = pv_fixed_outflow_krw`: 보험료 금액과 무관한 급부·사업비 현가.

현재 모형은 월 보험료 자체에 따른 가입·치료 행동 반응을 두지 않는다. 따라서 가격 가정에서 `G = C / (D - mA)`로 월 영업보험료를 산출한다. `D - mA`가 0 이하면 산출하지 않는다. 순보험료 정의는 기존과 같고 목표이익은 영업보험료에만 반영한다.

## 가격과 경험 비교의 분리

`base` 가격 가정으로 0%·5%·10% 영업보험료를 한 번씩 산출한다. 올림 전 보험료로 목표이익률을 검증하고, 원 단위 올림 보험료를 확정 가격으로 사용한다. 경험 비교에서는 이 가격을 다시 산출하지 않고 6개 시작률 시나리오에 고정한다.

6개 시나리오는 `market_low`, `base`, `market_high`, `response_zero`, `response_high`, `combined_stress`이다. 현재 차이는 최초 시작률뿐이며 치료 지속·중단·재개 가정은 공통이다. 모든 현가는 최초 가입자 1,000명 기준이고 가격 산출과 같은 연 2.75%로 할인한다.

지속·재개만 변경하는 별도 13개 경험 시나리오는 [treatment_sensitivity.md](treatment_sensitivity.md)를 참고한다. 두 분석은 현재 교차하지 않는다.

## 실행과 결과

저장소 루트에서 시작률 파일을 먼저 만든 뒤 실행한다.

```bash
python -m models.start_rate.run
python -m models.rider.profit_analysis
```

설정은 `config/profit_analysis.json`, 결과는 다음 별도 파일에 저장된다.

- `outputs/profit_analysis/premium_comparison.csv`: 목표이익률별 보험료와 base 현가.
- `outputs/profit_analysis/fixed_premium_scenario_results.csv`: 원 단위 확정 보험료 3개 × 경험 시나리오 6개.
- `outputs/profit_analysis/scenario_profit_summary.csv`: 시나리오를 행으로 둔 손익·이익률 요약.
- 같은 내용의 JSON과 `validation.json`.

이 실행은 `outputs/pricing/base_premium.json`이나 `reports/`의 기존 기준 스냅샷을 변경하지 않는다.
