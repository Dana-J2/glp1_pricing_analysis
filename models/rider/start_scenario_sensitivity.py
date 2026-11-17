import csv
import json

from models.rider.cashflows import (
    CashflowSettings,
    project_cashflows,
    summarize,
)
from models.rider.start_integration import (
    ROOT,
    read_starts,
    read_mortality,
)
from models.treatment.model import Parameters


def main():
    start_file = ROOT / "outputs/start_rate/start_probabilities.csv"
    premium_file = ROOT / "outputs/pricing/base_premium.json"

    if not start_file.exists():
        raise FileNotFoundError(
            "시작확률 파일이 없습니다. "
            "먼저 python -m models.start_rate.run 을 실행하세요."
        )

    if not premium_file.exists():
        raise FileNotFoundError(
            "기준 보험료 파일이 없습니다. "
            "먼저 python -m models.rider.pricing 을 실행하세요."
        )

    scenario_config = json.loads(
        (ROOT / "config/start_scenarios.json").read_text(encoding="utf-8")
    )
    scenario_names = [
        item["inputs"]["name"]
        for item in scenario_config["scenarios"]
    ]

    if len(scenario_names) != 6:
        raise ValueError(f"시나리오가 6개가 아닙니다: {len(scenario_names)}개")

    cashflow_config = json.loads(
        (ROOT / "config/cashflows.json").read_text(encoding="utf-8")
    )
    treatment_config = json.loads(
        (ROOT / "config/treatment.json").read_text(encoding="utf-8")
    )
    premium_data = json.loads(
        premium_file.read_text(encoding="utf-8")
    )

    settings = CashflowSettings(**cashflow_config)
    treatment = Parameters(**treatment_config)
    mortality = read_mortality()

    # 기준 시나리오에서 산출된 정밀 월 보험료를 여섯 시나리오 모두에 고정
    fixed_monthly_premium = float(
        premium_data["monthly_gross_premium_krw"]
    )

    results = []

    for scenario_name in scenario_names:
        starts = read_starts(start_file, scenario_name)

        daily_cashflows = project_cashflows(
            starts,
            mortality,
            settings=settings,
            treatment=treatment,
            path="yes",
        )

        summary = summarize(
            daily_cashflows,
            settings.initial_policies,
        )

        # 보험료 수입 현가
        premium_income = (
            fixed_monthly_premium * summary["pv_premium_units"]
        )

        # 보험료에서 차감되는 항목
        refund_pv = (
            fixed_monthly_premium * summary["pv_refund_units"]
        )
        variable_acquisition_pv = (
            fixed_monthly_premium * summary["pv_acquisition_units"]
        )
        collection_expense_pv = (
            fixed_monthly_premium * summary["pv_collection_units"]
        )

        # 보험금·심사비·유지비·고정 신계약비 등의 현가
        fixed_outflow_pv = summary["pv_fixed_outflow_krw"]

        total_outflow = (
            refund_pv
            + variable_acquisition_pv
            + collection_expense_pv
            + fixed_outflow_pv
        )

        balance = premium_income - total_outflow

        if balance > 0.01:
            status = "흑자"
        elif balance < -0.01:
            status = "적자"
        else:
            status = "수지균형"

        results.append({
            "scenario": scenario_name,
            "result": status,
            "fixed_monthly_premium_krw": fixed_monthly_premium,
            "total_starts": summary["total_starts"],
            "total_paid_claims": summary["total_paid_claims"],

            "pv_premium_income_krw": premium_income,
            "pv_refund_krw": refund_pv,
            "pv_variable_acquisition_krw": variable_acquisition_pv,
            "pv_collection_expense_krw": collection_expense_pv,
            "pv_fixed_outflow_krw": fixed_outflow_pv,
            "pv_total_outflow_krw": total_outflow,

            "balance_formula": (
                "보험료 수입 현가 - "
                "(반환액 현가 + 변동 신계약비 현가 + "
                "수금비 현가 + 고정지출 현가)"
            ),
            "pv_balance_krw": balance,
            "pv_balance_per_policy_krw": (
                balance / settings.initial_policies
            ),
        })

    base_balance = next(
        row["pv_balance_krw"]
        for row in results
        if row["scenario"] == "base"
    )

    for row in results:
        row["balance_change_vs_base_krw"] = (
            row["pv_balance_krw"] - base_balance
        )

    report_file = ROOT / "reports/start_scenario_sensitivity.csv"

    with report_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

        print(f"월 고정 보험료: {fixed_monthly_premium:,.6f}원")

    for row in results:
        print(f'\n[{row["scenario"]}] {row["result"]}')
        print(
            f'보험료 수입 현가 '
            f'{row["pv_premium_income_krw"]:,.0f}원'
        )
        print(
            f'총지출 현가 '
            f'{row["pv_total_outflow_krw"]:,.0f}원'
        )
        print(
            f'계산: 보험료 수입 현가 - 총지출 현가 '
            f'= {row["pv_premium_income_krw"]:,.0f} '
            f'- {row["pv_total_outflow_krw"]:,.0f} '
            f'= {row["pv_balance_krw"]:+,.0f}원'
        )
        print(
            "총지출 구성: "
            f'반환액 {row["pv_refund_krw"]:,.0f}원 + '
            f'변동 신계약비 {row["pv_variable_acquisition_krw"]:,.0f}원 + '
            f'수금비 {row["pv_collection_expense_krw"]:,.0f}원 + '
            f'고정지출 {row["pv_fixed_outflow_krw"]:,.0f}원'
        )


if __name__ == "__main__":
    main()