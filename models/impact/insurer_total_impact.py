"""약제비 특약 손익과 기존 보험금 변화를 같은 시작경로에서 결합한다.

특약 도입효과는 시장 저확산·고확산이 아니라 동일 상품의 특약 없음 `A`와
특약 있음 `A+B`를 비교한다. 판매보험료는 기존 산출 결과의 목표이익률 10%
원 단위 확정값을 그대로 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from models.impact.existing_claims import (
    BENEFIT_IDS,
    _base_parameters as existing_base_parameters,
    _simulate_market as simulate_existing_claims,
    _treatment_on_fraction,
    _with_overrides as existing_with_overrides,
    load_config as load_existing_claims_config,
)
from models.rider.cashflows import CashflowSettings, project_cashflows, summarize
from models.rider.mvp_age_analysis import build_mvp_designs, load_mvp_config
from models.rider.product_comparison import generate_start_rows, load_start_templates
from models.rider.profit_analysis import evaluate_projected_cashflows
from models.rider.start_integration import ROOT, read_mortality
from models.treatment.model import Parameters, params_from_json


DEFAULT_CONFIG = ROOT / "config/insurer_total_impact.json"
DEFAULT_OUTPUT = ROOT / "outputs/insurer_total_impact"
DISPLAY_NAMES = {
    "death": "사망",
    "mi": "비치명적 심근경색",
    "stroke": "비치명적 뇌졸중",
    "gallbladder": "담낭절제술",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _read_json(path)
    analysis = config["analysis"]
    weights = analysis["sex_weights"]
    if set(weights) != {"남", "여"} or not math.isclose(
        math.fsum(float(value) for value in weights.values()), 1.0
    ):
        raise ValueError("남녀 가중치는 합계 1이어야 합니다.")
    if int(analysis["representative_issue_age"]) != 62:
        raise ValueError("공통 대표 가입연령은 62세여야 합니다.")
    if not math.isclose(float(analysis["target_profit_margin"]), 0.10):
        raise ValueError("이번 분석은 목표이익률 10% 확정 보험료만 사용합니다.")
    scenarios = config["coverage_scenarios"]
    if tuple(scenarios) != ("low", "base", "high"):
        raise ValueError("보유율 시나리오는 low, base, high 순서여야 합니다.")
    for scenario in scenarios.values():
        if any(not 0 <= float(scenario[key]) <= 1 for key in BENEFIT_IDS):
            raise ValueError("담보 보유율은 [0,1]이어야 합니다.")
    return config


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_fixed_premiums(config: dict[str, Any]) -> dict[str, float]:
    source = _read_json(_resolve(config["inputs"]["fixed_premium_results"]))
    target = float(config["analysis"]["target_profit_margin"])
    if not math.isclose(target, 0.10):
        raise ValueError("저장된 판매보험료 열은 목표이익률 10% 전용입니다.")
    rows = source["mvp_products"]
    premiums = {
        row["product_id"]: float(row["monthly_premium_target_10pct_rounded_krw"])
        for row in rows
    }
    if len(premiums) != 8 or any(value != math.ceil(value) for value in premiums.values()):
        raise ValueError("8개 상품의 원 단위 확정 보험료를 확인하세요.")
    return premiums


def _weighted_numeric(
    values: dict[str, dict[str, Any]], weights: dict[str, float], keys: tuple[str, ...]
) -> dict[str, float]:
    return {
        key: math.fsum(float(weights[sex]) * float(values[sex][key]) for sex in weights)
        for key in keys
    }


def _evaluate_rider(
    *,
    start_rows: list[dict[str, Any]],
    design: dict[str, Any],
    fixed_premium: float,
    issue_age: int,
    sex_weights: dict[str, float],
    mortality: dict[tuple[int, str], float],
    treatment: Parameters,
) -> dict[str, Any]:
    evaluations: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for sex in sex_weights:
        settings = CashflowSettings(
            **{
                **design["settings_config"],
                "entry_age": issue_age,
                "sex": sex,
            }
        )
        projected = project_cashflows(
            start_rows, mortality, settings, treatment, path="yes"
        )
        evaluations[sex] = evaluate_projected_cashflows(
            projected, settings, fixed_premium
        )
        summaries[sex] = summarize(projected, settings.initial_policies)

    evaluation_keys = (
        "premium_income_pv_krw",
        "benefit_pv_krw",
        "expenses_pv_krw",
        "refund_pv_krw",
        "total_outflow_pv_krw",
        "profit_pv_krw",
    )
    summary_keys = (
        "total_starts",
        "recipients",
        "total_paid_claims",
        "exhausted",
        "deaths",
    )
    weighted = _weighted_numeric(evaluations, sex_weights, evaluation_keys)
    weighted.update(_weighted_numeric(summaries, sex_weights, summary_keys))
    weighted["profit_margin"] = (
        weighted["profit_pv_krw"] / weighted["premium_income_pv_krw"]
    )
    weighted["sex_results"] = {
        sex: {"evaluation": evaluations[sex], "summary": summaries[sex]}
        for sex in sex_weights
    }
    return weighted


def _evaluate_existing_claims(
    *,
    existing_config: dict[str, Any],
    start_rows: list[dict[str, Any]],
    on_fraction: np.ndarray,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    dates = [str(row["date"]) for row in start_rows]
    q_without = np.asarray([float(row["p_start_no"]) for row in start_rows])
    q_with = np.asarray([float(row["p_start_yes"]) for row in start_rows])
    without = simulate_existing_claims(
        existing_config,
        "rider_without_A",
        q_without,
        dates,
        parameters,
        on_fraction,
    )
    with_rider = simulate_existing_claims(
        existing_config,
        "rider_with_A_plus_B",
        q_with,
        dates,
        parameters,
        on_fraction,
    )
    claim_rows = []
    for benefit_id in BENEFIT_IDS:
        without_claim = without["claims"][benefit_id]
        with_claim = with_rider["claims"][benefit_id]
        change = with_claim["pv_krw"] - without_claim["pv_krw"]
        claim_rows.append(
            {
                "benefit_id": benefit_id,
                "benefit_name_ko": DISPLAY_NAMES[benefit_id],
                "without_rider_expected_claim_count": without_claim[
                    "expected_claim_count"
                ],
                "with_rider_expected_claim_count": with_claim[
                    "expected_claim_count"
                ],
                "without_rider_claim_pv_krw": without_claim["pv_krw"],
                "with_rider_claim_pv_krw": with_claim["pv_krw"],
                "claim_change_krw": change,
            }
        )
    daily_rows = []
    for without_day, with_day in zip(without["daily_rows"], with_rider["daily_rows"]):
        daily_rows.append(
            {
                "date": without_day["date"],
                "elapsed_day": without_day["elapsed_day"],
                "without_rider_A_cumulative_first_starts": without_day[
                    "cumulative_first_starts"
                ],
                "with_rider_A_plus_B_cumulative_first_starts": with_day[
                    "cumulative_first_starts"
                ],
                "additional_starts_B": with_day["cumulative_first_starts"]
                - without_day["cumulative_first_starts"],
                "without_rider_on_treatment_alive": without_day[
                    "on_treatment_alive"
                ],
                "with_rider_on_treatment_alive": with_day["on_treatment_alive"],
            }
        )
    return {
        "without": without,
        "with": with_rider,
        "claim_rows": claim_rows,
        "daily_rows": daily_rows,
    }


def coverage_adjustment(
    claim_rows: list[dict[str, Any]],
    holding_rates: dict[str, float],
    amount_multiplier: float = 1.0,
) -> dict[str, Any]:
    """기존 보험금 변화에 보유율과 가입금액 배수를 적용한다."""
    if amount_multiplier < 0 or not math.isfinite(amount_multiplier):
        raise ValueError("가입금액 배수는 0 이상이어야 합니다.")
    by_id = {row["benefit_id"]: row for row in claim_rows}
    components = {
        benefit_id: -float(by_id[benefit_id]["claim_change_krw"])
        * float(holding_rates[benefit_id])
        * amount_multiplier
        for benefit_id in BENEFIT_IDS
    }
    return {
        "components": components,
        "existing_claims_adjustment_krw": math.fsum(components.values()),
    }


def _market_reference_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve(config["inputs"]["market_expansion_claim_results"])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["benefit_id"] in BENEFIT_IDS]
    return [
        {
            "benefit_id": row["benefit_id"],
            "claim_change_krw": float(row["claim_change_krw"]),
        }
        for row in rows
    ]


def _coverage_row(
    *,
    product: dict[str, Any],
    scenario_id: str,
    scenario: dict[str, Any],
    market_reference: list[dict[str, Any]],
    initial_contracts: float,
) -> dict[str, Any]:
    holding_rates = {key: float(scenario[key]) for key in BENEFIT_IDS}
    adjustment = coverage_adjustment(product["claims"]["claim_rows"], holding_rates)
    market_adjustment = coverage_adjustment(market_reference, holding_rates)
    rider_npv = float(product["rider"]["profit_pv_krw"])
    total = rider_npv + adjustment["existing_claims_adjustment_krw"]
    components = adjustment["components"]
    return {
        "product_id": product["product_id"],
        "benefit_per_dispense_krw": product["benefit_per_dispense_krw"],
        "max_claims": product["max_claims"],
        "waiting_days": product["waiting_days"],
        "fixed_monthly_premium_krw": product["fixed_monthly_premium_krw"],
        "coverage_scenario_id": scenario_id,
        "coverage_scenario_name_ko": scenario["display_name_ko"],
        "death_holding_rate": holding_rates["death"],
        "mi_holding_rate": holding_rates["mi"],
        "stroke_holding_rate": holding_rates["stroke"],
        "gallbladder_holding_rate": holding_rates["gallbladder"],
        "rider_premium_income_pv_krw": product["rider"][
            "premium_income_pv_krw"
        ],
        "rider_benefit_pv_krw": product["rider"]["benefit_pv_krw"],
        "rider_expenses_pv_krw": product["rider"]["expenses_pv_krw"],
        "rider_refund_pv_krw": product["rider"]["refund_pv_krw"],
        "rider_total_outflow_pv_krw": product["rider"][
            "total_outflow_pv_krw"
        ],
        "rider_npv_krw": rider_npv,
        "rider_profit_margin": product["rider"]["profit_margin"],
        "death_existing_claims_adjustment_krw": components["death"],
        "mi_existing_claims_adjustment_krw": components["mi"],
        "stroke_existing_claims_adjustment_krw": components["stroke"],
        "gallbladder_existing_claims_adjustment_krw": components[
            "gallbladder"
        ],
        "existing_claims_adjustment_krw": adjustment[
            "existing_claims_adjustment_krw"
        ],
        "insurer_total_impact_krw": total,
        "rider_npv_per_contract_krw": rider_npv / initial_contracts,
        "existing_claims_adjustment_per_contract_krw": adjustment[
            "existing_claims_adjustment_krw"
        ]
        / initial_contracts,
        "insurer_total_impact_per_contract_krw": total / initial_contracts,
        "without_rider_first_starts_A": product["claims"]["without"][
            "total_first_starts"
        ],
        "with_rider_first_starts_A_plus_B": product["claims"]["with"][
            "total_first_starts"
        ],
        "additional_first_starts_B": product["claims"]["with"][
            "total_first_starts"
        ]
        - product["claims"]["without"]["total_first_starts"],
        "without_rider_treatment_exposure_person_years": product["claims"][
            "without"
        ]["treatment_exposure_person_years"],
        "with_rider_treatment_exposure_person_years": product["claims"]["with"][
            "treatment_exposure_person_years"
        ],
        "market_expansion_reference_adjustment_krw": market_adjustment[
            "existing_claims_adjustment_krw"
        ],
        "comparison_basis": "rider introduction A vs A+B",
        "cohort_basis": "initial 1000 contracts",
    }


def _validation(
    *,
    config: dict[str, Any],
    products: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    representative_id: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add(
        "with_rider_starts_not_less_than_without_rider",
        all(
            item["claims"]["with"]["total_first_starts"]
            >= item["claims"]["without"]["total_first_starts"] - 1e-10
            for item in products
        ),
    )
    zero = {key: 0.0 for key in BENEFIT_IDS}
    add(
        "zero_holding_rates_produce_zero_adjustment",
        all(
            abs(
                coverage_adjustment(item["claims"]["claim_rows"], zero)[
                    "existing_claims_adjustment_krw"
                ]
            )
            < 1e-9
            for item in products
        ),
    )
    rep = {
        row["coverage_scenario_id"]: row
        for row in coverage_rows
        if row["product_id"] == representative_id
    }
    add(
        "coverage_scenario_adjustment_increases_low_base_high",
        rep["low"]["existing_claims_adjustment_krw"]
        <= rep["base"]["existing_claims_adjustment_krw"]
        <= rep["high"]["existing_claims_adjustment_krw"],
    )
    add(
        "gallbladder_effect_is_negative_for_insurer",
        all(row["gallbladder_existing_claims_adjustment_krw"] <= 0 for row in coverage_rows),
    )
    max_component_error = max(
        abs(
            row["existing_claims_adjustment_krw"]
            - math.fsum(
                row[f"{key}_existing_claims_adjustment_krw"] for key in BENEFIT_IDS
            )
        )
        for row in coverage_rows
    )
    max_total_error = max(
        abs(
            row["insurer_total_impact_krw"]
            - row["rider_npv_krw"]
            - row["existing_claims_adjustment_krw"]
        )
        for row in coverage_rows
    )
    add(
        "benefit_components_sum_to_existing_claims_adjustment",
        max_component_error < 1e-6,
        f"max_error={max_component_error:.3g}",
    )
    add(
        "rider_npv_plus_claim_adjustment_equals_total_impact",
        max_total_error < 1e-6,
        f"max_error={max_total_error:.3g}",
    )
    add(
        "rider_contribution_smaller_than_market_expansion_reference",
        abs(rep["base"]["existing_claims_adjustment_krw"])
        < abs(rep["base"]["market_expansion_reference_adjustment_krw"]),
    )
    add("eight_products_by_three_coverage_scenarios", len(coverage_rows) == 24)
    add(
        "fixed_premium_unchanged_across_coverage_scenarios",
        all(
            len(
                {
                    row["fixed_monthly_premium_krw"]
                    for row in coverage_rows
                    if row["product_id"] == item["product_id"]
                }
            )
            == 1
            for item in products
        ),
    )
    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    return {"status": status, "checks": checks}


def build_analysis(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    analysis_config = config["analysis"]
    mvp_config = load_mvp_config(_resolve(config["inputs"]["mvp_product_config"]))
    cashflow_config = _read_json(_resolve(config["inputs"]["cashflow_config"]))
    treatment = Parameters(
        **_read_json(_resolve(config["inputs"]["treatment_config"]))
    )
    designs = build_mvp_designs(mvp_config, cashflow_config)
    templates = load_start_templates()
    start_template = templates[analysis_config["start_scenario"]]
    fixed_premiums = _load_fixed_premiums(config)
    mortality = read_mortality()

    existing_config = load_existing_claims_config(
        _resolve(config["inputs"]["existing_claims_config"])
    )
    existing_config["analysis"]["initial_cohort"] = float(
        analysis_config["initial_contracts"]
    )
    existing_treatment = params_from_json(
        ROOT / existing_config["treatment"]["params_path"]
    )
    first_bundle = generate_start_rows(start_template, designs[0])
    on_fraction, treatment_validation = _treatment_on_fraction(
        len(first_bundle["rows"]), existing_treatment
    )
    existing_parameters = existing_with_overrides(
        existing_base_parameters(existing_config),
        {
            "holding_rates": {key: 1.0 for key in BENEFIT_IDS},
            "benefit_amounts": {
                key: float(config["benefit_amounts_krw"][key])
                for key in BENEFIT_IDS
            },
        },
    )

    products: list[dict[str, Any]] = []
    for index, design in enumerate(designs):
        bundle = first_bundle if index == 0 else generate_start_rows(start_template, design)
        product_id = design["design_id"]
        rider = _evaluate_rider(
            start_rows=bundle["rows"],
            design=design,
            fixed_premium=fixed_premiums[product_id],
            issue_age=int(analysis_config["representative_issue_age"]),
            sex_weights={
                key: float(value)
                for key, value in analysis_config["sex_weights"].items()
            },
            mortality=mortality,
            treatment=treatment,
        )
        claims = _evaluate_existing_claims(
            existing_config=existing_config,
            start_rows=bundle["rows"],
            on_fraction=on_fraction,
            parameters=existing_parameters,
        )
        products.append(
            {
                "product_id": product_id,
                "benefit_per_dispense_krw": int(design["benefit"]),
                "max_claims": int(design["max_claims"]),
                "waiting_days": int(design["waiting_days"]),
                "fixed_monthly_premium_krw": fixed_premiums[product_id],
                "rider": rider,
                "claims": claims,
            }
        )

    market_reference = _market_reference_rows(config)
    coverage_rows = [
        _coverage_row(
            product=product,
            scenario_id=scenario_id,
            scenario=scenario,
            market_reference=market_reference,
            initial_contracts=float(analysis_config["initial_contracts"]),
        )
        for product in products
        for scenario_id, scenario in config["coverage_scenarios"].items()
    ]
    representative_id = analysis_config["representative_product_id"]
    representative = next(
        product for product in products if product["product_id"] == representative_id
    )
    representative_coverage = [
        row for row in coverage_rows if row["product_id"] == representative_id
    ]
    representative_base = next(
        row for row in representative_coverage if row["coverage_scenario_id"] == "base"
    )
    representative_summary = {
        **representative_base,
        **{
            f"raw_{row['benefit_id']}_claim_change_krw": row["claim_change_krw"]
            for row in representative["claims"]["claim_rows"]
        },
        "representative_issue_age": analysis_config["representative_issue_age"],
        "male_weight": analysis_config["sex_weights"]["남"],
        "female_weight": analysis_config["sex_weights"]["여"],
        "cohort_label": analysis_config["cohort_label"],
    }

    sensitivity_rows = []
    for scenario_id, scenario in config["coverage_scenarios"].items():
        holding_rates = {key: float(scenario[key]) for key in BENEFIT_IDS}
        for multiplier in config["benefit_amount_multipliers"]:
            adjustment = coverage_adjustment(
                representative["claims"]["claim_rows"],
                holding_rates,
                float(multiplier),
            )
            total = representative["rider"]["profit_pv_krw"] + adjustment[
                "existing_claims_adjustment_krw"
            ]
            sensitivity_rows.append(
                {
                    "product_id": representative_id,
                    "coverage_scenario_id": scenario_id,
                    "coverage_scenario_name_ko": scenario["display_name_ko"],
                    "benefit_amount_multiplier": float(multiplier),
                    "rider_npv_krw": representative["rider"]["profit_pv_krw"],
                    "existing_claims_adjustment_krw": adjustment[
                        "existing_claims_adjustment_krw"
                    ],
                    "insurer_total_impact_krw": total,
                    "insurer_total_impact_per_contract_krw": total
                    / float(analysis_config["initial_contracts"]),
                }
            )

    validation = _validation(
        config=config,
        products=products,
        coverage_rows=coverage_rows,
        representative_id=representative_id,
    )
    if validation["status"] != "PASS":
        failed = [row["check"] for row in validation["checks"] if not row["passed"]]
        raise AssertionError(f"보험사 종합영향 검증 실패: {failed}")
    return {
        "config": config,
        "products": products,
        "representative_summary": representative_summary,
        "representative_coverage_rows": representative_coverage,
        "coverage_rows": coverage_rows,
        "sensitivity_rows": sensitivity_rows,
        "representative_start_paths": representative["claims"]["daily_rows"],
        "validation": validation,
        "treatment_validation": treatment_validation,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path.name}에 저장할 행이 없습니다.")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(
    result: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "representative_product_summary.csv",
        [result["representative_summary"]],
    )
    _write_csv(
        output / "coverage_scenario_results.csv",
        result["representative_coverage_rows"],
    )
    _write_csv(output / "eight_product_results.csv", result["coverage_rows"])
    _write_csv(
        output / "benefit_amount_sensitivity.csv", result["sensitivity_rows"]
    )
    _write_csv(
        output / "representative_start_paths.csv",
        result["representative_start_paths"],
    )
    (output / "validation.json").write_text(
        json.dumps(_jsonable(result["validation"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "analysis_name_ko": result["config"]["analysis_name_ko"],
        "comparison_basis": "rider introduction A vs A+B",
        "representative_product": result["representative_summary"],
        "coverage_scenarios": result["representative_coverage_rows"],
        "validation": result["validation"],
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = build_analysis(args.config)
    write_analysis(result, args.output)
    print(
        json.dumps(
            {
                "status": result["validation"]["status"],
                "representative_product": result["representative_summary"],
                "coverage_scenarios": result["representative_coverage_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
