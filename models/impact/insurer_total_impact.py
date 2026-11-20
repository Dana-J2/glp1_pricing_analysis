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
from models.rider.cashflows import (
    CashflowSettings,
    project_cashflows,
    summarize,
    truncate_rows_to_policy_term,
)
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
    reference = config["select_reference_analysis"]
    weights = reference["sex_weights"]
    if set(weights) != {"남", "여"} or not math.isclose(
        math.fsum(float(value) for value in weights.values()), 1.0
    ):
        raise ValueError("남녀 가중치는 합계 1이어야 합니다.")
    if int(reference["representative_issue_age"]) != 62:
        raise ValueError("공통 대표 가입연령은 62세여야 합니다.")
    if not math.isclose(float(analysis["target_profit_margin"]), 0.10):
        raise ValueError("이번 분석은 목표이익률 10% 확정 보험료만 사용합니다.")
    scenarios = config["coverage_scenarios"]
    if tuple(scenarios) != ("low", "base", "high"):
        raise ValueError("보유율 시나리오는 low, base, high 순서여야 합니다.")
    for scenario in scenarios.values():
        if any(not 0 <= float(scenario[key]) <= 1 for key in BENEFIT_IDS):
            raise ValueError("담보 보유율은 [0,1]이어야 합니다.")
    enrollment = config["portfolio"]["enrollment_scenarios"]
    if tuple(enrollment) != ("base", "younger_focus", "older_high_risk_focus"):
        raise ValueError("가입구성 시나리오 순서를 확인하세요.")
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


def _build_select_reference(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    analysis_config = config["analysis"]
    reference_config = config["select_reference_analysis"]
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
            issue_age=int(reference_config["representative_issue_age"]),
            sex_weights={
                key: float(value)
                for key, value in reference_config["sex_weights"].items()
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
        "representative_issue_age": reference_config["representative_issue_age"],
        "male_weight": reference_config["sex_weights"]["남"],
        "female_weight": reference_config["sex_weights"]["여"],
        "cohort_label": reference_config["cohort_label"],
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


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with _resolve(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _term_for_age(config: dict[str, Any], age: int) -> int:
    portfolio = config["portfolio"]
    standard = portfolio["standard_term"]
    senior = portfolio["senior_term"]
    if int(standard["minimum_issue_age"]) <= age <= int(standard["maximum_issue_age"]):
        return int(standard["term_years"])
    if int(senior["minimum_issue_age"]) <= age <= int(senior["maximum_issue_age"]):
        return int(senior["term_years"])
    raise ValueError(f"판매대상 밖 가입연령: {age}")


def _selection_band(age: int) -> str:
    if 19 <= age <= 49:
        return "19_49"
    if 50 <= age <= 59:
        return "50_59"
    if 60 <= age <= 74:
        return "60_74"
    raise ValueError(f"선택계수 연령구간 밖: {age}")


def _load_population_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_csv_rows(config["inputs"]["population_age_sex"])
    cells: list[dict[str, Any]] = []
    for row in rows:
        age = int(row["age"])
        if not 19 <= age <= 74:
            continue
        sex = row["sex"]
        if sex not in {"남", "여"}:
            raise ValueError(f"알 수 없는 성별: {sex}")
        cells.append(
            {
                "issue_age": age,
                "sex": sex,
                "population": float(row["population"]),
                "reference_period": row["reference_period"],
                "population_source_id": row["source_id"],
                "population_evidence_type": row["evidence_type"],
                "term_years": _term_for_age(config, age),
            }
        )
    keys = {(row["issue_age"], row["sex"]) for row in cells}
    expected = {(age, sex) for age in range(19, 75) for sex in ("남", "여")}
    if len(cells) != 112 or keys != expected:
        raise ValueError("KOSIS 인구자료는 19~74세×남녀 112개 고유 셀이어야 합니다.")
    return sorted(cells, key=lambda row: (row["issue_age"], row["sex"]))


def _portfolio_composition_rows(
    config: dict[str, Any], population_cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    portfolio = config["portfolio"]
    factors = portfolio["eligibility_factors"]
    eligible = float(factors["eligible_proxy_rate"])
    nondiabetic = float(factors["nondiabetic_fraction_within_eligible"])
    unused = float(factors["unused_fraction"])
    high_risk = portfolio["select_like_high_risk_share"]
    initial = float(config["analysis"]["initial_contracts"])
    output: list[dict[str, Any]] = []
    for scenario_id, scenario in portfolio["enrollment_scenarios"].items():
        coefficients = scenario["selection_coefficients"]
        raw = []
        for cell in population_cells:
            band = _selection_band(int(cell["issue_age"]))
            coefficient = float(coefficients[band])
            unnormalized = (
                float(cell["population"])
                * eligible
                * nondiabetic
                * unused
                * coefficient
            )
            raw.append((cell, band, coefficient, unnormalized))
        denominator = math.fsum(item[3] for item in raw)
        if denominator <= 0:
            raise ValueError(f"{scenario_id} 포트폴리오 가중치 분모가 0 이하입니다.")
        for cell, band, coefficient, unnormalized in raw:
            weight = unnormalized / denominator
            age = int(cell["issue_age"])
            select_share = (
                float(high_risk["share"])
                if int(high_risk["minimum_applicable_age"])
                <= age
                <= int(high_risk["maximum_applicable_age"])
                else 0.0
            )
            output.append(
                {
                    "enrollment_scenario_id": scenario_id,
                    "enrollment_scenario_name_ko": scenario["display_name_ko"],
                    "issue_age": age,
                    "sex": cell["sex"],
                    "age_band": band.replace("_", "~"),
                    "term_years": cell["term_years"],
                    "population": cell["population"],
                    "eligible_proxy_rate": eligible,
                    "nondiabetic_fraction_within_eligible": nondiabetic,
                    "unused_fraction": unused,
                    "selection_coefficient": coefficient,
                    "unnormalized_weight": unnormalized,
                    "portfolio_weight": weight,
                    "portfolio_contracts": weight * initial,
                    "select_like_high_risk_share": select_share,
                    "select_like_general_population_share": float(
                        high_risk["general_population_share"]
                    ),
                    "select_like_share_basis": high_risk["evidence_type"],
                    "select_like_contracts": weight * initial * select_share,
                    "population_source_id": cell["population_source_id"],
                    "population_evidence_type": cell["population_evidence_type"],
                    "selection_evidence_type": scenario["evidence_type"],
                    "eligibility_age_sex_granularity": "전체 성인 비율을 모든 연령·성별 셀에 공통 적용",
                }
            )
    return output


def _load_sale_lookup(
    config: dict[str, Any], product_ids: set[str]
) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = _read_csv_rows(config["inputs"]["sale_product_age_lookup"])
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        product_id = row["product_id"]
        if product_id not in product_ids:
            continue
        age = int(row["issue_age"])
        sex = row["sex"]
        if not 19 <= age <= 74 or sex not in {"남", "여"}:
            continue
        if not math.isclose(float(row["target_profit_margin"]), 0.10):
            raise ValueError("판매 조회표는 목표이익률 10%여야 합니다.")
        key = (product_id, age, sex)
        if key in lookup:
            raise ValueError(f"판매보험료 조회표 중복: {key}")
        term = int(row["term_years"])
        expected_term = _term_for_age(config, age)
        if term != expected_term:
            raise ValueError(f"보험기간 불일치: {key}, {term} != {expected_term}")
        lookup[key] = {
            "product_id": product_id,
            "issue_age": age,
            "sex": sex,
            "term_years": term,
            "fixed_monthly_premium_krw": float(
                row["monthly_gross_premium_rounded_up_krw"]
            ),
            "monthly_gross_premium_unrounded_krw": float(
                row["monthly_gross_premium_unrounded_krw"]
            ),
            "lookup_product_type": row["product_type"],
        }
    expected = {
        (product_id, age, sex)
        for product_id in product_ids
        for age in range(19, 75)
        for sex in ("남", "여")
    }
    if set(lookup) != expected or len(lookup) != 896:
        missing = sorted(expected - set(lookup))[:5]
        raise ValueError(f"8개 상품×56세×남녀 판매보험료가 필요합니다: missing={missing}")
    return lookup


def _build_age_sex_rider_rows(
    *,
    config: dict[str, Any],
    designs: list[dict[str, Any]],
    sale_lookup: dict[tuple[str, int, str], dict[str, Any]],
    population_cells: list[dict[str, Any]],
    templates: dict[str, Any],
    mortality: dict[tuple[int, str], float],
    treatment: Parameters,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, list[dict[str, Any]]]]]:
    rows: list[dict[str, Any]] = []
    term_start_rows: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for design in designs:
        product_id = design["design_id"]
        bundle = generate_start_rows(templates[config["analysis"]["start_scenario"]], design)
        term_rows = {
            10: truncate_rows_to_policy_term(bundle["rows"], 10),
            5: truncate_rows_to_policy_term(bundle["rows"], 5),
        }
        term_start_rows[product_id] = term_rows
        for cell in population_cells:
            age = int(cell["issue_age"])
            sex = cell["sex"]
            sale = sale_lookup[(product_id, age, sex)]
            term = int(sale["term_years"])
            settings = CashflowSettings(
                **{**design["settings_config"], "entry_age": age, "sex": sex}
            )
            projected = project_cashflows(
                term_rows[term], mortality, settings, treatment, path="yes"
            )
            evaluation = evaluate_projected_cashflows(
                projected, settings, sale["fixed_monthly_premium_krw"]
            )
            summary = summarize(projected, settings.initial_policies)
            rows.append(
                {
                    "product_id": product_id,
                    "benefit_per_dispense_krw": int(design["benefit"]),
                    "max_claims": int(design["max_claims"]),
                    "waiting_days": int(design["waiting_days"]),
                    "issue_age": age,
                    "sex": sex,
                    "term_years": term,
                    "fixed_monthly_premium_krw": sale["fixed_monthly_premium_krw"],
                    "pricing_source": "data/derived/sale_product_age_lookup.csv",
                    "rider_premium_income_pv_per_1000_krw": evaluation[
                        "premium_income_pv_krw"
                    ],
                    "rider_benefit_pv_per_1000_krw": evaluation["benefit_pv_krw"],
                    "rider_expenses_pv_per_1000_krw": evaluation["expenses_pv_krw"],
                    "rider_refund_pv_per_1000_krw": evaluation["refund_pv_krw"],
                    "rider_total_outflow_pv_per_1000_krw": evaluation[
                        "total_outflow_pv_krw"
                    ],
                    "rider_npv_per_1000_krw": evaluation["profit_pv_krw"],
                    "rider_profit_margin": evaluation["profit_margin"],
                    "expected_starts_per_1000": summary["total_starts"],
                    "expected_paid_claims_per_1000": summary["total_paid_claims"],
                    "expected_recipients_per_1000": summary["recipients"],
                    "expected_exhausted_per_1000": summary["exhausted"],
                    "cohort_basis": "동일 연령·성별 최초 1,000계약",
                }
            )
    return rows, term_start_rows


def _aggregate_rider_portfolios(
    *,
    config: dict[str, Any],
    age_sex_rows: list[dict[str, Any]],
    composition_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cell_index = {
        (row["product_id"], row["issue_age"], row["sex"]): row
        for row in age_sex_rows
    }
    scenarios = config["portfolio"]["enrollment_scenarios"]
    product_ids = sorted({row["product_id"] for row in age_sex_rows})
    value_fields = (
        "rider_premium_income_pv_per_1000_krw",
        "rider_benefit_pv_per_1000_krw",
        "rider_expenses_pv_per_1000_krw",
        "rider_refund_pv_per_1000_krw",
        "rider_total_outflow_pv_per_1000_krw",
        "rider_npv_per_1000_krw",
        "expected_starts_per_1000",
        "expected_paid_claims_per_1000",
        "expected_recipients_per_1000",
        "expected_exhausted_per_1000",
    )
    output = []
    for product_id in product_ids:
        product_example = next(row for row in age_sex_rows if row["product_id"] == product_id)
        for scenario_id, scenario in scenarios.items():
            weights = [
                row
                for row in composition_rows
                if row["enrollment_scenario_id"] == scenario_id
            ]
            result = {
                "product_id": product_id,
                "benefit_per_dispense_krw": product_example[
                    "benefit_per_dispense_krw"
                ],
                "max_claims": product_example["max_claims"],
                "waiting_days": product_example["waiting_days"],
                "enrollment_scenario_id": scenario_id,
                "enrollment_scenario_name_ko": scenario["display_name_ko"],
                "weighted_average_monthly_premium_krw": math.fsum(
                    row["portfolio_weight"]
                    * cell_index[(product_id, row["issue_age"], row["sex"])][
                        "fixed_monthly_premium_krw"
                    ]
                    for row in weights
                ),
                "pricing_calculation_order": "연령·성별 셀별 현금흐름 계산 후 가중합",
                "cohort_basis": "포트폴리오 최초 1,000계약",
            }
            for field in value_fields:
                result[field] = math.fsum(
                    row["portfolio_weight"]
                    * cell_index[(product_id, row["issue_age"], row["sex"])][field]
                    for row in weights
                )
            result["rider_profit_margin"] = (
                result["rider_npv_per_1000_krw"]
                / result["rider_premium_income_pv_per_1000_krw"]
            )
            output.append(result)
    return output


def _existing_claim_change_by_product_term(
    *,
    config: dict[str, Any],
    term_start_rows: dict[str, dict[int, list[dict[str, Any]]]],
) -> tuple[dict[tuple[str, int, str], float], list[dict[str, Any]]]:
    existing_config = load_existing_claims_config(
        _resolve(config["inputs"]["existing_claims_config"])
    )
    existing_config["analysis"]["initial_cohort"] = float(
        config["analysis"]["initial_contracts"]
    )
    treatment = params_from_json(ROOT / existing_config["treatment"]["params_path"])
    parameters = existing_with_overrides(
        existing_base_parameters(existing_config),
        {
            "holding_rates": {key: 1.0 for key in BENEFIT_IDS},
            "benefit_amounts": {
                key: float(config["benefit_amounts_krw"][key]) for key in BENEFIT_IDS
            },
        },
    )
    initial = float(config["analysis"]["initial_contracts"])
    index: dict[tuple[str, int, str], float] = {}
    rows: list[dict[str, Any]] = []
    for product_id, by_term in term_start_rows.items():
        for term, start_rows in by_term.items():
            on_fraction, _ = _treatment_on_fraction(len(start_rows), treatment)
            result = _evaluate_existing_claims(
                existing_config=existing_config,
                start_rows=start_rows,
                on_fraction=on_fraction,
                parameters=parameters,
            )
            for claim in result["claim_rows"]:
                benefit_id = claim["benefit_id"]
                per_contract = float(claim["claim_change_krw"]) / initial
                index[(product_id, term, benefit_id)] = per_contract
                rows.append(
                    {
                        "product_id": product_id,
                        "term_years": term,
                        "benefit_id": benefit_id,
                        "benefit_name_ko": claim["benefit_name_ko"],
                        "without_rider_claim_pv_per_select_contract_krw": float(
                            claim["without_rider_claim_pv_krw"]
                        )
                        / initial,
                        "with_rider_claim_pv_per_select_contract_krw": float(
                            claim["with_rider_claim_pv_krw"]
                        )
                        / initial,
                        "claim_change_per_select_contract_krw": per_contract,
                        "clinical_event_rate_age_assumption": "SELECT 전체 사건률을 동일 보험기간 셀에 공통 적용",
                    }
                )
    return index, rows


def _aggregate_existing_claims(
    *,
    config: dict[str, Any],
    composition_rows: list[dict[str, Any]],
    claim_index: dict[tuple[str, int, str], float],
    product_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_rows: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    for product_id in product_ids:
        for scenario_id, scenario in config["portfolio"]["enrollment_scenarios"].items():
            composition = [
                row
                for row in composition_rows
                if row["enrollment_scenario_id"] == scenario_id
            ]
            raw_by_benefit = {
                benefit_id: math.fsum(
                    row["select_like_contracts"]
                    * claim_index[(product_id, int(row["term_years"]), benefit_id)]
                    for row in composition
                )
                for benefit_id in BENEFIT_IDS
            }
            select_contracts = math.fsum(row["select_like_contracts"] for row in composition)
            for coverage_id, coverage in config["coverage_scenarios"].items():
                components = {}
                for benefit_id in BENEFIT_IDS:
                    holding = float(coverage[benefit_id])
                    adjustment = -raw_by_benefit[benefit_id] * holding
                    components[benefit_id] = adjustment
                    component_rows.append(
                        {
                            "product_id": product_id,
                            "enrollment_scenario_id": scenario_id,
                            "enrollment_scenario_name_ko": scenario[
                                "display_name_ko"
                            ],
                            "coverage_scenario_id": coverage_id,
                            "coverage_scenario_name_ko": coverage["display_name_ko"],
                            "benefit_id": benefit_id,
                            "benefit_name_ko": DISPLAY_NAMES[benefit_id],
                            "select_like_contracts": select_contracts,
                            "holding_rate": holding,
                            "raw_claim_change_krw": raw_by_benefit[benefit_id],
                            "insurer_existing_claims_adjustment_krw": adjustment,
                            "cohort_basis": "포트폴리오 최초 1,000계약",
                        }
                    )
                totals.append(
                    {
                        "product_id": product_id,
                        "enrollment_scenario_id": scenario_id,
                        "coverage_scenario_id": coverage_id,
                        "select_like_contracts": select_contracts,
                        **{
                            f"{benefit_id}_existing_claims_adjustment_krw": components[
                                benefit_id
                            ]
                            for benefit_id in BENEFIT_IDS
                        },
                        "existing_claims_adjustment_krw": math.fsum(
                            components.values()
                        ),
                    }
                )
    return component_rows, totals


def _combine_portfolio_results(
    *,
    config: dict[str, Any],
    rider_rows: list[dict[str, Any]],
    existing_totals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = {
        (row["product_id"], row["enrollment_scenario_id"], row["coverage_scenario_id"]): row
        for row in existing_totals
    }
    output = []
    for rider in rider_rows:
        for coverage_id, coverage in config["coverage_scenarios"].items():
            key = (rider["product_id"], rider["enrollment_scenario_id"], coverage_id)
            claim = existing[key]
            rider_npv = float(rider["rider_npv_per_1000_krw"])
            adjustment = float(claim["existing_claims_adjustment_krw"])
            output.append(
                {
                    "product_id": rider["product_id"],
                    "benefit_per_dispense_krw": rider["benefit_per_dispense_krw"],
                    "max_claims": rider["max_claims"],
                    "waiting_days": rider["waiting_days"],
                    "enrollment_scenario_id": rider["enrollment_scenario_id"],
                    "enrollment_scenario_name_ko": rider[
                        "enrollment_scenario_name_ko"
                    ],
                    "coverage_scenario_id": coverage_id,
                    "coverage_scenario_name_ko": coverage["display_name_ko"],
                    "weighted_average_monthly_premium_krw": rider[
                        "weighted_average_monthly_premium_krw"
                    ],
                    "rider_npv_krw": rider_npv,
                    **{
                        f"{benefit_id}_existing_claims_adjustment_krw": claim[
                            f"{benefit_id}_existing_claims_adjustment_krw"
                        ]
                        for benefit_id in BENEFIT_IDS
                    },
                    "existing_claims_adjustment_krw": adjustment,
                    "insurer_total_impact_krw": rider_npv + adjustment,
                    "rider_npv_per_contract_krw": rider_npv
                    / float(config["analysis"]["initial_contracts"]),
                    "existing_claims_adjustment_per_contract_krw": adjustment
                    / float(config["analysis"]["initial_contracts"]),
                    "insurer_total_impact_per_contract_krw": (rider_npv + adjustment)
                    / float(config["analysis"]["initial_contracts"]),
                    "comparison_basis": "특약 없음 A vs 특약 있음 A+B",
                    "cohort_basis": "포트폴리오 최초 1,000계약",
                }
            )
    return output


def _portfolio_validation(
    *,
    config: dict[str, Any],
    composition_rows: list[dict[str, Any]],
    sale_lookup: dict[tuple[str, int, str], dict[str, Any]],
    age_sex_rows: list[dict[str, Any]],
    rider_rows: list[dict[str, Any]],
    total_rows: list[dict[str, Any]],
    select_reference: dict[str, Any],
    claim_index: dict[tuple[str, int, str], float],
    product_ids: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    sums = {
        scenario_id: math.fsum(
            row["portfolio_weight"]
            for row in composition_rows
            if row["enrollment_scenario_id"] == scenario_id
        )
        for scenario_id in config["portfolio"]["enrollment_scenarios"]
    }
    add(
        "portfolio_weights_sum_to_one",
        all(math.isclose(value, 1.0, abs_tol=1e-12) for value in sums.values()),
        str(sums),
    )
    add("all_896_sale_prices_present", len(sale_lookup) == 896)
    add(
        "term_by_issue_age",
        all(
            row["term_years"] == (10 if row["issue_age"] <= 69 else 5)
            for row in age_sex_rows
        ),
    )
    add(
        "portfolio_uses_age_sex_sale_lookup_not_fixed_49f_price",
        all(
            row["pricing_source"] == "data/derived/sale_product_age_lookup.csv"
            and math.isclose(
                row["fixed_monthly_premium_krw"],
                sale_lookup[(row["product_id"], row["issue_age"], row["sex"])][
                    "fixed_monthly_premium_krw"
                ],
            )
            for row in age_sex_rows
        ),
    )
    cell_index = {
        (row["product_id"], row["issue_age"], row["sex"]): row
        for row in age_sex_rows
    }
    max_weight_error = 0.0
    for aggregate in rider_rows:
        weighted = math.fsum(
            row["portfolio_weight"]
            * cell_index[
                (aggregate["product_id"], row["issue_age"], row["sex"])
            ]["rider_npv_per_1000_krw"]
            for row in composition_rows
            if row["enrollment_scenario_id"]
            == aggregate["enrollment_scenario_id"]
        )
        max_weight_error = max(
            max_weight_error, abs(weighted - aggregate["rider_npv_per_1000_krw"])
        )
    add(
        "cell_weighted_sum_equals_portfolio_rider_npv",
        max_weight_error < 1e-6,
        f"max_error={max_weight_error:.3g}",
    )
    max_total_error = max(
        abs(
            row["insurer_total_impact_krw"]
            - row["rider_npv_krw"]
            - row["existing_claims_adjustment_krw"]
        )
        for row in total_rows
    )
    add(
        "components_equal_total_impact",
        max_total_error < 1e-6,
        f"max_error={max_total_error:.3g}",
    )

    factors = config["portfolio"]["eligibility_factors"]
    high_risk = config["portfolio"]["select_like_high_risk_share"]
    conditional_denominator = (
        float(factors["eligible_proxy_rate"])
        * float(factors["nondiabetic_fraction_within_eligible"])
        * float(factors["unused_fraction"])
    )
    expected_conditional_share = (
        float(high_risk["general_population_share"]) / conditional_denominator
    )
    add(
        "conditional_select_like_share_formula",
        math.isclose(
            float(high_risk["share"]), expected_conditional_share, abs_tol=1e-15
        )
        and math.isclose(
            float(high_risk["conditioning_denominator_rate"]),
            conditional_denominator,
            abs_tol=1e-15,
        ),
        f"share={float(high_risk['share']):.12f}, denominator={conditional_denominator:.12f}",
    )

    zero_risk_composition = [
        {
            **row,
            "select_like_high_risk_share": 0.0,
            "select_like_contracts": 0.0,
        }
        for row in composition_rows
    ]
    _, zero_risk_totals = _aggregate_existing_claims(
        config=config,
        composition_rows=zero_risk_composition,
        claim_index=claim_index,
        product_ids=product_ids,
    )
    max_zero_risk_adjustment = max(
        abs(float(row["existing_claims_adjustment_krw"]))
        for row in zero_risk_totals
    )
    add(
        "zero_high_risk_share_produces_zero_adjustment",
        max_zero_risk_adjustment < 1e-12,
        f"max_abs_adjustment={max_zero_risk_adjustment:.3g}",
    )

    zero_coverage_config = {
        **config,
        "coverage_scenarios": {
            "zero": {
                "display_name_ko": "보유율 0 검증",
                **{benefit_id: 0.0 for benefit_id in BENEFIT_IDS},
            }
        },
    }
    _, zero_coverage_totals = _aggregate_existing_claims(
        config=zero_coverage_config,
        composition_rows=composition_rows,
        claim_index=claim_index,
        product_ids=product_ids,
    )
    max_zero_coverage_adjustment = max(
        abs(float(row["existing_claims_adjustment_krw"]))
        for row in zero_coverage_totals
    )
    add(
        "zero_holding_rates_produce_zero_adjustment",
        max_zero_coverage_adjustment < 1e-12,
        f"max_abs_adjustment={max_zero_coverage_adjustment:.3g}",
    )
    expected = config["select_reference_analysis"]["expected_regression"]
    observed = select_reference["representative_summary"]
    regression_fields = {
        "representative_fixed_monthly_premium_krw": observed[
            "fixed_monthly_premium_krw"
        ],
        "representative_rider_npv_krw": observed["rider_npv_krw"],
        "representative_base_existing_claims_adjustment_krw": observed[
            "existing_claims_adjustment_krw"
        ],
        "representative_base_total_impact_krw": observed[
            "insurer_total_impact_krw"
        ],
    }
    max_reference_error = max(
        abs(float(regression_fields[key]) - float(expected[key])) for key in expected
    )
    add(
        "select_62_reference_unchanged",
        max_reference_error < 1e-6,
        f"max_error={max_reference_error:.3g}",
    )
    status = "PASS" if all(row["passed"] for row in checks) else "FAIL"
    return {"status": status, "checks": checks}


def build_analysis(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    select_reference = _build_select_reference(config_path)
    mvp_config = load_mvp_config(_resolve(config["inputs"]["mvp_product_config"]))
    cashflow_config = _read_json(_resolve(config["inputs"]["cashflow_config"]))
    designs = build_mvp_designs(mvp_config, cashflow_config)
    product_ids = {design["design_id"] for design in designs}
    population_cells = _load_population_cells(config)
    composition_rows = _portfolio_composition_rows(config, population_cells)
    sale_lookup = _load_sale_lookup(config, product_ids)
    age_sex_rows, term_start_rows = _build_age_sex_rider_rows(
        config=config,
        designs=designs,
        sale_lookup=sale_lookup,
        population_cells=population_cells,
        templates=load_start_templates(),
        mortality=read_mortality(),
        treatment=Parameters(**_read_json(_resolve(config["inputs"]["treatment_config"]))),
    )
    rider_rows = _aggregate_rider_portfolios(
        config=config,
        age_sex_rows=age_sex_rows,
        composition_rows=composition_rows,
    )
    claim_index, select_claim_rows = _existing_claim_change_by_product_term(
        config=config, term_start_rows=term_start_rows
    )
    component_rows, existing_totals = _aggregate_existing_claims(
        config=config,
        composition_rows=composition_rows,
        claim_index=claim_index,
        product_ids=sorted(product_ids),
    )
    total_rows = _combine_portfolio_results(
        config=config, rider_rows=rider_rows, existing_totals=existing_totals
    )
    representative_id = config["analysis"]["representative_product_id"]
    main_result = next(
        row
        for row in total_rows
        if row["product_id"] == representative_id
        and row["enrollment_scenario_id"] == "base"
        and row["coverage_scenario_id"] == "base"
    )
    comparison_rows = [
        {
            "analysis_group": "select_reference",
            "analysis_group_name_ko": "SELECT 유사 대표집단 참고분석",
            "product_id": representative_id,
            "enrollment_scenario_id": "select_age62_72_28",
            "coverage_scenario_id": "base",
            "rider_npv_krw": select_reference["representative_summary"]["rider_npv_krw"],
            "existing_claims_adjustment_krw": select_reference[
                "representative_summary"
            ]["existing_claims_adjustment_krw"],
            "insurer_total_impact_krw": select_reference["representative_summary"][
                "insurer_total_impact_krw"
            ],
            "cohort_basis": "62세, 남 72%·여 28%, 최초 1,000계약",
        }
    ]
    comparison_rows.extend(
        {
            "analysis_group": "portfolio",
            "analysis_group_name_ko": "국내 가입대상 포트폴리오",
            "product_id": row["product_id"],
            "enrollment_scenario_id": row["enrollment_scenario_id"],
            "coverage_scenario_id": row["coverage_scenario_id"],
            "rider_npv_krw": row["rider_npv_krw"],
            "existing_claims_adjustment_krw": row[
                "existing_claims_adjustment_krw"
            ],
            "insurer_total_impact_krw": row["insurer_total_impact_krw"],
            "cohort_basis": "연령·성별 가중 최초 1,000계약",
        }
        for row in total_rows
        if row["product_id"] == representative_id
        and row["coverage_scenario_id"] == "base"
    )
    validation = _portfolio_validation(
        config=config,
        composition_rows=composition_rows,
        sale_lookup=sale_lookup,
        age_sex_rows=age_sex_rows,
        rider_rows=rider_rows,
        total_rows=total_rows,
        select_reference=select_reference,
        claim_index=claim_index,
        product_ids=sorted(product_ids),
    )
    if validation["status"] != "PASS":
        failed = [row["check"] for row in validation["checks"] if not row["passed"]]
        raise AssertionError(f"포트폴리오 종합영향 검증 실패: {failed}")
    return {
        "config": config,
        "select_reference": select_reference,
        "composition_rows": composition_rows,
        "age_sex_rider_rows": age_sex_rows,
        "portfolio_rider_rows": rider_rows,
        "select_claim_rows": select_claim_rows,
        "existing_claim_component_rows": component_rows,
        "portfolio_total_rows": total_rows,
        "comparison_rows": comparison_rows,
        "main_result": main_result,
        "validation": validation,
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
    reference = result["select_reference"]
    # 기존 v1 산출물은 SELECT 유사 대표집단 참고분석으로 그대로 보존한다.
    _write_csv(
        output / "representative_product_summary.csv",
        [reference["representative_summary"]],
    )
    _write_csv(
        output / "coverage_scenario_results.csv",
        reference["representative_coverage_rows"],
    )
    _write_csv(output / "eight_product_results.csv", reference["coverage_rows"])
    _write_csv(
        output / "benefit_amount_sensitivity.csv", reference["sensitivity_rows"]
    )
    _write_csv(
        output / "representative_start_paths.csv",
        reference["representative_start_paths"],
    )
    _write_csv(output / "portfolio_composition.csv", result["composition_rows"])
    _write_csv(output / "age_sex_rider_results.csv", result["age_sex_rider_rows"])
    _write_csv(output / "portfolio_rider_summary.csv", result["portfolio_rider_rows"])
    _write_csv(
        output / "select_claim_change_per_contract.csv", result["select_claim_rows"]
    )
    _write_csv(
        output / "portfolio_existing_claims_components.csv",
        result["existing_claim_component_rows"],
    )
    _write_csv(output / "portfolio_total_impact.csv", result["portfolio_total_rows"])
    _write_csv(
        output / "select_reference_vs_portfolio.csv", result["comparison_rows"]
    )
    (output / "validation.json").write_text(
        json.dumps(_jsonable(result["validation"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "analysis_name_ko": result["config"]["analysis_name_ko"],
        "comparison_basis": "특약 없음 A vs 특약 있음 A+B",
        "main_result_basis": "기준 가입구성 × 기준 담보 보유율",
        "main_result": result["main_result"],
        "select_reference_analysis": reference["representative_summary"],
        "portfolio_scenarios": result["comparison_rows"],
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
                "main_result": result["main_result"],
                "select_reference": result["select_reference"][
                    "representative_summary"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
