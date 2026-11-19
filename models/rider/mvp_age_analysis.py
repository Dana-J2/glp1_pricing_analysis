"""MVP 8개 상품조건과 가입연령별 대표상품 분석."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from models.rider.cashflows import (
    CashflowSettings,
    project_cashflows,
    summarize,
    truncate_rows_to_policy_term,
)
from models.rider.issue_age import (
    load_issue_age_rules,
    treatment_parameters_for_issue_age,
    validate_issue_age,
)
from models.rider.pricing import calculate_premiums
from models.rider.product_comparison import (
    build_full_analysis,
    build_pricing_stage,
    build_product_design,
    generate_start_rows,
    load_start_templates,
)
from models.rider.profit_analysis import evaluate_projected_cashflows
from models.rider.start_integration import ROOT, read_mortality
from models.treatment.model import Parameters


MVP_CONFIG_PATH = ROOT / "config/mvp_product_options.json"
AGE_RULES_PATH = ROOT / "config/issue_age_rules.json"
DERIVED_DIR = ROOT / "data/derived"
REPORT_PATH = ROOT / "reports/mvp_issue_age_validation.json"
SENIOR_REPORT_PATH = ROOT / "reports/senior_extension_5year_validation.json"
OUTPUT_DIR = ROOT / "outputs/mvp_issue_age_analysis"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path.name}에 저장할 결과가 없습니다")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check(checks: list[dict[str, Any]], name: str, condition: bool, detail: str = "") -> None:
    checks.append({"check": name, "passed": bool(condition), "detail": detail})


def load_mvp_config(path: str | Path = MVP_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_mvp_designs(
    config: dict[str, Any] | None = None,
    base_cashflow_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = config or load_mvp_config()
    base_cashflow_config = base_cashflow_config or json.loads(
        (ROOT / "config/cashflows.json").read_text(encoding="utf-8")
    )
    products = config["products"]
    ids = [item["product_id"] for item in products]
    if len(products) != 8 or len(ids) != len(set(ids)):
        raise ValueError("고유한 MVP 상품 8개가 필요합니다")
    designs = []
    for item in products:
        expected_support = int(item["benefit_per_dispense_krw"]) * int(
            item["max_claims"]
        )
        if expected_support != int(item["maximum_support_krw"]):
            raise ValueError(f"{item['product_id']} 최대 지원금 불일치")
        if int(item["decision_horizon_units"]) != int(
            config["decision_horizon_units"]
        ):
            raise ValueError("MVP 체감 계획기간은 모두 13회여야 합니다")
        design = build_product_design(
            benefit=item["benefit_per_dispense_krw"],
            max_claims=item["max_claims"],
            waiting_days=item["waiting_days"],
            decision_horizon_units=item["decision_horizon_units"],
            base_cashflow_config=base_cashflow_config,
            product_id=item["product_id"],
        )
        design["mvp_config"] = item
        designs.append(design)
    return designs


def _treatment_combined_stress_config(
    base_treatment_config: dict[str, Any],
) -> dict[str, Any]:
    scenarios = json.loads(
        (ROOT / "config/treatment_scenarios.json").read_text(encoding="utf-8")
    )["scenarios"]
    item = next(
        row for row in scenarios if row["scenario_id"] == "treatment_combined_stress"
    )
    resolved = dict(base_treatment_config)
    resolved.update(item["overrides"])
    Parameters(**resolved).validate()
    return resolved


def build_mvp_summary(
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """8개 설계를 현재 공통 엔진에서 다시 계산한다."""
    config = config or load_mvp_config()
    base_cashflow = json.loads((ROOT / "config/cashflows.json").read_text())
    base_treatment = json.loads((ROOT / "config/treatment.json").read_text())
    designs = build_mvp_designs(config, base_cashflow)
    templates = load_start_templates()
    mortality = read_mortality()
    treatment = Parameters(**base_treatment)
    treatment_stress = Parameters(**_treatment_combined_stress_config(base_treatment))
    margins = [
        float(config["regression_target_profit_margin"]),
        float(config["secondary_target_profit_margin"]),
        float(config["primary_target_profit_margin"]),
    ]
    rows: list[dict[str, Any]] = []
    caches: dict[str, Any] = {}
    for design in designs:
        bundles = {
            scenario: generate_start_rows(templates[scenario], design)
            for scenario in ("base", "response_high", "combined_stress")
        }
        projected = {
            scenario: project_cashflows(
                bundle["rows"], mortality, design["settings"], treatment
            )
            for scenario, bundle in bundles.items()
        }
        projected["treatment_combined_stress"] = project_cashflows(
            bundles["base"]["rows"], mortality, design["settings"], treatment_stress
        )
        summaries = {
            scenario: summarize(path, design["settings"].initial_policies)
            for scenario, path in projected.items()
        }
        prices = {
            margin: calculate_premiums(summaries["base"], margin)
            for margin in margins
        }
        price10 = prices[float(config["primary_target_profit_margin"])]
        fixed10 = price10["monthly_gross_premium_rounded_up_krw"]
        stress = {
            scenario: evaluate_projected_cashflows(
                projected[scenario], design["settings"], fixed10
            )["profit_pv_krw"]
            for scenario in (
                "response_high",
                "combined_stress",
                "treatment_combined_stress",
            )
        }
        base = summaries["base"]
        initial = design["settings"].initial_policies
        item = design["mvp_config"]
        row = {
            "product_id": design["design_id"],
            "benefit_per_dispense_krw": int(design["benefit"]),
            "max_claims": design["max_claims"],
            "waiting_days": design["waiting_days"],
            "maximum_support_krw": int(design["maximum_support_krw"]),
            "decision_horizon_units": design["decision_horizon_units"],
            "primary_target_profit_margin": config["primary_target_profit_margin"],
            "secondary_target_profit_margin": config["secondary_target_profit_margin"],
            "included_in_mvp": bool(item["included_in_mvp"]),
            "monthly_premium_target_0pct_rounded_krw": prices[0.0][
                "monthly_gross_premium_rounded_up_krw"
            ],
            "monthly_premium_target_5pct_rounded_krw": prices[0.05][
                "monthly_gross_premium_rounded_up_krw"
            ],
            "monthly_premium_target_10pct_rounded_krw": prices[0.10][
                "monthly_gross_premium_rounded_up_krw"
            ],
            "expected_benefit_pv_per_contract_krw": base["pv_benefit_krw"] / initial,
            "expected_paid_claims_per_contract": base["total_paid_claims"] / initial,
            "recipient_ratio_percent": 100.0 * base["recipients"] / initial,
            "exhaustion_ratio_percent": 100.0 * base["exhausted"] / initial,
            "response_high_npv_target_10pct_per_1000_krw": stress["response_high"],
            "start_combined_stress_npv_target_10pct_per_1000_krw": stress[
                "combined_stress"
            ],
            "treatment_combined_stress_npv_target_10pct_per_1000_krw": stress[
                "treatment_combined_stress"
            ],
            "risk_warning": item.get("risk_warning", ""),
            "cohort_basis": "initial 1000 policies",
            "source_27_condition_match": "not_checked",
        }
        rows.append(row)
        caches[design["design_id"]] = {
            "design": design,
            "bundles": bundles,
            "projected": projected,
            "summaries": summaries,
            "prices": prices,
        }
    return rows, {"designs": designs, "caches": caches}


def _read_team_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["지원금_원"]),
        int(row["최대지급횟수"]),
        int(row["대기기간_일"]),
    )


def validate_team_27_sources(
    pricing_path: Path,
    stress_path: Path,
    mvp_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pricing = _read_team_csv(pricing_path)
    stress = _read_team_csv(stress_path)
    checks: list[dict[str, Any]] = []
    expected_keys = {
        (benefit, cap, wait)
        for benefit in (100000, 150000, 200000)
        for cap in (6, 13, 26)
        for wait in (30, 90, 180)
    }
    pricing_keys = [_source_key(row) for row in pricing]
    stress_keys = [_source_key(row) for row in stress]
    _check(
        checks,
        "27개 가격·혜택 조건 완전성",
        len(pricing) == 27 and set(pricing_keys) == expected_keys and len(set(pricing_keys)) == 27,
    )
    _check(
        checks,
        "27개 스트레스 조건 완전성",
        len(stress) == 27 and set(stress_keys) == expected_keys and len(set(stress_keys)) == 27,
    )
    _check(
        checks,
        "팀원 CSV 결측 없음",
        all(value not in (None, "") for row in pricing + stress for value in row.values()),
    )

    pricing_stage = build_pricing_stage()
    full_stage = build_full_analysis(pricing_stage)
    github_price = {
        (
            int(row["benefit_per_dispense_krw"]),
            int(row["max_claims"]),
            int(row["waiting_days"]),
            float(row["target_profit_margin"]),
        ): row
        for row in pricing_stage["pricing_rows"]
    }
    github_decision = {
        (
            int(next(d["benefit"] for d in pricing_stage["designs"] if d["design_id"] == row["design_id"])),
            int(next(d["max_claims"] for d in pricing_stage["designs"] if d["design_id"] == row["design_id"])),
            90,
        ): row
        for row in full_stage["decision_summary"]
        if math.isclose(row["target_profit_margin"], 0.10)
    }
    price90_ok = True
    stress90_ok = True
    for row in pricing:
        key = _source_key(row)
        if key[2] != 90:
            continue
        for margin, column in ((0.05, "월보험료_목표5_원"), (0.10, "월보험료_목표10_원")):
            current = github_price[(key[0], key[1], key[2], margin)]
            price90_ok &= int(row[column]) == int(
                current["monthly_gross_premium_rounded_up_krw"]
            )
        current = github_price[(key[0], key[1], key[2], 0.10)]
        price90_ok &= abs(
            float(row["계약당_기대보험금PV_원"])
            - current["expected_benefit_pv_per_initial_policy_krw"]
        ) < 0.01
        price90_ok &= abs(
            float(row["수급률_pct"]) - 100 * current["recipient_ratio_initial"]
        ) < 0.0005
        price90_ok &= abs(
            float(row["한도소진율_pct"]) - 100 * current["exhaustion_ratio_initial"]
        ) < 0.0005
    for row in stress:
        key = _source_key(row)
        if key[2] != 90:
            continue
        current = github_decision[key]
        stress90_ok &= abs(
            float(row["높은지원반응_NPV_만원"])
            - current["response_high_profit_pv_krw"] / 10000.0
        ) < 0.005
        stress90_ok &= abs(
            float(row["시작률복합_NPV_만원"])
            - current["start_combined_stress_profit_pv_krw"] / 10000.0
        ) < 0.005
        stress90_ok &= abs(
            float(row["치료행태복합_NPV_만원"])
            - current["treatment_combined_stress_profit_pv_krw"] / 10000.0
        ) < 0.005
    _check(checks, "기존 90일 상품 9개 가격·혜택 일치", price90_ok)
    _check(checks, "기존 90일 상품 9개 스트레스 일치", stress90_ok)

    mvp_keys = {
        (
            int(row["benefit_per_dispense_krw"]),
            int(row["max_claims"]),
            int(row["waiting_days"]),
        )
        for row in mvp_rows
    }
    source_pricing = {_source_key(row): row for row in pricing if _source_key(row) in mvp_keys}
    source_stress = {_source_key(row): row for row in stress if _source_key(row) in mvp_keys}
    _check(
        checks,
        "MVP 8개 정확 필터",
        len(mvp_keys) == len(source_pricing) == len(source_stress) == 8,
    )
    mvp_match = True
    for row in mvp_rows:
        row_match = True
        key = (
            int(row["benefit_per_dispense_krw"]),
            int(row["max_claims"]),
            int(row["waiting_days"]),
        )
        p = source_pricing[key]
        s = source_stress[key]
        row_match &= all(
            int(row[field]) == int(p[source_field])
            for field, source_field in (
                ("monthly_premium_target_0pct_rounded_krw", "월보험료_목표0_원"),
                ("monthly_premium_target_5pct_rounded_krw", "월보험료_목표5_원"),
                ("monthly_premium_target_10pct_rounded_krw", "월보험료_목표10_원"),
            )
        )
        row_match &= abs(
            row["expected_benefit_pv_per_contract_krw"]
            - float(p["계약당_기대보험금PV_원"])
        ) < 0.01
        row_match &= abs(
            row["recipient_ratio_percent"] - float(p["수급률_pct"])
        ) < 0.0005
        row_match &= abs(
            row["exhaustion_ratio_percent"] - float(p["한도소진율_pct"])
        ) < 0.0005
        for field, source_field in (
            ("response_high_npv_target_10pct_per_1000_krw", "높은지원반응_NPV_만원"),
            ("start_combined_stress_npv_target_10pct_per_1000_krw", "시작률복합_NPV_만원"),
            ("treatment_combined_stress_npv_target_10pct_per_1000_krw", "치료행태복합_NPV_만원"),
        ):
            row_match &= abs(row[field] / 10000.0 - float(s[source_field])) < 0.005
        mvp_match &= row_match
        row["source_27_condition_match"] = "PASS" if row_match else "FAIL"
    _check(checks, "현재 엔진의 MVP 8개와 팀원 표 일치", mvp_match)

    baseline = next(
        row for row in mvp_rows if row["product_id"] == "benefit100k_max13_wait90"
    )
    _check(
        checks,
        "기준상품 0%·5%·10% 보험료 재현",
        [
            baseline["monthly_premium_target_0pct_rounded_krw"],
            baseline["monthly_premium_target_5pct_rounded_krw"],
            baseline["monthly_premium_target_10pct_rounded_krw"],
        ]
        == [2808, 2965, 3141],
    )
    return {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
        "sources": {
            "pricing_benefit_summary": {
                "file_name": pricing_path.name,
                "sha256": hashlib.sha256(pricing_path.read_bytes()).hexdigest(),
            },
            "target_10pct_stress_summary": {
                "file_name": stress_path.name,
                "sha256": hashlib.sha256(stress_path.read_bytes()).hexdigest(),
            },
        },
    }


def build_age_analysis(
    *, use_age_specific_discontinuation: bool
) -> list[dict[str, Any]]:
    rules = load_issue_age_rules()
    base_cashflow = json.loads((ROOT / "config/cashflows.json").read_text())
    base_treatment = json.loads((ROOT / "config/treatment.json").read_text())
    representative = rules["representative_product"]
    design = build_product_design(
        benefit=representative["benefit_per_dispense_krw"],
        max_claims=representative["max_claims"],
        waiting_days=representative["waiting_days"],
        decision_horizon_units=representative["decision_horizon_units"],
        base_cashflow_config=base_cashflow,
        product_id=representative["product_id"],
    )
    templates = load_start_templates()
    bundles = {
        scenario: generate_start_rows(templates[scenario], design)
        for scenario in ("base", "response_high", "combined_stress")
    }
    treatment_stress_overrides = next(
        row["overrides"]
        for row in json.loads(
            (ROOT / "config/treatment_scenarios.json").read_text()
        )["scenarios"]
        if row["scenario_id"] == "treatment_combined_stress"
    )
    mortality = read_mortality()
    results: list[dict[str, Any]] = []
    for issue_age in rules["evaluation_issue_ages"]:
        policy = validate_issue_age(issue_age, rules=rules)
        for sex in rules["sexes"]:
            settings = CashflowSettings(
                **{**design["settings_config"], "entry_age": issue_age, "sex": sex}
            )
            treatment, discontinuation = treatment_parameters_for_issue_age(
                issue_age,
                base_treatment,
                use_age_specific_discontinuation=use_age_specific_discontinuation,
                rules=rules,
            )
            treatment_stress = replace(treatment, **treatment_stress_overrides)
            projections = {
                "base": project_cashflows(
                    bundles["base"]["rows"], mortality, settings, treatment
                ),
                "response_high": project_cashflows(
                    bundles["response_high"]["rows"], mortality, settings, treatment
                ),
                "combined_stress": project_cashflows(
                    bundles["combined_stress"]["rows"], mortality, settings, treatment
                ),
                "treatment_combined_stress": project_cashflows(
                    bundles["base"]["rows"], mortality, settings, treatment_stress
                ),
            }
            summaries = {
                scenario: summarize(rows, settings.initial_policies)
                for scenario, rows in projections.items()
            }
            base_summary = summaries["base"]
            for margin in representative["research_target_profit_margins"]:
                price = calculate_premiums(base_summary, float(margin))
                fixed = price["monthly_gross_premium_rounded_up_krw"]
                evaluation = {
                    scenario: evaluate_projected_cashflows(
                        projections[scenario], settings, fixed
                    )
                    for scenario in projections
                }
                initial = settings.initial_policies
                results.append(
                    {
                        "product_id": design["design_id"],
                        "issue_age": issue_age,
                        "sex": sex,
                        "term_years": policy["term_years"],
                        "attained_age": policy["attained_age"],
                        "issue_age_mode": policy["status"],
                        "product_type": policy["status"],
                        "status": policy["status"],
                        "target_profit_margin": float(margin),
                        "monthly_net_premium_krw": price["monthly_net_premium_krw"],
                        "monthly_gross_premium_unrounded_krw": price[
                            "monthly_gross_premium_krw"
                        ],
                        "monthly_gross_premium_rounded_up_krw": fixed,
                        "unrounded_achieved_profit_margin": price[
                            "achieved_profit_margin"
                        ],
                        "rounded_achieved_profit_margin": price[
                            "rounded_achieved_profit_margin"
                        ],
                        "expected_benefit_pv_per_contract_krw": base_summary[
                            "pv_benefit_krw"
                        ]
                        / initial,
                        "expected_paid_claims_per_contract": base_summary[
                            "total_paid_claims"
                        ]
                        / initial,
                        "recipient_ratio": base_summary["recipients"] / initial,
                        "exhaustion_ratio": base_summary["exhausted"] / initial,
                        "expected_deaths_per_1000_contracts": base_summary["deaths"],
                        "base_npv_per_contract_krw": evaluation["base"][
                            "profit_pv_krw"
                        ]
                        / initial,
                        "base_npv_per_1000_contracts_krw": evaluation["base"][
                            "profit_pv_krw"
                        ],
                        "response_high_npv_per_contract_krw": evaluation[
                            "response_high"
                        ]["profit_pv_krw"]
                        / initial,
                        "response_high_npv_per_1000_contracts_krw": evaluation[
                            "response_high"
                        ]["profit_pv_krw"],
                        "start_combined_stress_npv_per_contract_krw": evaluation[
                            "combined_stress"
                        ]["profit_pv_krw"]
                        / initial,
                        "start_combined_stress_npv_per_1000_contracts_krw": evaluation[
                            "combined_stress"
                        ]["profit_pv_krw"],
                        "treatment_combined_stress_npv_per_contract_krw": evaluation[
                            "treatment_combined_stress"
                        ]["profit_pv_krw"]
                        / initial,
                        "treatment_combined_stress_npv_per_1000_contracts_krw": evaluation[
                            "treatment_combined_stress"
                        ]["profit_pv_krw"],
                        "discontinuation_method": discontinuation["method"],
                        "annual_discontinuation_probability": discontinuation[
                            "annual_discontinuation_probability"
                        ],
                        "age_specific_discontinuation_enabled": discontinuation[
                            "age_specific_option_enabled"
                        ],
                        "start_rate_age_assumption": "common_by_age",
                        "restart_rate_age_assumption": "common_by_age",
                        "sex_difference_scope": "mortality_only",
                        "lapse_rate": settings.monthly_lapse,
                        "cohort_basis": "initial 1000 policies",
                    }
                )
    return results


def build_senior_extension_analysis(
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """MVP 8개×70~74세×남녀의 실제 5년 판매가격과 고정가격 수지."""
    config = config or load_mvp_config()
    rules = load_issue_age_rules()
    sale_margin = float(rules["sale_target_profit_margin"])
    if not math.isclose(sale_margin, float(config["sale_target_profit_margin"])):
        raise ValueError("가입연령·MVP 설정의 판매 목표이익률이 다릅니다")
    base_cashflow = json.loads((ROOT / "config/cashflows.json").read_text())
    base_treatment = json.loads((ROOT / "config/treatment.json").read_text())
    treatment = Parameters(**base_treatment)
    treatment_stress = Parameters(**_treatment_combined_stress_config(base_treatment))
    mortality = read_mortality()
    templates = load_start_templates()
    senior = rules["senior_extension_product"]
    term_years = int(senior["term_years"])
    results: list[dict[str, Any]] = []

    for design in build_mvp_designs(config, base_cashflow):
        bundles = {
            scenario: generate_start_rows(templates[scenario], design)
            for scenario in ("base", "response_high", "combined_stress")
        }
        term_rows = {
            scenario: truncate_rows_to_policy_term(bundle["rows"], term_years)
            for scenario, bundle in bundles.items()
        }
        start_date = term_rows["base"][0]["date"]
        maturity_date = project_cashflows(
            term_rows["base"], mortality, design["settings"], treatment
        )[-1]["date"]
        for issue_age in range(
            int(senior["minimum_issue_age"]), int(senior["maximum_issue_age"]) + 1
        ):
            policy = validate_issue_age(
                issue_age, term_years, mode="senior_extension", rules=rules
            )
            for sex in rules["sexes"]:
                settings = CashflowSettings(
                    **{**design["settings_config"], "entry_age": issue_age, "sex": sex}
                )
                projections = {
                    "base": project_cashflows(
                        term_rows["base"], mortality, settings, treatment
                    ),
                    "response_high": project_cashflows(
                        term_rows["response_high"], mortality, settings, treatment
                    ),
                    "combined_stress": project_cashflows(
                        term_rows["combined_stress"], mortality, settings, treatment
                    ),
                    "treatment_combined_stress": project_cashflows(
                        term_rows["base"], mortality, settings, treatment_stress
                    ),
                }
                summaries = {
                    scenario: summarize(rows, settings.initial_policies)
                    for scenario, rows in projections.items()
                }
                base_summary = summaries["base"]
                price = calculate_premiums(base_summary, sale_margin)
                fixed = price["monthly_gross_premium_rounded_up_krw"]
                evaluations = {
                    scenario: evaluate_projected_cashflows(rows, settings, fixed)
                    for scenario, rows in projections.items()
                }
                initial = settings.initial_policies
                item = design["mvp_config"]
                results.append(
                    {
                        "product_id": design["design_id"],
                        "benefit_per_dispense_krw": int(design["benefit"]),
                        "max_claims": design["max_claims"],
                        "waiting_days": design["waiting_days"],
                        "maximum_support_krw": int(design["maximum_support_krw"]),
                        "decision_horizon_units": design["decision_horizon_units"],
                        "issue_age": issue_age,
                        "sex": sex,
                        "term_years": term_years,
                        "attained_age": policy["attained_age"],
                        "issue_age_mode": policy["status"],
                        "product_type": policy["status"],
                        "status": policy["status"],
                        "policy_eligible": policy["policy_eligible"],
                        "target_profit_margin": sale_margin,
                        "monthly_net_premium_krw": price["monthly_net_premium_krw"],
                        "monthly_gross_premium_unrounded_krw": price[
                            "monthly_gross_premium_krw"
                        ],
                        "monthly_gross_premium_rounded_up_krw": fixed,
                        "unrounded_achieved_profit_margin": price[
                            "achieved_profit_margin"
                        ],
                        "rounded_achieved_profit_margin": price[
                            "rounded_achieved_profit_margin"
                        ],
                        "expected_benefit_pv_per_contract_krw": base_summary[
                            "pv_benefit_krw"
                        ]
                        / initial,
                        "expected_paid_claims_per_contract": base_summary[
                            "total_paid_claims"
                        ]
                        / initial,
                        "recipient_ratio": base_summary["recipients"] / initial,
                        "exhaustion_ratio": base_summary["exhausted"] / initial,
                        "expected_deaths_per_1000_contracts": base_summary["deaths"],
                        "base_npv_per_contract_krw": evaluations["base"][
                            "profit_pv_krw"
                        ]
                        / initial,
                        "base_npv_per_1000_contracts_krw": evaluations["base"][
                            "profit_pv_krw"
                        ],
                        "response_high_npv_per_contract_krw": evaluations[
                            "response_high"
                        ]["profit_pv_krw"]
                        / initial,
                        "response_high_npv_per_1000_contracts_krw": evaluations[
                            "response_high"
                        ]["profit_pv_krw"],
                        "start_combined_stress_npv_per_contract_krw": evaluations[
                            "combined_stress"
                        ]["profit_pv_krw"]
                        / initial,
                        "start_combined_stress_npv_per_1000_contracts_krw": evaluations[
                            "combined_stress"
                        ]["profit_pv_krw"],
                        "treatment_combined_stress_npv_per_contract_krw": evaluations[
                            "treatment_combined_stress"
                        ]["profit_pv_krw"]
                        / initial,
                        "treatment_combined_stress_npv_per_1000_contracts_krw": evaluations[
                            "treatment_combined_stress"
                        ]["profit_pv_krw"],
                        "fixed_premium_used_for_stress_krw": fixed,
                        "discontinuation_method": "default_overall_49pct",
                        "annual_discontinuation_probability": 0.49,
                        "age_specific_discontinuation_enabled": False,
                        "start_rate_age_assumption": "common_by_age",
                        "restart_rate_age_assumption": "common_by_age",
                        "sex_difference_scope": "mortality_only",
                        "lapse_rate": settings.monthly_lapse,
                        "contract_start_date": start_date,
                        "contract_end_date_exclusive": maturity_date,
                        "calendar_days_in_term": len(term_rows["base"]),
                        "risk_warning": item.get("risk_warning", ""),
                        "cohort_basis": "initial 1000 policies",
                    }
                )
    return results


def build_sale_product_age_results(
    baseline_age_rows: list[dict[str, Any]],
    senior_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """판매 규칙을 한 표에 모은다. 기본형은 추적 중인 대표연령 결과다."""
    standard = []
    for row in baseline_age_rows:
        if not math.isclose(float(row["target_profit_margin"]), 0.10):
            continue
        item = dict(row)
        item.update(
            issue_age_mode="standard_mvp",
            product_type="standard_mvp",
            status="standard_mvp",
            policy_eligible=True,
            sale_results_scope="representative_standard_ages",
            benefit_per_dispense_krw=100000,
            max_claims=13,
            waiting_days=90,
            maximum_support_krw=1300000,
            decision_horizon_units=13,
            fixed_premium_used_for_stress_krw=item[
                "monthly_gross_premium_rounded_up_krw"
            ],
            contract_start_date="2026-11-15",
            contract_end_date_exclusive="2036-11-15",
            calendar_days_in_term=3653,
            risk_warning="",
        )
        standard.append(item)
    senior = []
    for row in senior_rows:
        item = dict(row)
        item["issue_age_mode"] = "senior_extension_5year"
        item["product_type"] = "senior_extension_5year"
        item["status"] = "senior_extension_5year"
        item["sale_results_scope"] = "all_senior_ages_all_mvp_products"
        senior.append(item)
    return standard + senior


def build_sale_product_age_lookup(
    config: dict[str, Any] | None = None,
    senior_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """판매 가능한 8개 상품×보험나이 19~74세×남녀의 가격 조회표를 만든다.

    19~69세는 실제 10년 경로, 70~74세는 실제 5년 경로를 사용한다.
    각 상품별 시작률을 다시 생성하며 판매 목표이익률 10%만 저장한다.
    """
    config = config or load_mvp_config()
    rules = load_issue_age_rules()
    sale_margin = float(rules["sale_target_profit_margin"])
    if not math.isclose(sale_margin, float(config["sale_target_profit_margin"])):
        raise ValueError("가입연령·MVP 설정의 판매 목표이익률이 다릅니다")

    base_cashflow = json.loads((ROOT / "config/cashflows.json").read_text())
    base_treatment = json.loads((ROOT / "config/treatment.json").read_text())
    treatment = Parameters(**base_treatment)
    mortality = read_mortality()
    templates = load_start_templates()
    standard = rules["standard_product"]
    rows: list[dict[str, Any]] = []

    for design in build_mvp_designs(config, base_cashflow):
        start_rows = generate_start_rows(templates["base"], design)["rows"]
        start_date = start_rows[0]["date"]
        for issue_age in range(
            int(standard["minimum_issue_age"]),
            int(standard["maximum_issue_age"]) + 1,
        ):
            policy = validate_issue_age(
                issue_age, int(standard["term_years"]), mode="standard", rules=rules
            )
            for sex in rules["sexes"]:
                settings = CashflowSettings(
                    **{**design["settings_config"], "entry_age": issue_age, "sex": sex}
                )
                projected = project_cashflows(
                    start_rows, mortality, settings, treatment
                )
                summary = summarize(projected, settings.initial_policies)
                price = calculate_premiums(summary, sale_margin)
                initial = settings.initial_policies
                fixed = price["monthly_gross_premium_rounded_up_krw"]
                item = design["mvp_config"]
                rows.append(
                    {
                        "product_id": design["design_id"],
                        "benefit_per_dispense_krw": int(design["benefit"]),
                        "max_claims": design["max_claims"],
                        "waiting_days": design["waiting_days"],
                        "maximum_support_krw": int(design["maximum_support_krw"]),
                        "decision_horizon_units": design["decision_horizon_units"],
                        "issue_age": issue_age,
                        "sex": sex,
                        "term_years": policy["term_years"],
                        "attained_age": policy["attained_age"],
                        "issue_age_mode": policy["status"],
                        "product_type": policy["status"],
                        "status": policy["status"],
                        "policy_eligible": policy["policy_eligible"],
                        "target_profit_margin": sale_margin,
                        "monthly_net_premium_krw": price["monthly_net_premium_krw"],
                        "monthly_gross_premium_unrounded_krw": price[
                            "monthly_gross_premium_krw"
                        ],
                        "monthly_gross_premium_rounded_up_krw": fixed,
                        "unrounded_achieved_profit_margin": price[
                            "achieved_profit_margin"
                        ],
                        "rounded_achieved_profit_margin": price[
                            "rounded_achieved_profit_margin"
                        ],
                        "expected_benefit_pv_per_contract_krw": summary[
                            "pv_benefit_krw"
                        ]
                        / initial,
                        "expected_paid_claims_per_contract": summary[
                            "total_paid_claims"
                        ]
                        / initial,
                        "recipient_ratio": summary["recipients"] / initial,
                        "exhaustion_ratio": summary["exhausted"] / initial,
                        "expected_deaths_per_1000_contracts": summary["deaths"],
                        "fixed_premium_used_for_stress_krw": fixed,
                        "discontinuation_method": "default_overall_49pct",
                        "annual_discontinuation_probability": 0.49,
                        "age_specific_discontinuation_enabled": False,
                        "start_rate_age_assumption": "common_by_age",
                        "restart_rate_age_assumption": "common_by_age",
                        "sex_difference_scope": "mortality_only",
                        "lapse_rate": settings.monthly_lapse,
                        "contract_start_date": start_date,
                        "contract_end_date_exclusive": projected[-1]["date"],
                        "calendar_days_in_term": len(start_rows),
                        "risk_warning": item.get("risk_warning") or "none",
                        "cohort_basis": "initial 1000 policies",
                    }
                )

    if senior_rows is None:
        senior_rows = build_senior_extension_analysis(config)
    lookup_fields = list(rows[0])
    for source in senior_rows:
        item = dict(source)
        item["issue_age_mode"] = "senior_extension_5year"
        item["product_type"] = "senior_extension_5year"
        item["status"] = "senior_extension_5year"
        item["risk_warning"] = item.get("risk_warning") or "none"
        rows.append({field: item[field] for field in lookup_fields})
    return rows


def validate_sale_product_age_lookup(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    keys = {(r["product_id"], r["issue_age"], r["sex"]) for r in rows}
    _check(checks, "판매가격 조회표 896개 고유 조합", len(rows) == len(keys) == 896)
    _check(
        checks,
        "판매 목표이익률 10%만 사용",
        all(math.isclose(float(r["target_profit_margin"]), 0.10) for r in rows),
    )
    _check(
        checks,
        "가입연령별 보험기간·상품상태 일치",
        all(
            (
                19 <= int(r["issue_age"]) <= 69
                and int(r["term_years"]) == 10
                and r["issue_age_mode"] == r["product_type"] == r["status"] == "standard_mvp"
            )
            or (
                70 <= int(r["issue_age"]) <= 74
                and int(r["term_years"]) == 5
                and r["issue_age_mode"] == r["product_type"] == r["status"] == "senior_extension_5year"
            )
            for r in rows
        ),
    )
    _check(
        checks,
        "75세 이상 제외·보장종료연령 79세 이하",
        all(int(r["issue_age"]) <= 74 and int(r["attained_age"]) <= 79 for r in rows),
    )
    _check(
        checks,
        "상품 8개와 성별·연령 조합 완전성",
        len({r["product_id"] for r in rows}) == 8
        and {r["sex"] for r in rows} == {"여", "남"}
        and {int(r["issue_age"]) for r in rows} == set(range(19, 75)),
    )
    required = (
        "product_id",
        "issue_age",
        "sex",
        "term_years",
        "issue_age_mode",
        "product_type",
        "status",
        "target_profit_margin",
        "monthly_gross_premium_rounded_up_krw",
        "contract_start_date",
        "contract_end_date_exclusive",
    )
    _check(
        checks,
        "필수 조회 필드 결측 없음",
        all(r.get(field) not in (None, "") for r in rows for field in required),
    )
    reference = next(
        r
        for r in rows
        if r["product_id"] == "benefit100k_max13_wait90"
        and int(r["issue_age"]) == 49
        and r["sex"] == "여"
    )
    _check(
        checks,
        "49세 여성 기준상품 10% 보험료 3,141원",
        int(reference["monthly_gross_premium_rounded_up_krw"]) == 3141,
    )
    _check(
        checks,
        "올림 전 목표이익률 10% 달성",
        max(abs(float(r["unrounded_achieved_profit_margin"]) - 0.10) for r in rows)
        < 1e-12,
    )
    return {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
        "summary": {"row_count": len(rows), "unique_key_count": len(keys)},
    }


def build_extension_policy_comparison() -> list[dict[str, Any]]:
    """노트북용 69·74·79세 정책 상태 비교. 가격 결과는 주 분석과 분리한다."""
    rules = load_issue_age_rules()
    cases = [
        (69, 10, "standard"),
        (74, 5, "senior_extension"),
        (79, 10, "research"),
    ]
    return [validate_issue_age(age, term, mode=mode, rules=rules) for age, term, mode in cases]


def validate_analysis(
    mvp_rows: list[dict[str, Any]],
    baseline_age_rows: list[dict[str, Any]],
    exploratory_age_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    _check(checks, "MVP 상품 8개", len(mvp_rows) == 8)
    _check(
        checks,
        "MVP 지급액·한도·대기기간 범위",
        {row["benefit_per_dispense_krw"] for row in mvp_rows} == {100000, 150000}
        and {row["max_claims"] for row in mvp_rows} == {6, 13}
        and {row["waiting_days"] for row in mvp_rows} == {30, 90},
    )
    _check(
        checks,
        "최대 지원금 산식",
        all(
            row["maximum_support_krw"]
            == row["benefit_per_dispense_krw"] * row["max_claims"]
            for row in mvp_rows
        ),
    )
    baseline = next(
        row for row in mvp_rows if row["product_id"] == "benefit100k_max13_wait90"
    )
    _check(
        checks,
        "기존 49세 여성 기준 보험료 유지",
        [
            baseline["monthly_premium_target_0pct_rounded_krw"],
            baseline["monthly_premium_target_5pct_rounded_krw"],
            baseline["monthly_premium_target_10pct_rounded_krw"],
        ]
        == [2808, 2965, 3141],
    )
    expected_age_rows = 7 * 2 * 2
    _check(
        checks,
        "기본·연령별 탐색 결과 행 수",
        len(baseline_age_rows) == len(exploratory_age_rows) == expected_age_rows,
    )
    _check(
        checks,
        "기본 중단률 옵션 비활성",
        all(not row["age_specific_discontinuation_enabled"] for row in baseline_age_rows),
    )
    _check(
        checks,
        "연령별 중단률 옵션 명시 활성",
        all(row["age_specific_discontinuation_enabled"] for row in exploratory_age_rows),
    )
    _check(
        checks,
        "올림 전 목표이익률 달성",
        max(
            abs(row["unrounded_achieved_profit_margin"] - row["target_profit_margin"])
            for row in baseline_age_rows + exploratory_age_rows
        )
        < 1e-12,
    )
    _check(
        checks,
        "인원·지급지표 범위",
        all(
            row["expected_paid_claims_per_contract"] >= 0
            and 0 <= row["exhaustion_ratio"] <= 1
            and 0 <= row["expected_deaths_per_1000_contracts"] <= 1000
            for row in baseline_age_rows + exploratory_age_rows
        ),
    )
    return {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
    }


def validate_senior_extension_analysis(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    keys = {(r["product_id"], r["issue_age"], r["sex"]) for r in rows}
    _check(checks, "고령 확장형 80개 고유 조합", len(rows) == len(keys) == 80)
    _check(
        checks,
        "판매 규칙·목표이익률",
        all(
            r["term_years"] == 5
            and r["product_type"] == "senior_extension_5year"
            and r["policy_eligible"]
            and r["attained_age"] <= 79
            and math.isclose(r["target_profit_margin"], 0.10)
            for r in rows
        ),
    )
    _check(
        checks,
        "달력 기준 5년 경로",
        all(
            r["contract_start_date"] == "2026-11-15"
            and r["contract_end_date_exclusive"] == "2031-11-15"
            and r["calendar_days_in_term"] == 1826
            for r in rows
        ),
    )
    _check(
        checks,
        "올림 전 목표이익률 10% 달성",
        max(abs(r["unrounded_achieved_profit_margin"] - 0.10) for r in rows)
        < 1e-12,
    )
    reference = {
        (70, "여"): (3290, 105971.0, 17867000.0, 6500000.0, -49895000.0, 9246000.0),
        (70, "남"): (3294, 104132.0, 17637000.0, 6466000.0, -48971000.0, 9175000.0),
        (74, "여"): (3292, 105010.0, 17745000.0, 6480000.0, -49415000.0, 9207000.0),
        (74, "남"): (3297, 102279.0, 17349000.0, 6377000.0, -48095000.0, 9048000.0),
    }
    representative = {
        (r["issue_age"], r["sex"]): r
        for r in rows
        if r["product_id"] == "benefit100k_max13_wait90"
        and (r["issue_age"], r["sex"]) in reference
    }
    reference_ok = len(representative) == 4
    for key, expected in reference.items():
        if key not in representative:
            reference_ok = False
            continue
        row = representative[key]
        actual = (
            row["monthly_gross_premium_rounded_up_krw"],
            row["expected_benefit_pv_per_contract_krw"],
            row["base_npv_per_1000_contracts_krw"],
            row["response_high_npv_per_1000_contracts_krw"],
            row["start_combined_stress_npv_per_1000_contracts_krw"],
            row["treatment_combined_stress_npv_per_1000_contracts_krw"],
        )
        reference_ok &= actual[0] == expected[0]
        reference_ok &= abs(actual[1] - expected[1]) < 2.0
        reference_ok &= all(abs(a - e) < 60000.0 for a, e in zip(actual[2:], expected[2:]))
    _check(checks, "대표상품 임시 대조값 재현", reference_ok)
    response_losses = [r for r in rows if r["response_high_npv_per_1000_contracts_krw"] < 0]
    _check(
        checks,
        "지원반응 손실 20개·15만 원×13회 한정",
        len(response_losses) == 20
        and all(r["benefit_per_dispense_krw"] == 150000 and r["max_claims"] == 13 for r in response_losses),
    )
    _check(
        checks,
        "시작률 복합 스트레스 80개 모두 손실",
        all(r["start_combined_stress_npv_per_1000_contracts_krw"] < 0 for r in rows),
    )
    _check(
        checks,
        "치료행태 복합 스트레스 80개 모두 흑자",
        all(r["treatment_combined_stress_npv_per_1000_contracts_krw"] > 0 for r in rows),
    )
    premium_values = [r["monthly_gross_premium_rounded_up_krw"] for r in rows]
    _check(
        checks,
        "보험료 범위 2,212~4,558원",
        min(premium_values) == 2212 and max(premium_values) == 4558,
    )
    return {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
        "summary": {
            "row_count": len(rows),
            "premium_min_krw": min(premium_values),
            "premium_max_krw": max(premium_values),
            "response_high_loss_count": len(response_losses),
            "start_combined_stress_loss_count": sum(
                r["start_combined_stress_npv_per_1000_contracts_krw"] < 0
                for r in rows
            ),
            "treatment_combined_stress_profit_count": sum(
                r["treatment_combined_stress_npv_per_1000_contracts_krw"] > 0
                for r in rows
            ),
        },
    }


def save_results(
    mvp_rows: list[dict[str, Any]],
    baseline_age_rows: list[dict[str, Any]],
    exploratory_age_rows: list[dict[str, Any]],
    senior_rows: list[dict[str, Any]],
    sale_rows: list[dict[str, Any]],
    sale_lookup_rows: list[dict[str, Any]] | None,
    validation: dict[str, Any],
    senior_validation: dict[str, Any],
    sale_lookup_validation: dict[str, Any] | None,
    source_validation: dict[str, Any] | None,
) -> None:
    _write_csv(DERIVED_DIR / "mvp_product_options_summary.csv", mvp_rows)
    _write_csv(DERIVED_DIR / "issue_age_baseline_results.csv", baseline_age_rows)
    _write_csv(
        DERIVED_DIR / "issue_age_discontinuation_exploratory_results.csv",
        exploratory_age_rows,
    )
    _write_csv(DERIVED_DIR / "senior_extension_5year_results.csv", senior_rows)
    _write_csv(
        DERIVED_DIR / "sale_product_age_representative_summary.csv", sale_rows
    )
    if sale_lookup_rows is not None:
        _write_csv(DERIVED_DIR / "sale_product_age_lookup.csv", sale_lookup_rows)
    if source_validation is None and REPORT_PATH.exists():
        previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        source_validation = previous.get("team_27_source_validation")
    report = {
        "analysis_validation": validation,
        "team_27_source_validation": source_validation,
        "output_files": [
            "data/derived/mvp_product_options_summary.csv",
            "data/derived/issue_age_baseline_results.csv",
            "data/derived/issue_age_discontinuation_exploratory_results.csv",
            "data/derived/senior_extension_5year_results.csv",
            "data/derived/sale_product_age_representative_summary.csv",
            *(
                ["data/derived/sale_product_age_lookup.csv"]
                if sale_lookup_rows is not None
                else []
            ),
        ],
        "sale_product_age_lookup_validation": sale_lookup_validation,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "mvp_products": mvp_rows,
                "validation": validation,
                "source_validation": source_validation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    SENIOR_REPORT_PATH.write_text(
        json.dumps(
            {
                "validation": senior_validation,
                "output_files": [
                    "data/derived/senior_extension_5year_results.csv",
                    "data/derived/sale_product_age_representative_summary.csv",
                    *(
                        ["data/derived/sale_product_age_lookup.csv"]
                        if sale_lookup_rows is not None
                        else []
                    ),
                ],
                "scope": {
                    "products": 8,
                    "issue_ages": [70, 71, 72, 73, 74],
                    "sexes": ["여", "남"],
                    "term_years": 5,
                    "sale_target_profit_margin": 0.10,
                    "cohort_basis": "initial 1000 policies",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _find_team_sources(downloads: Path) -> tuple[Path | None, Path | None]:
    files = sorted(downloads.glob("WEGOFIT_27*20261118.csv"))
    pricing = next((path for path in files if "NPV" not in path.name), None)
    stress = next((path for path in files if "NPV" in path.name), None)
    return pricing, stress


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP 8개 상품과 가입연령 분석")
    parser.add_argument("--team-pricing-csv", type=Path)
    parser.add_argument("--team-stress-csv", type=Path)
    parser.add_argument("--skip-team-source-validation", action="store_true")
    parser.add_argument(
        "--build-sale-lookup",
        action="store_true",
        help="8개 상품×19~74세×남녀 896행 판매가격 조회표 생성",
    )
    args = parser.parse_args()
    mvp_rows, _ = build_mvp_summary()
    source_validation = None
    if not args.skip_team_source_validation:
        pricing_path, stress_path = args.team_pricing_csv, args.team_stress_csv
        if pricing_path is None or stress_path is None:
            auto_pricing, auto_stress = _find_team_sources(Path.home() / "Downloads")
            pricing_path = pricing_path or auto_pricing
            stress_path = stress_path or auto_stress
        if pricing_path is None or stress_path is None:
            raise FileNotFoundError(
                "팀원 27개 조건 CSV를 찾지 못했습니다. 경로 옵션을 지정하거나 "
                "--skip-team-source-validation을 사용하세요."
            )
        source_validation = validate_team_27_sources(
            pricing_path, stress_path, mvp_rows
        )
        if source_validation["status"] != "PASS":
            raise AssertionError("팀원 27개 조건 CSV 대조 실패")
    elif (DERIVED_DIR / "mvp_product_options_summary.csv").exists():
        previous_rows = _read_team_csv(
            DERIVED_DIR / "mvp_product_options_summary.csv"
        )
        previous_match = {
            row["product_id"]: row.get("source_27_condition_match", "not_checked")
            for row in previous_rows
        }
        for row in mvp_rows:
            row["source_27_condition_match"] = previous_match.get(
                row["product_id"], "not_checked"
            )
    baseline_age_rows = build_age_analysis(use_age_specific_discontinuation=False)
    exploratory_age_rows = build_age_analysis(use_age_specific_discontinuation=True)
    senior_rows = build_senior_extension_analysis()
    sale_rows = build_sale_product_age_results(baseline_age_rows, senior_rows)
    sale_lookup_rows = None
    sale_lookup_validation = None
    if args.build_sale_lookup:
        sale_lookup_rows = build_sale_product_age_lookup(senior_rows=senior_rows)
        sale_lookup_validation = validate_sale_product_age_lookup(sale_lookup_rows)
        if sale_lookup_validation["status"] != "PASS":
            raise AssertionError("판매가격 조회표 검증 실패")
    validation = validate_analysis(mvp_rows, baseline_age_rows, exploratory_age_rows)
    if validation["status"] != "PASS":
        raise AssertionError("MVP·가입연령 분석 검증 실패")
    senior_validation = validate_senior_extension_analysis(senior_rows)
    if senior_validation["status"] != "PASS":
        raise AssertionError("고령 확장형 5년 분석 검증 실패")
    save_results(
        mvp_rows,
        baseline_age_rows,
        exploratory_age_rows,
        senior_rows,
        sale_rows,
        sale_lookup_rows,
        validation,
        senior_validation,
        sale_lookup_validation,
        source_validation,
    )
    print(
        json.dumps(
            {
                "mvp_product_count": len(mvp_rows),
                "baseline_age_result_count": len(baseline_age_rows),
                "exploratory_age_result_count": len(exploratory_age_rows),
                "senior_extension_result_count": len(senior_rows),
                "sale_product_age_result_count": len(sale_rows),
                "sale_product_age_lookup_count": (
                    len(sale_lookup_rows) if sale_lookup_rows is not None else 0
                ),
                "validation": validation["status"],
                "senior_extension_validation": senior_validation["status"],
                "sale_product_age_lookup_validation": (
                    sale_lookup_validation["status"]
                    if sale_lookup_validation
                    else "SKIPPED"
                ),
                "team_source_validation": (
                    source_validation["status"] if source_validation else "SKIPPED"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
