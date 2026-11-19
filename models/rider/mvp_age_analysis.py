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

from models.rider.cashflows import CashflowSettings, project_cashflows, summarize
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
        writer = csv.DictWriter(handle, fieldnames=fields)
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
            for margin in representative["target_profit_margins"]:
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
                        "issue_age_mode": policy["mode"],
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
                        "exhaustion_ratio": base_summary["exhausted"] / initial,
                        "expected_deaths_per_1000_contracts": base_summary["deaths"],
                        "base_npv_per_contract_krw": evaluation["base"][
                            "profit_pv_krw"
                        ]
                        / initial,
                        "response_high_npv_per_contract_krw": evaluation[
                            "response_high"
                        ]["profit_pv_krw"]
                        / initial,
                        "start_combined_stress_npv_per_contract_krw": evaluation[
                            "combined_stress"
                        ]["profit_pv_krw"]
                        / initial,
                        "treatment_combined_stress_npv_per_contract_krw": evaluation[
                            "treatment_combined_stress"
                        ]["profit_pv_krw"]
                        / initial,
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


def build_extension_policy_comparison() -> list[dict[str, Any]]:
    """노트북용 69·74·79세 정책 상태 비교. 가격 결과는 주 분석과 분리한다."""
    rules = load_issue_age_rules()
    cases = [
        (69, 10, "standard"),
        (74, 5, "short_term_extension"),
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


def save_results(
    mvp_rows: list[dict[str, Any]],
    baseline_age_rows: list[dict[str, Any]],
    exploratory_age_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    source_validation: dict[str, Any] | None,
) -> None:
    _write_csv(DERIVED_DIR / "mvp_product_options_summary.csv", mvp_rows)
    _write_csv(DERIVED_DIR / "issue_age_baseline_results.csv", baseline_age_rows)
    _write_csv(
        DERIVED_DIR / "issue_age_discontinuation_exploratory_results.csv",
        exploratory_age_rows,
    )
    report = {
        "analysis_validation": validation,
        "team_27_source_validation": source_validation,
        "output_files": [
            "data/derived/mvp_product_options_summary.csv",
            "data/derived/issue_age_baseline_results.csv",
            "data/derived/issue_age_discontinuation_exploratory_results.csv",
        ],
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
    baseline_age_rows = build_age_analysis(use_age_specific_discontinuation=False)
    exploratory_age_rows = build_age_analysis(use_age_specific_discontinuation=True)
    validation = validate_analysis(mvp_rows, baseline_age_rows, exploratory_age_rows)
    if validation["status"] != "PASS":
        raise AssertionError("MVP·가입연령 분석 검증 실패")
    save_results(
        mvp_rows,
        baseline_age_rows,
        exploratory_age_rows,
        validation,
        source_validation,
    )
    print(
        json.dumps(
            {
                "mvp_product_count": len(mvp_rows),
                "baseline_age_result_count": len(baseline_age_rows),
                "exploratory_age_result_count": len(exploratory_age_rows),
                "validation": validation["status"],
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
