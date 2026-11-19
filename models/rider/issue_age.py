"""MVP 가입연령 정책과 선택적 연령별 중단률 탐색 기능."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from models.rider.start_integration import ROOT
from models.treatment.model import Parameters


DEFAULT_RULES_PATH = ROOT / "config/issue_age_rules.json"


def load_issue_age_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    if rules["age_basis"] != "insurance_age":
        raise ValueError("가입연령 기준은 보험나이여야 합니다")
    standard = rules["standard_product"]
    senior = rules["senior_extension_product"]
    if standard["minimum_issue_age"] > standard["maximum_issue_age"]:
        raise ValueError("기본형 최소·최대 가입연령 순서를 확인하세요")
    if senior["minimum_issue_age"] > senior["maximum_issue_age"]:
        raise ValueError("고령 확장형 최소·최대 가입연령 순서를 확인하세요")
    if senior["maximum_issue_age"] != rules["maximum_sale_issue_age"]:
        raise ValueError("최대 판매 가입연령과 고령 확장형 상한이 다릅니다")
    if not math.isclose(float(rules["sale_target_profit_margin"]), 0.10):
        raise ValueError("판매 목표이익률은 10%여야 합니다")
    return rules


def validate_issue_age(
    issue_age: int,
    term_years: int | None = None,
    *,
    mode: str = "standard",
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """판매 규칙과 연구용 계산 권한을 분리해 검증한다.

    `standard`는 19~69세 10년형, `senior_extension`은 70~74세
    5년형, `research`는 명시적으로 선택한 75~79세 연구 계산이다.
    연구 모드는 판매 가능 여부를 변경하지 않는다. 기존 호출 호환을 위해
    `short_term_extension`은 `senior_extension`의 별칭으로만 허용한다.
    """
    rules = rules or load_issue_age_rules()
    if isinstance(issue_age, bool) or not isinstance(issue_age, int):
        raise ValueError("가입연령은 정수 보험나이여야 합니다")
    if mode == "short_term_extension":
        mode = "senior_extension"
    if mode not in {"standard", "senior_extension", "research"}:
        raise ValueError("가입연령 검증 모드를 확인하세요")

    if mode == "standard":
        standard = rules["standard_product"]
        term = standard["term_years"] if term_years is None else term_years
        if not standard["minimum_issue_age"] <= issue_age <= standard["maximum_issue_age"]:
            raise ValueError("기본 MVP 가입연령은 보험나이 19~69세입니다")
        if term != standard["term_years"]:
            raise ValueError("기본 MVP 보험기간은 정확히 10년입니다")
        attained = issue_age + term
        if attained > rules["maximum_attained_age"]:
            raise ValueError("보장종료연령이 최대 79세를 초과합니다")
        return {
            "mode": mode,
            "issue_age": issue_age,
            "term_years": term,
            "attained_age": attained,
            "policy_eligible": True,
            "status": "standard_mvp",
        }

    if mode == "senior_extension":
        extension = rules["senior_extension_product"]
        term = extension["term_years"] if term_years is None else term_years
        if not extension["minimum_issue_age"] <= issue_age <= extension["maximum_issue_age"]:
            raise ValueError("고령 확장형 가입연령은 보험나이 70~74세입니다")
        if term != extension["term_years"]:
            raise ValueError("70~74세 고령 확장형 보험기간은 정확히 5년입니다")
        attained = issue_age + term
        if attained > rules["maximum_attained_age"]:
            raise ValueError("보장종료연령이 최대 79세를 초과합니다")
        return {
            "mode": mode,
            "issue_age": issue_age,
            "term_years": term,
            "attained_age": attained,
            "policy_eligible": True,
            "status": "senior_extension_5year",
        }

    research = rules["research_mode"]
    term = rules["standard_product"]["term_years"] if term_years is None else term_years
    if not research["minimum_entry_age"] <= issue_age <= research["maximum_entry_age"]:
        raise ValueError("연구용 계산 가입연령은 보험나이 75~79세입니다")
    if not isinstance(term, int) or term <= 0:
        raise ValueError("연구용 보험기간은 양의 정수여야 합니다")
    return {
        "mode": mode,
        "issue_age": issue_age,
        "term_years": term,
        "attained_age": issue_age + term,
        "policy_eligible": False,
        "status": "research_only_not_for_sale",
    }


def age_band_discontinuation_probability(
    issue_age: int, rules: dict[str, Any] | None = None
) -> float:
    rules = rules or load_issue_age_rules()
    bands = rules["age_specific_discontinuation_exploratory"][
        "annual_discontinuation_probability_by_age_band"
    ]
    matches = [
        band
        for band in bands
        if band["minimum_age"] <= issue_age <= band["maximum_age"]
    ]
    if len(matches) != 1:
        raise ValueError(f"연령 {issue_age}세의 중단률 구간을 하나로 매핑할 수 없습니다")
    probability = float(matches[0]["probability"])
    if not 0 <= probability < 1:
        raise ValueError("연간 중단확률은 [0,1)이어야 합니다")
    return probability


def annual_to_daily_stop_probability(annual_probability: float) -> float:
    if not math.isfinite(annual_probability) or not 0 <= annual_probability < 1:
        raise ValueError("연간 중단확률은 [0,1)이어야 합니다")
    return -math.expm1(math.log1p(-annual_probability) / 365.0)


def treatment_parameters_for_issue_age(
    issue_age: int,
    base_treatment_config: dict[str, Any],
    *,
    use_age_specific_discontinuation: bool = False,
    rules: dict[str, Any] | None = None,
) -> tuple[Parameters, dict[str, Any]]:
    """첫해 지속곡선 모양을 유지하며 연령군 12개월 값에 맞춘다.

    기준 3·6·9·12개월 지속률에 동일한 누적위험 배수 `k`를 적용한다.
    `S_age(t)=S_base(t)^k`, `k=log(S_age(12m))/log(S_base(12m))`다.
    기본 옵션을 끄면 원래 치료 설정을 그대로 반환한다.
    """
    rules = rules or load_issue_age_rules()
    base = dict(base_treatment_config)
    if not use_age_specific_discontinuation:
        return Parameters(**base), {
            "method": "default_overall_49pct",
            "annual_discontinuation_probability": 1.0
            - float(base["first_year_12m_persistence"]),
            "curve_transform_exponent": 1.0,
            "age_specific_option_enabled": False,
        }

    if float(base.get("stop_reduction", 0.0)) != 0.0:
        raise ValueError("연령별 기준곡선 보정 전 stop_reduction은 0이어야 합니다")
    discontinuation = age_band_discontinuation_probability(issue_age, rules)
    target_persistence = 1.0 - discontinuation
    base_12m = float(base["first_year_12m_persistence"])
    exponent = math.log(target_persistence) / math.log(base_12m)
    for key in (
        "first_year_3m_persistence",
        "first_year_6m_persistence",
        "first_year_9m_persistence",
        "first_year_12m_persistence",
    ):
        base[key] = float(base[key]) ** exponent
    params = Parameters(**base)
    params.validate()
    return params, {
        "method": "age_specific_mailhac_curve_power_transform",
        "annual_discontinuation_probability": discontinuation,
        "target_12m_persistence": target_persistence,
        "curve_transform_exponent": exponent,
        "age_specific_option_enabled": True,
        "constant_daily_probability_equivalent": annual_to_daily_stop_probability(
            discontinuation
        ),
        "resolved_first_year_anchors": {
            key: asdict(params)[key]
            for key in (
                "first_year_3m_persistence",
                "first_year_6m_persistence",
                "first_year_9m_persistence",
                "first_year_12m_persistence",
            )
        },
    }
