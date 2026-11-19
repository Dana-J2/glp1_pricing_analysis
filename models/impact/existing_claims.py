"""Daily expected-value model for existing insurance claim impacts.

The model compares dedicated low/high Wegovy market paths for the same initial
SELECT-like high-risk cohort.  It reuses the repository's start-rate generator
and treatment persistence/restart engine, but excludes rider premiums, waiting
periods, payment limits and refunds.

Non-fatal claim histories are tracked separately as marginal unpaid/paid
expected states.  A non-fatal event does not remove a life from later treatment,
death or another benefit's risk set.  Death is processed last each day and
removes the life from all following days.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import date
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from models.start_rate.model import Scenario, run as run_start_rate
from models.treatment.model import (
    Parameters,
    params_from_json,
    treatment_pathway,
    validate_expected_path,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/existing_claims_impact.json"
DEFAULT_OUTPUT = ROOT / "outputs/existing_claims_impact"
BENEFIT_IDS = ("death", "mi", "stroke", "gallbladder")
NONFATAL_IDS = ("mi", "stroke", "gallbladder")
EXPOSURE_METHODS = (
    "on_treatment_published",
    "on_treatment_adjusted",
    "itt_window",
)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    analysis = config["analysis"]
    if analysis["lapse_rate"] != 0:
        raise ValueError("현재 종합영향 모형은 해지율 0만 지원합니다.")
    if analysis["base_exposure_method"] not in EXPOSURE_METHODS:
        raise ValueError("알 수 없는 기준 노출방식")
    if analysis["initial_cohort"] <= 0 or analysis["term_years"] <= 0:
        raise ValueError("양의 코호트와 관찰기간이 필요합니다.")
    return config


def _path_name(market_id: str, relative_multiplier: float) -> str:
    if math.isclose(relative_multiplier, 1.0):
        return market_id
    suffix = f"{relative_multiplier:.2f}".replace(".", "p")
    return f"{market_id}_relative_{suffix}"


def _start_scenario(
    config: dict[str, Any], market_id: str, relative_multiplier: float
) -> Scenario:
    start_cfg = config["start_rate"]
    template = dict(start_cfg["template"])
    years = int(template["contract_years"])
    template.update(
        name=_path_name(market_id, relative_multiplier),
        annual_first_people=int(
            start_cfg["market_scenarios"][market_id]["annual_first_people"]
        ),
        selection_ratios=(float(template.pop("selection_ratio")),) * years,
        nondiabetic_relative_start=float(relative_multiplier),
    )
    return Scenario(**template).validate()


def generate_start_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Regenerate no-rider daily start probabilities for every multiplier."""
    path_map: dict[tuple[str, float], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    multipliers = config["start_rate"]["select_relative_start_multipliers"]
    market_ids = tuple(config["start_rate"]["market_scenarios"])
    for relative in multipliers:
        relative = float(relative)
        for market_id in market_ids:
            scenario = _start_scenario(config, market_id, relative)
            summary, years, daily = run_start_rate(scenario, keep_daily=True)
            q = np.asarray(
                [row["conditional_daily_q_without"] for row in daily], dtype=float
            )
            dates = [row["date"] for row in daily]
            if np.any((q < 0) | (q > 1)):
                raise ValueError("시작확률이 [0,1] 범위를 벗어났습니다.")
            key = (market_id, relative)
            path_map[key] = {
                "scenario": scenario,
                "summary": summary,
                "annual": years,
                "daily_q": q,
                "dates": dates,
            }
            summaries.append(
                {
                    "path_id": scenario.name,
                    "market_id": market_id,
                    "select_relative_start_multiplier": relative,
                    "annual_first_people": scenario.annual_first_people,
                    "selection_ratio": scenario.selection_ratios[0],
                    "year1_conditional_start_percent": years[0][
                        "conditional_A_percent"
                    ],
                    "normalized_10y_first_starts_no_death": summary[
                        "year10_cumulative_without"
                    ],
                    "path_type": "no_rider",
                }
            )
            daily_rows.extend(
                {
                    "path_id": scenario.name,
                    "market_id": market_id,
                    "select_relative_start_multiplier": relative,
                    "elapsed_day": row["elapsed_days"],
                    "date": row["date"],
                    "p_start_no": row["conditional_daily_q_without"],
                }
                for row in daily
            )
    lengths = {len(value["daily_q"]) for value in path_map.values()}
    if len(lengths) != 1:
        raise AssertionError("전용 시작률 경로의 길이가 서로 다릅니다.")
    return {"paths": path_map, "summaries": summaries, "daily_rows": daily_rows}


def _fft_convolve(a: np.ndarray, b: np.ndarray, length: int) -> np.ndarray:
    size = 1 << (len(a) + len(b) - 2).bit_length()
    values = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)
    values = values[:length]
    values[np.abs(values) < 1e-12] = 0.0
    return values


def _daily_probability(annual_hazard: float, multiplier: float = 1.0) -> float:
    if annual_hazard < 0 or multiplier < 0:
        raise ValueError("위험강도와 배수는 음수가 될 수 없습니다.")
    return -math.expm1(-annual_hazard * multiplier / 365.0)


def _select_gallstone_hazard_ratio(config: dict[str, Any]) -> float:
    gall = config["benefits"]["gallbladder"]
    h_treated = -math.log1p(-float(gall["select_treated_gallstone_risk"]))
    h_placebo = -math.log1p(-float(gall["select_placebo_gallstone_risk"]))
    return h_treated / h_placebo


def _base_parameters(config: dict[str, Any]) -> dict[str, Any]:
    analysis = config["analysis"]
    benefits = config["benefits"]
    return {
        "death_hr": float(benefits["death"]["published_multiplier"]),
        "mi_hr": float(benefits["mi"]["published_multiplier"]),
        "stroke_hr": float(benefits["stroke"]["published_multiplier"]),
        "gallstone_rr": _select_gallstone_hazard_ratio(config),
        "surgery_conversion_rate": float(
            benefits["gallbladder"]["surgery_conversion_rate"]
        ),
        "exposure_method": analysis["base_exposure_method"],
        "benefit_effect_years": float(analysis["effect_years"]),
        "gallstone_effect_years": float(analysis["gallstone_effect_years"]),
        "discount_rate": float(analysis["annual_discount_rate"]),
        "start_multiplier": 1.0,
        "risk_multipliers": {benefit_id: 1.0 for benefit_id in BENEFIT_IDS},
        "holding_rates": {
            benefit_id: float(benefits[benefit_id]["holding_rate"])
            for benefit_id in BENEFIT_IDS
        },
        "benefit_amounts": {
            benefit_id: float(benefits[benefit_id]["benefit_krw"])
            for benefit_id in BENEFIT_IDS
        },
    }


def _with_overrides(
    base: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in (overrides or {}).items():
        if key in ("risk_multipliers", "holding_rates", "benefit_amounts"):
            out[key].update(value)
        else:
            out[key] = value
    if out["exposure_method"] not in EXPOSURE_METHODS:
        raise ValueError("알 수 없는 노출방식")
    for mapping in ("risk_multipliers", "holding_rates", "benefit_amounts"):
        if set(out[mapping]) != set(BENEFIT_IDS):
            raise ValueError(f"{mapping}의 담보 키를 확인하세요.")
    if any(not 0 <= value <= 1 for value in out["holding_rates"].values()):
        raise ValueError("담보 보유율은 [0,1]이어야 합니다.")
    return out


def _treatment_on_fraction(
    days: int, params: Parameters
) -> tuple[np.ndarray, dict[str, Any]]:
    starts = np.zeros(days, dtype=float)
    starts[0] = 1.0
    path = treatment_pathway(starts, params=params)
    validation = validate_expected_path(path)
    if not validation["passed"]:
        raise AssertionError("치료 상태전이 인원 보존 실패")
    on = np.asarray(path["on_treatment"], dtype=float)
    if np.any((on < -1e-12) | (on > 1 + 1e-12)):
        raise AssertionError("단위 시작자 치료 중 기대비율 범위 오류")
    return np.clip(on, 0.0, 1.0), validation


def _active_multiplier(
    published: float,
    method: str,
    trial_exposure_share: float,
) -> float:
    if published <= 0:
        raise ValueError("HR/RR은 양수여야 합니다.")
    if method == "on_treatment_adjusted":
        return math.exp(math.log(published) / trial_exposure_share)
    return published


def _multiplier_path(
    published: float,
    method: str,
    on_fraction: np.ndarray,
    effect_days: int,
    trial_exposure_share: float,
) -> np.ndarray:
    active = _active_multiplier(published, method, trial_exposure_share)
    result = np.ones(len(on_fraction), dtype=float)
    window = np.arange(len(on_fraction)) < effect_days
    if method == "itt_window":
        result[window] = active
    else:
        result[window] = 1.0 + on_fraction[window] * (active - 1.0)
    return result


def _risk_inputs(
    config: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    benefits = config["benefits"]
    gall = benefits["gallbladder"]
    gall_years = float(gall["select_followup_months"]) / 12.0
    gall_hazard = -math.log1p(-float(gall["select_placebo_gallstone_risk"])) / gall_years
    return {
        "death_hazard": (
            float(benefits["death"]["base_events"])
            / float(benefits["death"]["base_person_years"])
            * parameters["risk_multipliers"]["death"]
        ),
        "mi_hazard": (
            float(benefits["mi"]["base_events"])
            / float(benefits["mi"]["base_person_years"])
            * parameters["risk_multipliers"]["mi"]
        ),
        "stroke_hazard": (
            float(benefits["stroke"]["base_events"])
            / float(benefits["stroke"]["base_person_years"])
            * parameters["risk_multipliers"]["stroke"]
        ),
        "gallstone_hazard": gall_hazard
        * parameters["risk_multipliers"]["gallbladder"],
    }


def _started_profiles(
    config: dict[str, Any],
    parameters: dict[str, Any],
    on_fraction: np.ndarray,
) -> dict[str, Any]:
    days = len(on_fraction)
    analysis = config["analysis"]
    exposure_share = float(analysis["trial_exposure_months"]) / float(
        analysis["trial_followup_months"]
    )
    inputs = _risk_inputs(config, parameters)
    benefit_effect_days = int(round(parameters["benefit_effect_years"] * 365))
    gall_effect_days = int(round(parameters["gallstone_effect_years"] * 365))
    method = parameters["exposure_method"]

    death_mult = _multiplier_path(
        parameters["death_hr"], method, on_fraction, benefit_effect_days, exposure_share
    )
    mi_mult = _multiplier_path(
        parameters["mi_hr"], method, on_fraction, benefit_effect_days, exposure_share
    )
    stroke_mult = _multiplier_path(
        parameters["stroke_hr"], method, on_fraction, benefit_effect_days, exposure_share
    )
    gall_mult = _multiplier_path(
        parameters["gallstone_rr"], method, on_fraction, gall_effect_days, exposure_share
    )

    death_q = np.asarray(
        [_daily_probability(inputs["death_hazard"], value) for value in death_mult]
    )
    mi_q = np.asarray(
        [_daily_probability(inputs["mi_hazard"], value) for value in mi_mult]
    )
    stroke_q = np.asarray(
        [_daily_probability(inputs["stroke_hazard"], value) for value in stroke_mult]
    )
    gallstone_q = np.asarray(
        [_daily_probability(inputs["gallstone_hazard"], value) for value in gall_mult]
    )
    surgery_q = gallstone_q * float(parameters["surgery_conversion_rate"])

    alive_open = np.zeros(days, dtype=float)
    alive_close = np.zeros(days, dtype=float)
    deaths = np.zeros(days, dtype=float)
    alive = 1.0
    for day in range(days):
        alive_open[day] = alive
        deaths[day] = alive * death_q[day]
        alive -= deaths[day]
        alive_close[day] = alive

    event_q = {"mi": mi_q, "stroke": stroke_q, "gallbladder": surgery_q}
    event_counts: dict[str, np.ndarray] = {}
    unpaid_close: dict[str, np.ndarray] = {}
    for benefit_id in NONFATAL_IDS:
        unpaid = 1.0
        counts = np.zeros(days, dtype=float)
        closes = np.zeros(days, dtype=float)
        for day in range(days):
            counts[day] = unpaid * event_q[benefit_id][day]
            unpaid = (unpaid - counts[day]) * (1.0 - death_q[day])
            closes[day] = unpaid
        event_counts[benefit_id] = counts
        unpaid_close[benefit_id] = closes

    return {
        "alive_open": alive_open,
        "alive_close": alive_close,
        "deaths": deaths,
        "on_treatment_alive": alive_open * on_fraction,
        "event_counts": event_counts,
        "unpaid_close": unpaid_close,
        "daily_probabilities": {
            "death": death_q,
            "mi": mi_q,
            "stroke": stroke_q,
            "gallbladder": surgery_q,
        },
    }


def _simulate_market(
    config: dict[str, Any],
    market_id: str,
    daily_q_start: np.ndarray,
    dates: list[str],
    parameters: dict[str, Any],
    on_fraction: np.ndarray,
) -> dict[str, Any]:
    days = len(daily_q_start)
    if len(dates) != days or len(on_fraction) != days:
        raise ValueError("일별 경로 길이가 일치하지 않습니다.")
    initial = float(config["analysis"]["initial_cohort"])
    inputs = _risk_inputs(config, parameters)
    profiles = _started_profiles(config, parameters, on_fraction)

    base_q = {
        "death": _daily_probability(inputs["death_hazard"]),
        "mi": _daily_probability(inputs["mi_hazard"]),
        "stroke": _daily_probability(inputs["stroke_hazard"]),
        "gallbladder": _daily_probability(inputs["gallstone_hazard"])
        * float(parameters["surgery_conversion_rate"]),
    }

    naive_alive = initial
    naive_unpaid = {benefit_id: initial for benefit_id in NONFATAL_IDS}
    starts = np.zeros(days, dtype=float)
    starts_unpaid = {
        benefit_id: np.zeros(days, dtype=float) for benefit_id in NONFATAL_IDS
    }
    naive_deaths = np.zeros(days, dtype=float)
    naive_events = {
        benefit_id: np.zeros(days, dtype=float) for benefit_id in NONFATAL_IDS
    }
    naive_alive_close = np.zeros(days, dtype=float)
    prior_paid_starts = {benefit_id: 0.0 for benefit_id in NONFATAL_IDS}

    for day, q_start in enumerate(daily_q_start):
        start_today = naive_alive * q_start
        starts[day] = start_today
        naive_alive -= start_today
        for benefit_id in NONFATAL_IDS:
            unpaid_start = naive_unpaid[benefit_id] * q_start
            starts_unpaid[benefit_id][day] = unpaid_start
            prior_paid_starts[benefit_id] += start_today - unpaid_start
            naive_unpaid[benefit_id] -= unpaid_start
            event_today = naive_unpaid[benefit_id] * base_q[benefit_id]
            naive_events[benefit_id][day] = event_today
            naive_unpaid[benefit_id] -= event_today
        death_today = naive_alive * base_q["death"]
        naive_deaths[day] = death_today
        naive_alive -= death_today
        for benefit_id in NONFATAL_IDS:
            naive_unpaid[benefit_id] *= 1.0 - base_q["death"]
        naive_alive_close[day] = naive_alive

    started_deaths = _fft_convolve(starts, profiles["deaths"], days)
    started_alive_close = _fft_convolve(starts, profiles["alive_close"], days)
    on_treatment_alive = _fft_convolve(
        starts, profiles["on_treatment_alive"], days
    )
    started_events = {
        benefit_id: _fft_convolve(
            starts_unpaid[benefit_id], profiles["event_counts"][benefit_id], days
        )
        for benefit_id in NONFATAL_IDS
    }

    daily_counts = {"death": naive_deaths + started_deaths}
    daily_counts.update(
        {
            benefit_id: naive_events[benefit_id] + started_events[benefit_id]
            for benefit_id in NONFATAL_IDS
        }
    )
    discount = (1.0 + float(parameters["discount_rate"])) ** (
        -(np.arange(days, dtype=float) + 0.5) / 365.0
    )
    claims: dict[str, dict[str, float]] = {}
    for benefit_id in BENEFIT_IDS:
        count = float(daily_counts[benefit_id].sum())
        amount = float(parameters["benefit_amounts"][benefit_id])
        holding = float(parameters["holding_rates"][benefit_id])
        claims[benefit_id] = {
            "expected_claim_count": count,
            "nominal_krw": count * amount * holding,
            "pv_krw": float(
                np.sum(daily_counts[benefit_id] * amount * holding * discount)
            ),
        }

    cumulative_deaths = np.cumsum(daily_counts["death"])
    live_close = naive_alive_close + started_alive_close
    mass_error = initial - live_close - cumulative_deaths
    if float(np.max(np.abs(mass_error))) > 1e-6:
        raise AssertionError("사망·생존 인원 보존 실패")

    daily_rows = []
    cumulative_starts = np.cumsum(starts)
    for day in range(days):
        daily_rows.append(
            {
                "market_id": market_id,
                "date": dates[day],
                "elapsed_day": day,
                "p_start_no": float(daily_q_start[day]),
                "first_starts": float(starts[day]),
                "cumulative_first_starts": float(cumulative_starts[day]),
                "naive_alive_close": float(naive_alive_close[day]),
                "started_alive_close": float(started_alive_close[day]),
                "on_treatment_alive": float(on_treatment_alive[day]),
                "death_claims": float(daily_counts["death"][day]),
                "mi_claims": float(daily_counts["mi"][day]),
                "stroke_claims": float(daily_counts["stroke"][day]),
                "gallbladder_claims": float(daily_counts["gallbladder"][day]),
                "active_close": float(live_close[day]),
                "cumulative_deaths": float(cumulative_deaths[day]),
                "mass_error": float(mass_error[day]),
                "discount": float(discount[day]),
            }
        )

    return {
        "market_id": market_id,
        "total_first_starts": float(starts.sum()),
        "treatment_exposure_person_years": float(on_treatment_alive.sum() / 365.0),
        "prior_paid_starts": prior_paid_starts,
        "claims": claims,
        "max_mass_error": float(np.max(np.abs(mass_error))),
        "end_active": float(live_close[-1]),
        "total_deaths": float(cumulative_deaths[-1]),
        "daily_rows": daily_rows,
    }


def _claim_comparison_rows(
    config: dict[str, Any], low: dict[str, Any], high: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for benefit_id in BENEFIT_IDS:
        low_claim = low["claims"][benefit_id]
        high_claim = high["claims"][benefit_id]
        change = high_claim["pv_krw"] - low_claim["pv_krw"]
        rows.append(
            {
                "benefit_id": benefit_id,
                "benefit_name_ko": config["benefits"][benefit_id][
                    "display_name_ko"
                ],
                "low_market_expected_claim_count": low_claim[
                    "expected_claim_count"
                ],
                "high_market_expected_claim_count": high_claim[
                    "expected_claim_count"
                ],
                "low_market_pv_krw": low_claim["pv_krw"],
                "high_market_pv_krw": high_claim["pv_krw"],
                "claim_change_krw": change,
                "insurer_impact_krw": -change,
            }
        )
    total_change = sum(row["claim_change_krw"] for row in rows)
    rows.append(
        {
            "benefit_id": "total",
            "benefit_name_ko": "종합",
            "low_market_expected_claim_count": sum(
                row["low_market_expected_claim_count"] for row in rows
            ),
            "high_market_expected_claim_count": sum(
                row["high_market_expected_claim_count"] for row in rows
            ),
            "low_market_pv_krw": sum(row["low_market_pv_krw"] for row in rows),
            "high_market_pv_krw": sum(row["high_market_pv_krw"] for row in rows),
            "claim_change_krw": total_change,
            "insurer_impact_krw": -total_change,
        }
    )
    return rows


def _offset_ratio(rows: list[dict[str, Any]]) -> float:
    by_id = {row["benefit_id"]: row for row in rows}
    reduction = -sum(
        min(0.0, by_id[benefit_id]["claim_change_krw"])
        for benefit_id in ("death", "mi", "stroke")
    )
    gall_increase = max(0.0, by_id["gallbladder"]["claim_change_krw"])
    return gall_increase / reduction if reduction > 0 else math.nan


def _evaluate_pair(
    config: dict[str, Any],
    start_paths: dict[str, Any],
    on_fraction: np.ndarray,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    relative = float(parameters["start_multiplier"])
    low_path = start_paths["paths"][("existing_market_low", relative)]
    high_path = start_paths["paths"][("existing_market_high", relative)]
    low = _simulate_market(
        config,
        "existing_market_low",
        low_path["daily_q"],
        low_path["dates"],
        parameters,
        on_fraction,
    )
    high = _simulate_market(
        config,
        "existing_market_high",
        high_path["daily_q"],
        high_path["dates"],
        parameters,
        on_fraction,
    )
    rows = _claim_comparison_rows(config, low, high)
    total = next(row for row in rows if row["benefit_id"] == "total")
    return {
        "parameters": parameters,
        "low": low,
        "high": high,
        "claim_rows": rows,
        "total_claim_change_krw": total["claim_change_krw"],
        "insurer_impact_krw": total["insurer_impact_krw"],
        "per_policy_claim_change_krw": total["claim_change_krw"]
        / float(config["analysis"]["initial_cohort"]),
        "per_policy_insurer_impact_krw": total["insurer_impact_krw"]
        / float(config["analysis"]["initial_cohort"]),
        "gallbladder_offset_ratio": _offset_ratio(rows),
    }


def run_market_pair(
    config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    starts = generate_start_paths(config)
    treatment_path = ROOT / config["treatment"]["params_path"]
    treatment = params_from_json(treatment_path)
    days = len(next(iter(starts["paths"].values()))["daily_q"])
    on_fraction, _ = _treatment_on_fraction(days, treatment)
    parameters = _with_overrides(_base_parameters(config), overrides)
    return _evaluate_pair(config, starts, on_fraction, parameters)


def _flatten_sensitivity(
    variable: str,
    setting: str,
    display: str,
    result: dict[str, Any],
    note: str = "",
) -> dict[str, Any]:
    by_id = {row["benefit_id"]: row for row in result["claim_rows"]}
    return {
        "variable": variable,
        "setting": setting,
        "display": display,
        "note": note,
        "death_claim_change_krw": by_id["death"]["claim_change_krw"],
        "mi_claim_change_krw": by_id["mi"]["claim_change_krw"],
        "stroke_claim_change_krw": by_id["stroke"]["claim_change_krw"],
        "gallbladder_claim_change_krw": by_id["gallbladder"][
            "claim_change_krw"
        ],
        "total_claim_change_krw": result["total_claim_change_krw"],
        "insurer_impact_krw": result["insurer_impact_krw"],
        "gallbladder_offset_ratio": result["gallbladder_offset_ratio"],
    }


def _one_way_sensitivity(
    config: dict[str, Any],
    starts: dict[str, Any],
    on_fraction: np.ndarray,
    base_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = config["one_way_sensitivity"]
    rows: list[dict[str, Any]] = []

    def evaluate(
        variable: str,
        setting: str,
        display: str,
        overrides: dict[str, Any],
        note: str = "",
    ) -> None:
        parameters = _with_overrides(base_parameters, overrides)
        result = _evaluate_pair(config, starts, on_fraction, parameters)
        rows.append(_flatten_sensitivity(variable, setting, display, result, note))

    for value in spec["death_hr"]:
        evaluate("death_hr", str(value), f"HR {value:.2f}", {"death_hr": value})
    for value in spec["benefit_effect_years"]:
        evaluate(
            "benefit_effect_years",
            str(value),
            f"{value:g}년",
            {"benefit_effect_years": value},
            "SELECT 추적기간을 넘는 외삽" if value > 3.32 else "",
        )
    for value in spec["mi_hr"]:
        evaluate("mi_hr", str(value), f"HR {value:.2f}", {"mi_hr": value})
    for value in spec["stroke_hr"]:
        evaluate("stroke_hr", str(value), f"HR {value:.2f}", {"stroke_hr": value})

    gall = config["benefits"]["gallbladder"]
    gall_settings = {
        "select_base": _select_gallstone_hazard_ratio(config),
        "rr_1.40": float(gall["safwan_reported_low"]),
        "rr_2.59": float(gall["safwan_rr"]),
        "rr_4.82": float(gall["safwan_reported_high"]),
    }
    for setting in spec["gallstone_scenarios"]:
        evaluate(
            "gallstone_rr",
            setting,
            setting.replace("select_base", "SELECT 기준").replace("rr_", "RR "),
            {"gallstone_rr": gall_settings[setting]},
            "Safwan RR도 SELECT 위약군 기초위험에 적용" if setting != "select_base" else "",
        )
    for value in spec["surgery_conversion_rates"]:
        evaluate(
            "surgery_conversion_rate",
            str(value),
            f"{value:.1%}",
            {"surgery_conversion_rate": value},
        )
    for value in spec["exposure_methods"]:
        evaluate("exposure_method", value, value, {"exposure_method": value})
    for value in spec["select_relative_start_multipliers"]:
        evaluate(
            "select_relative_start_multiplier",
            str(value),
            f"{value:.1f}배",
            {"start_multiplier": value},
            "시작률 모형 재실행 경로",
        )
    for value in spec["discount_rates"]:
        evaluate(
            "discount_rate", str(value), f"{value:.2%}", {"discount_rate": value}
        )
    for benefit_id in BENEFIT_IDS:
        for value in spec["holding_rates"]:
            evaluate(
                f"holding_rate_{benefit_id}",
                str(value),
                f"{config['benefits'][benefit_id]['display_name_ko']} {value:.0%}",
                {"holding_rates": {benefit_id: value}},
            )
    for value in spec["base_risk_multipliers"]:
        evaluate(
            "all_base_risk_multiplier",
            str(value),
            f"{value:.2f}배",
            {"risk_multipliers": {benefit_id: value for benefit_id in BENEFIT_IDS}},
        )
    return rows


def _combined_scenarios(
    config: dict[str, Any],
    starts: dict[str, Any],
    on_fraction: np.ndarray,
    base_parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    details: dict[str, dict[str, Any]] = {}
    for scenario_id, spec in config["combined_scenarios"].items():
        overrides = {key: value for key, value in spec.items() if key != "display_name_ko"}
        parameters = _with_overrides(base_parameters, overrides)
        result = _evaluate_pair(config, starts, on_fraction, parameters)
        details[scenario_id] = result
        row = _flatten_sensitivity(
            "combined_scenario",
            scenario_id,
            spec["display_name_ko"],
            result,
            "발생확률 미부여",
        )
        row["scenario_id"] = scenario_id
        row["display_name_ko"] = spec["display_name_ko"]
        rows.append(row)
    return rows, details


def _validation(
    config: dict[str, Any],
    starts: dict[str, Any],
    treatment_validation: dict[str, Any],
    on_fraction: np.ndarray,
    base_parameters: dict[str, Any],
    base: dict[str, Any],
    combined_details: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(check: str, condition: bool, detail: str = "") -> None:
        checks.append({"check": check, "passed": bool(condition), "detail": detail})

    add(
        "high_market_has_more_first_starts",
        base["high"]["total_first_starts"] > base["low"]["total_first_starts"],
    )
    add(
        "high_market_has_more_treatment_exposure",
        base["high"]["treatment_exposure_person_years"]
        > base["low"]["treatment_exposure_person_years"],
    )
    no_effect = combined_details["no_effect"]
    add(
        "all_effect_multipliers_one_produce_zero_change",
        max(
            abs(row["claim_change_krw"])
            for row in no_effect["claim_rows"]
            if row["benefit_id"] != "total"
        )
        < 1e-4,
    )
    gall_neutral_row = next(
        row
        for row in no_effect["claim_rows"]
        if row["benefit_id"] == "gallbladder"
    )
    add(
        "equal_gallstone_risk_produces_zero_extra_surgery_claim",
        abs(gall_neutral_row["claim_change_krw"]) < 1e-4,
    )
    add(
        "death_removed_and_mass_conserved",
        max(base["low"]["max_mass_error"], base["high"]["max_mass_error"])
        < 1e-6,
    )
    add(
        "nonfatal_paid_lives_can_start_later",
        all(
            base[market]["prior_paid_starts"][benefit_id] > 0
            for market in ("low", "high")
            for benefit_id in ("mi", "stroke")
        ),
    )
    add(
        "nonfatal_benefits_are_independent_and_remain_payable",
        all(
            base[market]["claims"][benefit_id]["expected_claim_count"] > 0
            for market in ("low", "high")
            for benefit_id in NONFATAL_IDS
        ),
        "담보별 미청구 상태를 별도 추적하므로 서로 다른 비치명적 담보 지급을 허용",
    )
    add(
        "each_nonfatal_benefit_paid_at_most_once",
        all(
            base[market]["claims"][benefit_id]["expected_claim_count"]
            <= config["analysis"]["initial_cohort"] + 1e-8
            for market in ("low", "high")
            for benefit_id in NONFATAL_IDS
        ),
    )
    doubled = _evaluate_pair(
        config,
        starts,
        on_fraction,
        _with_overrides(
            base_parameters,
            {
                "benefit_amounts": {
                    "death": 2.0 * base_parameters["benefit_amounts"]["death"]
                }
            },
        ),
    )
    base_death_change = next(
        row for row in base["claim_rows"] if row["benefit_id"] == "death"
    )["claim_change_krw"]
    doubled_death_change = next(
        row for row in doubled["claim_rows"] if row["benefit_id"] == "death"
    )["claim_change_krw"]
    add(
        "double_benefit_doubles_claim_change",
        math.isclose(doubled_death_change, 2.0 * base_death_change, rel_tol=1e-12, abs_tol=1e-5),
    )
    half_holding = _evaluate_pair(
        config,
        starts,
        on_fraction,
        _with_overrides(base_parameters, {"holding_rates": {"death": 0.5}}),
    )
    half_death_change = next(
        row for row in half_holding["claim_rows"] if row["benefit_id"] == "death"
    )["claim_change_krw"]
    add(
        "half_holding_rate_halves_claim_change",
        math.isclose(half_death_change, 0.5 * base_death_change, rel_tol=1e-12, abs_tol=1e-5),
    )
    component_sum = sum(
        row["claim_change_krw"]
        for row in base["claim_rows"]
        if row["benefit_id"] != "total"
    )
    add(
        "benefit_changes_sum_to_total",
        math.isclose(component_sum, base["total_claim_change_krw"], abs_tol=1e-6),
    )
    add(
        "insurer_impact_is_negative_total_change",
        math.isclose(
            base["insurer_impact_krw"], -base["total_claim_change_krw"], abs_tol=1e-9
        ),
    )
    add("treatment_path_mass_conserved", treatment_validation["passed"])
    add(
        "start_probabilities_in_unit_interval",
        all(
            np.all((value["daily_q"] >= 0) & (value["daily_q"] <= 1))
            for value in starts["paths"].values()
        ),
    )
    add(
        "selection_ratio_is_one_for_existing_claim_paths",
        all(
            math.isclose(item["selection_ratio"], 1.0)
            for item in starts["summaries"]
        ),
    )
    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    return {
        "status": status,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }


def build_analysis(
    config_path: str | Path = DEFAULT_CONFIG,
    include_sensitivities: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    starts = generate_start_paths(config)
    treatment = params_from_json(ROOT / config["treatment"]["params_path"])
    days = len(next(iter(starts["paths"].values()))["daily_q"])
    on_fraction, treatment_validation = _treatment_on_fraction(days, treatment)
    base_parameters = _base_parameters(config)
    base = _evaluate_pair(config, starts, on_fraction, base_parameters)
    combined_rows, combined_details = _combined_scenarios(
        config, starts, on_fraction, base_parameters
    )
    sensitivity_rows = (
        _one_way_sensitivity(config, starts, on_fraction, base_parameters)
        if include_sensitivities
        else []
    )
    validation = _validation(
        config,
        starts,
        treatment_validation,
        on_fraction,
        base_parameters,
        base,
        combined_details,
    )
    if validation["status"] != "PASS":
        failed = [item["check"] for item in validation["checks"] if not item["passed"]]
        raise AssertionError(f"종합영향 검증 실패: {failed}")
    return {
        "config": config,
        "treatment_parameters": asdict(treatment),
        "treatment_validation": treatment_validation,
        "start_paths": starts,
        "base": base,
        "one_way_sensitivity": sensitivity_rows,
        "combined_scenarios": combined_rows,
        "combined_details": combined_details,
        "validation": validation,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Scenario):
        return asdict(value)
    return value


def write_analysis(analysis: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "existing_market_start_paths.csv", analysis["start_paths"]["daily_rows"])
    _write_csv(output / "start_path_summary.csv", analysis["start_paths"]["summaries"])
    _write_csv(output / "base_claim_pv_results.csv", analysis["base"]["claim_rows"])
    _write_csv(
        output / "base_exposure_summary.csv",
        [
            {
                "market_id": analysis["base"][market]["market_id"],
                "total_first_starts": analysis["base"][market]["total_first_starts"],
                "treatment_exposure_person_years": analysis["base"][market][
                    "treatment_exposure_person_years"
                ],
                "end_active": analysis["base"][market]["end_active"],
                "total_deaths": analysis["base"][market]["total_deaths"],
                "max_mass_error": analysis["base"][market]["max_mass_error"],
            }
            for market in ("low", "high")
        ],
    )
    _write_csv(output / "base_low_daily.csv", analysis["base"]["low"]["daily_rows"])
    _write_csv(output / "base_high_daily.csv", analysis["base"]["high"]["daily_rows"])
    _write_csv(
        output / "base_additional_metrics.csv",
        [
            {
                "initial_cohort": analysis["config"]["analysis"]["initial_cohort"],
                "low_market_first_starts": analysis["base"]["low"][
                    "total_first_starts"
                ],
                "high_market_first_starts": analysis["base"]["high"][
                    "total_first_starts"
                ],
                "low_market_treatment_exposure_person_years": analysis["base"][
                    "low"
                ]["treatment_exposure_person_years"],
                "high_market_treatment_exposure_person_years": analysis["base"][
                    "high"
                ]["treatment_exposure_person_years"],
                "claim_change_per_1000_policies_krw": analysis["base"][
                    "total_claim_change_krw"
                ],
                "claim_change_per_policy_krw": analysis["base"][
                    "per_policy_claim_change_krw"
                ],
                "insurer_impact_per_1000_policies_krw": analysis["base"][
                    "insurer_impact_krw"
                ],
                "insurer_impact_per_policy_krw": analysis["base"][
                    "per_policy_insurer_impact_krw"
                ],
                "gallbladder_offset_ratio": analysis["base"][
                    "gallbladder_offset_ratio"
                ],
            }
        ],
    )
    _write_csv(output / "one_way_sensitivity.csv", analysis["one_way_sensitivity"])
    _write_csv(output / "combined_scenarios.csv", analysis["combined_scenarios"])
    common_rows = {
        row["benefit_id"]: row
        for row in analysis["base"]["claim_rows"]
        if row["benefit_id"] != "total"
    }
    _write_csv(
        output / "legacy_notebook_reference_comparison.csv",
        [
            {
                "benefit_id": benefit_id,
                "legacy_notebook": reference["source"],
                "legacy_claim_change_krw": reference["claim_change_krw"],
                "common_model_claim_change_krw": common_rows[benefit_id][
                    "claim_change_krw"
                ],
                "difference_common_minus_legacy_krw": common_rows[benefit_id][
                    "claim_change_krw"
                ]
                - float(reference["claim_change_krw"]),
                "main_differences": reference["main_differences"],
            }
            for benefit_id, reference in analysis["config"][
                "legacy_notebook_references"
            ].items()
        ],
    )
    _write_csv(
        output / "holding_rate_sensitivity.csv",
        [
            row
            for row in analysis["one_way_sensitivity"]
            if row["variable"].startswith("holding_rate_")
        ],
    )
    summary = {
        "analysis_name_ko": analysis["config"]["analysis_name_ko"],
        "base_parameters": analysis["base"]["parameters"],
        "base_low": {
            key: value
            for key, value in analysis["base"]["low"].items()
            if key not in ("daily_rows", "claims")
        },
        "base_high": {
            key: value
            for key, value in analysis["base"]["high"].items()
            if key not in ("daily_rows", "claims")
        },
        "base_claim_rows": analysis["base"]["claim_rows"],
        "total_claim_change_krw": analysis["base"]["total_claim_change_krw"],
        "insurer_impact_krw": analysis["base"]["insurer_impact_krw"],
        "per_policy_claim_change_krw": analysis["base"][
            "per_policy_claim_change_krw"
        ],
        "per_policy_insurer_impact_krw": analysis["base"][
            "per_policy_insurer_impact_krw"
        ],
        "gallbladder_offset_ratio": analysis["base"][
            "gallbladder_offset_ratio"
        ],
        "validation": analysis["validation"],
    }
    (output / "base_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "validation.json").write_text(
        json.dumps(_jsonable(analysis["validation"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "assumptions_snapshot.json").write_text(
        json.dumps(_jsonable(analysis["config"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="위고비 시장 확산에 따른 기존 담보 보험금 현가 변화"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    analysis = build_analysis(args.config, include_sensitivities=not args.skip_sensitivity)
    write_analysis(analysis, args.output_dir)
    print(
        json.dumps(
            _jsonable(
                {
                    "low_first_starts": analysis["base"]["low"]["total_first_starts"],
                    "high_first_starts": analysis["base"]["high"]["total_first_starts"],
                    "low_treatment_exposure_py": analysis["base"]["low"][
                        "treatment_exposure_person_years"
                    ],
                    "high_treatment_exposure_py": analysis["base"]["high"][
                        "treatment_exposure_person_years"
                    ],
                    "claim_rows": analysis["base"]["claim_rows"],
                    "validation": analysis["validation"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"결과 폴더: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
