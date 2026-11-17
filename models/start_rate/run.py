import csv
import json
from pathlib import Path
from models.start_rate.model import Scenario, run

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/start_rate"

EXPECTED_NAMES = {
    "market_low",
    "base",
    "market_high",
    "response_zero",
    "response_high",
    "combined_stress",
}


def save_csv(filename, rows):
    if not rows:
        raise ValueError(f"{filename}에 저장할 결과가 없습니다.")

    with (OUT / filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    config_path = ROOT / "config/start_scenarios.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    scenarios_config = config["scenarios"]
    names = {item["inputs"]["name"] for item in scenarios_config}

    if names != EXPECTED_NAMES:
        raise ValueError(
            "six_scenarios.json의 시나리오 이름이나 개수를 확인하세요."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    summaries = []
    annual_results = []
    daily_results = []
    scenario_objects = []

    for item in scenarios_config:
        args = dict(item["inputs"])
        args["selection_ratios"] = tuple(args["selection_ratios"])

        scenario = Scenario(**args).validate()
        summary, years, days = run(scenario, keep_daily=True)

        summary["role_ko"] = item["role_ko"]

        summaries.append(summary)
        annual_results.extend(years)
        daily_results.extend(days)
        scenario_objects.append(scenario)

    # 과제에서 요구한 날짜별 조건부 시작확률 파일
    start_probabilities = [
        {
            "scenario": row["scenario"],
            "elapsed_day": row["elapsed_days"],
            "date": row["date"],
            "p_start_no": row["conditional_daily_q_without"],
            "p_start_yes": row["conditional_daily_q_with"],
        }
        for row in daily_results
    ]

    save_csv("start_probabilities.csv", start_probabilities)

    # 첫해 시작률 요약
    start_summary = [
        {
            "scenario": row["scenario"],
            "first_year_A_percent": row["year1_A_percent"],
            "first_year_A_plus_B_percent": row["year1_with_percent"],
            "difference_B_percentage_points": row["year1_B_pp"],
        }
        for row in summaries
    ]

    save_csv("start_summary.csv", start_summary)

    # 결과를 따로 확인할 수 있도록 전체 요약과 날짜별 원자료도 저장
    save_csv("six_scenarios_full_summary.csv", summaries)
    save_csv("six_scenarios_daily_model_output.csv", daily_results)

    # 기본 검증
    checks = []

    def check(name, condition):
        passed = bool(condition)
        checks.append({"check": name, "passed": passed})
        if not passed:
            raise AssertionError(f"검증 실패: {name}")

    check("six_scenarios_loaded", len(scenarios_config) == 6)

    check(
        "daily_probabilities_between_zero_and_one",
        all(
            0 <= row["p_start_no"] <= 1
            and 0 <= row["p_start_yes"] <= 1
            for row in start_probabilities
        ),
    )

    # 시나리오별 날짜 중복 여부와 날짜 수 확인
    dates_by_scenario = {}
    for row in start_probabilities:
        dates_by_scenario.setdefault(row["scenario"], []).append(row["date"])

    check(
        "no_duplicate_dates_within_each_scenario",
        all(
            len(dates) == len(set(dates))
            for dates in dates_by_scenario.values()
        ),
    )

    check(
        "same_number_of_dates_for_all_scenarios",
        len({len(dates) for dates in dates_by_scenario.values()}) == 1,
    )

    # 날짜별 결과를 빠르게 찾기 위한 표
    daily_lookup = {
        (row["scenario"], row["date"]): row
        for row in start_probabilities
    }

    # 지원 반응이 0이면 특약 없음·있음 확률이 같아야 함
    zero_rows = [
        row for row in start_probabilities
        if row["scenario"] == "response_zero"
    ]

    check(
        "zero_response_yes_equals_no",
        all(
            abs(row["p_start_no"] - row["p_start_yes"]) < 1e-12
            for row in zero_rows
        ),
    )

    # 연 신규 시작자가 12만 명인 시나리오끼리는
    # 특약 없음 경로가 같아야 함
    common_market_names = ["base", "response_zero", "response_high"]
    common_dates = dates_by_scenario["base"]

    same_no_path = True
    for date in common_dates:
        reference_value = daily_lookup[("base", date)]["p_start_no"]
        for name in common_market_names[1:]:
            other_value = daily_lookup[(name, date)]["p_start_no"]
            if abs(reference_value - other_value) >= 1e-12:
                same_no_path = False

    check(
        "same_market_size_has_same_no_rider_path",
        same_no_path,
    )

    # 6개 시나리오의 결과가 모두 저장됐는지 확인
    check(
        "all_six_scenarios_in_daily_output",
        {row["scenario"] for row in start_probabilities} == EXPECTED_NAMES,
    )

    check(
        "all_six_scenarios_in_summary",
        {row["scenario"] for row in start_summary} == EXPECTED_NAMES,
    )

    checks_path = OUT / "checks_six_scenarios.json"
    checks_path.write_text(
        json.dumps(
            {
                "checks_passed": len(checks),
                "active_scenarios": 6,
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("6개 시나리오 실행 완료")
    print(f"검증 통과: {len(checks)}개")
    print(f"결과 폴더: {OUT}")
    print(json.dumps(start_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()