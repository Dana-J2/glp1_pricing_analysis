import csv
from datetime import date
import json
import math
import unittest

from models.rider.cashflows import (
    project_cashflows,
    summarize,
    truncate_rows_to_policy_term,
)
from models.rider.issue_age import (
    age_band_discontinuation_probability,
    annual_to_daily_stop_probability,
    load_issue_age_rules,
    treatment_parameters_for_issue_age,
    validate_issue_age,
)
from models.rider.mvp_age_analysis import (
    build_mvp_designs,
    load_mvp_config,
    validate_senior_extension_analysis,
)
from models.rider.pricing import calculate_premiums
from models.rider.product_comparison import generate_start_rows, load_start_templates
from models.rider.start_integration import ROOT, read_mortality
from models.treatment.model import Parameters, episode_survival


class MvpProductAndIssueAgeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mvp_config = load_mvp_config()
        cls.rules = load_issue_age_rules()
        cls.base_cashflow = json.loads((ROOT / "config/cashflows.json").read_text())
        cls.base_treatment = json.loads((ROOT / "config/treatment.json").read_text())
        cls.designs = build_mvp_designs(cls.mvp_config, cls.base_cashflow)

    def test_exactly_eight_mvp_conditions(self):
        self.assertEqual(len(self.designs), 8)
        self.assertEqual({d["benefit"] for d in self.designs}, {100000.0, 150000.0})
        self.assertEqual({d["max_claims"] for d in self.designs}, {6, 13})
        self.assertEqual({d["waiting_days"] for d in self.designs}, {30, 90})
        self.assertTrue(all(d["decision_horizon_units"] == 13 for d in self.designs))

    def test_maximum_support_is_benefit_times_limit(self):
        for design in self.designs:
            self.assertEqual(
                design["maximum_support_krw"],
                design["benefit"] * design["max_claims"],
            )

    def test_standard_issue_age_boundaries(self):
        with self.assertRaises(ValueError):
            validate_issue_age(18, rules=self.rules)
        self.assertTrue(validate_issue_age(19, rules=self.rules)["policy_eligible"])
        result = validate_issue_age(69, rules=self.rules)
        self.assertTrue(result["policy_eligible"])
        self.assertEqual(result["attained_age"], 79)
        with self.assertRaises(ValueError):
            validate_issue_age(70, rules=self.rules)

    def test_attained_age_limit_and_explicit_research_mode(self):
        with self.assertRaisesRegex(ValueError, "정확히 10년"):
            validate_issue_age(69, 11, rules=self.rules)
        extension = validate_issue_age(
            74, 5, mode="senior_extension", rules=self.rules
        )
        self.assertEqual(extension["attained_age"], 79)
        self.assertTrue(extension["policy_eligible"])
        self.assertEqual(extension["status"], "senior_extension_5year")
        for age in (75, 79):
            research = validate_issue_age(age, 5, mode="research", rules=self.rules)
            self.assertFalse(research["policy_eligible"])
            self.assertEqual(research["status"], "research_only_not_for_sale")
        with self.assertRaises(ValueError):
            validate_issue_age(74, 5, mode="research", rules=self.rules)
        with self.assertRaises(ValueError):
            validate_issue_age(80, 10, mode="research", rules=self.rules)

    def test_senior_extension_requires_exactly_five_years(self):
        for age in (70, 74):
            result = validate_issue_age(age, 5, mode="senior_extension", rules=self.rules)
            self.assertTrue(result["policy_eligible"])
            self.assertLessEqual(result["attained_age"], 79)
        for term in (1, 4, 6, 10):
            with self.assertRaisesRegex(ValueError, "정확히 5년"):
                validate_issue_age(70, term, mode="senior_extension", rules=self.rules)
        with self.assertRaises(ValueError):
            validate_issue_age(75, 5, mode="senior_extension", rules=self.rules)
        with self.assertRaises(ValueError):
            validate_issue_age(70, 10, mode="standard", rules=self.rules)

    def test_all_sale_contracts_end_by_age_79(self):
        policies = [validate_issue_age(age, 10, rules=self.rules) for age in range(19, 70)]
        policies += [
            validate_issue_age(age, 5, mode="senior_extension", rules=self.rules)
            for age in range(70, 75)
        ]
        self.assertTrue(all(row["policy_eligible"] for row in policies))
        self.assertTrue(all(row["attained_age"] <= 79 for row in policies))

    def test_calendar_term_truncation_includes_fifth_anniversary_eve(self):
        design = next(
            d for d in self.designs if d["design_id"] == "benefit100k_max13_wait90"
        )
        rows = generate_start_rows(load_start_templates()["base"], design)["rows"]
        truncated = truncate_rows_to_policy_term(rows, 5)
        self.assertEqual(len(truncated), 1826)
        self.assertEqual(truncated[0]["date"], "2026-11-15")
        self.assertEqual(truncated[-1]["date"], "2031-11-14")
        self.assertEqual(
            [int(row["elapsed_day"]) for row in truncated], list(range(1826))
        )
        projected = project_cashflows(
            truncated,
            read_mortality(),
            design["settings"],
            Parameters(**self.base_treatment),
        )
        self.assertEqual(projected[-1]["date"], "2031-11-15")
        self.assertEqual(projected[-1]["matured_today"], projected[-1]["active_open"])
        self.assertTrue(
            all(date.fromisoformat(row["date"]) < date(2031, 11, 15) for row in projected[:-1])
        )

    def test_sale_margin_is_ten_percent_only(self):
        self.assertEqual(self.mvp_config["sale_target_profit_margin"], 0.10)
        self.assertEqual(self.rules["sale_target_profit_margin"], 0.10)
        self.assertEqual(self.mvp_config["secondary_research_margin"], 0.05)
        with (ROOT / "data/derived/sale_product_age_results.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(float(row["target_profit_margin"]) == 0.10 for row in rows))
        self.assertTrue(all(int(row["issue_age"]) <= 74 for row in rows))
        self.assertEqual(
            {row["product_type"] for row in rows},
            {"standard_mvp", "senior_extension_5year"},
        )

    def test_age_band_mapping_boundaries(self):
        expected = {
            19: 0.648,
            29: 0.648,
            30: 0.542,
            44: 0.542,
            45: 0.436,
            59: 0.436,
            60: 0.446,
            69: 0.446,
            70: 0.490,
        }
        self.assertEqual(
            {age: age_band_discontinuation_probability(age, self.rules) for age in expected},
            expected,
        )

    def test_annual_daily_probability_round_trip(self):
        for annual in (0.436, 0.446, 0.49, 0.542, 0.648):
            daily = annual_to_daily_stop_probability(annual)
            recovered = 1.0 - (1.0 - daily) ** 365
            self.assertAlmostEqual(recovered, annual, places=12)

    def test_age_curve_uses_direct_group_values_and_recovers_12m_target(self):
        for age in (19, 30, 45, 60, 70):
            params, metadata = treatment_parameters_for_issue_age(
                age,
                self.base_treatment,
                use_age_specific_discontinuation=True,
                rules=self.rules,
            )
            target = 1.0 - age_band_discontinuation_probability(age, self.rules)
            self.assertAlmostEqual(episode_survival(365, params)[365], target, places=12)
            self.assertEqual(
                metadata["method"], "age_specific_mailhac_curve_power_transform"
            )
        self.assertNotIn(
            "relative_risk",
            self.rules["age_specific_discontinuation_exploratory"],
        )

    def test_disabling_age_option_preserves_default_treatment(self):
        params, metadata = treatment_parameters_for_issue_age(
            49,
            self.base_treatment,
            use_age_specific_discontinuation=False,
            rules=self.rules,
        )
        self.assertEqual(params, Parameters(**self.base_treatment))
        self.assertFalse(metadata["age_specific_option_enabled"])

    def test_existing_49_female_baseline_premiums_are_unchanged(self):
        design = next(
            d for d in self.designs if d["design_id"] == "benefit100k_max13_wait90"
        )
        start = generate_start_rows(load_start_templates()["base"], design)
        projected = project_cashflows(
            start["rows"],
            read_mortality(),
            design["settings"],
            Parameters(**self.base_treatment),
        )
        summary = summarize(projected, design["settings"].initial_policies)
        premiums = [
            calculate_premiums(summary, margin)[
                "monthly_gross_premium_rounded_up_krw"
            ]
            for margin in (0.0, 0.05, 0.10)
        ]
        self.assertEqual(premiums, [2808, 2965, 3141])

    def test_tracked_derived_results_have_expected_rows(self):
        expected = {
            "mvp_product_options_summary.csv": 8,
            "issue_age_baseline_results.csv": 28,
            "issue_age_discontinuation_exploratory_results.csv": 28,
            "senior_extension_5year_results.csv": 80,
            "sale_product_age_results.csv": 94,
        }
        for filename, count in expected.items():
            with (ROOT / "data/derived" / filename).open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), count)

    def test_tracked_senior_extension_price_and_stress_results(self):
        with (ROOT / "data/derived/senior_extension_5year_results.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        numeric_fields = (
            "issue_age",
            "monthly_gross_premium_rounded_up_krw",
            "expected_benefit_pv_per_contract_krw",
            "base_npv_per_1000_contracts_krw",
            "response_high_npv_per_1000_contracts_krw",
            "start_combined_stress_npv_per_1000_contracts_krw",
            "treatment_combined_stress_npv_per_1000_contracts_krw",
        )
        typed = []
        for row in rows:
            item = dict(row)
            for field in numeric_fields:
                item[field] = float(item[field])
            item["benefit_per_dispense_krw"] = int(item["benefit_per_dispense_krw"])
            item["max_claims"] = int(item["max_claims"])
            item["term_years"] = int(item["term_years"])
            item["attained_age"] = int(item["attained_age"])
            item["target_profit_margin"] = float(item["target_profit_margin"])
            item["policy_eligible"] = item["policy_eligible"] == "True"
            item["unrounded_achieved_profit_margin"] = float(
                item["unrounded_achieved_profit_margin"]
            )
            item["calendar_days_in_term"] = int(item["calendar_days_in_term"])
            typed.append(item)
        validation = validate_senior_extension_analysis(typed)
        self.assertEqual(validation["status"], "PASS")
        representative = {
            (int(r["issue_age"]), r["sex"]): int(
                float(r["monthly_gross_premium_rounded_up_krw"])
            )
            for r in rows
            if r["product_id"] == "benefit100k_max13_wait90"
        }
        self.assertEqual(representative[(70, "여")], 3290)
        self.assertEqual(representative[(74, "남")], 3297)


if __name__ == "__main__":
    unittest.main()
