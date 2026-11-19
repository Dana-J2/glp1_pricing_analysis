import csv
import json
import math
import unittest

from models.rider.cashflows import project_cashflows, summarize
from models.rider.issue_age import (
    age_band_discontinuation_probability,
    annual_to_daily_stop_probability,
    load_issue_age_rules,
    treatment_parameters_for_issue_age,
    validate_issue_age,
)
from models.rider.mvp_age_analysis import build_mvp_designs, load_mvp_config
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
        with self.assertRaisesRegex(ValueError, "79세를 초과"):
            validate_issue_age(69, 11, rules=self.rules)
        extension = validate_issue_age(
            74, 5, mode="short_term_extension", rules=self.rules
        )
        self.assertEqual(extension["attained_age"], 79)
        self.assertFalse(extension["policy_eligible"])
        for age in (70, 74, 75, 79):
            research = validate_issue_age(age, 10, mode="research", rules=self.rules)
            self.assertFalse(research["policy_eligible"])
            self.assertEqual(research["status"], "research_only_not_for_sale")
        with self.assertRaises(ValueError):
            validate_issue_age(80, 10, mode="research", rules=self.rules)

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
        }
        for filename, count in expected.items():
            with (ROOT / "data/derived" / filename).open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), count)


if __name__ == "__main__":
    unittest.main()
