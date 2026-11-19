import math
import unittest

from models.impact.existing_claims import BENEFIT_IDS, build_analysis


class ExistingClaimsImpactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = build_analysis(include_sensitivities=False)
        cls.base = cls.analysis["base"]
        cls.checks = {
            row["check"]: row for row in cls.analysis["validation"]["checks"]
        }

    def test_validation_and_market_order(self):
        self.assertEqual(self.analysis["validation"]["status"], "PASS")
        self.assertGreater(
            self.base["high"]["total_first_starts"],
            self.base["low"]["total_first_starts"],
        )
        self.assertGreater(
            self.base["high"]["treatment_exposure_person_years"],
            self.base["low"]["treatment_exposure_person_years"],
        )

    def test_no_effect_and_equal_gallstone_risk_are_neutral(self):
        result = self.analysis["combined_details"]["no_effect"]
        for row in result["claim_rows"]:
            self.assertAlmostEqual(row["claim_change_krw"], 0.0, places=4)

    def test_death_and_treatment_mass_is_conserved(self):
        for market in ("low", "high"):
            self.assertLess(self.base[market]["max_mass_error"], 1e-6)
        self.assertTrue(self.checks["treatment_path_mass_conserved"]["passed"])

    def test_nonfatal_claim_histories_remain_independent(self):
        self.assertTrue(
            self.checks["nonfatal_paid_lives_can_start_later"]["passed"]
        )
        self.assertTrue(
            self.checks[
                "nonfatal_benefits_are_independent_and_remain_payable"
            ]["passed"]
        )

    def test_each_nonfatal_benefit_is_paid_once_at_most(self):
        initial = self.analysis["config"]["analysis"]["initial_cohort"]
        for market in ("low", "high"):
            for benefit_id in ("mi", "stroke", "gallbladder"):
                self.assertLessEqual(
                    self.base[market]["claims"][benefit_id]["expected_claim_count"],
                    initial + 1e-8,
                )

    def test_amount_and_holding_rate_scale_linearly(self):
        self.assertTrue(self.checks["double_benefit_doubles_claim_change"]["passed"])
        self.assertTrue(self.checks["half_holding_rate_halves_claim_change"]["passed"])

    def test_component_sum_and_sign_definition(self):
        rows = [
            row
            for row in self.base["claim_rows"]
            if row["benefit_id"] != "total"
        ]
        component_sum = sum(row["claim_change_krw"] for row in rows)
        self.assertAlmostEqual(component_sum, self.base["total_claim_change_krw"], places=6)
        self.assertAlmostEqual(
            self.base["insurer_impact_krw"], -component_sum, places=6
        )

    def test_start_paths_are_dedicated_no_rider_paths(self):
        summaries = self.analysis["start_paths"]["summaries"]
        self.assertEqual(len(summaries), 6)
        self.assertTrue(all(row["path_type"] == "no_rider" for row in summaries))
        self.assertTrue(
            all(math.isclose(row["selection_ratio"], 1.0) for row in summaries)
        )
        self.assertEqual(
            {row["select_relative_start_multiplier"] for row in summaries},
            {0.5, 1.0, 1.5},
        )

    def test_start_probabilities_are_valid(self):
        self.assertTrue(self.checks["start_probabilities_in_unit_interval"]["passed"])

    def test_common_model_covers_all_four_benefits(self):
        ids = {
            row["benefit_id"]
            for row in self.base["claim_rows"]
            if row["benefit_id"] != "total"
        }
        self.assertEqual(ids, set(BENEFIT_IDS))

    def test_legacy_notebook_references_are_preserved_for_comparison(self):
        references = self.analysis["config"]["legacy_notebook_references"]
        self.assertEqual(set(references), set(BENEFIT_IDS))
        self.assertAlmostEqual(references["death"]["claim_change_krw"], -96961864.0)
        self.assertAlmostEqual(
            references["gallbladder"]["claim_change_krw"], 6119385.3826
        )


if __name__ == "__main__":
    unittest.main()
