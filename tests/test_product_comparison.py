import unittest

from models.rider.product_comparison import build_pricing_stage


class ProductComparisonChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage=build_pricing_stage()
        cls.rows=cls.stage['pricing_rows']

    def test_nine_designs_and_baseline_prices(self):
        self.assertEqual(len({row['design_id'] for row in self.rows}),9)
        self.assertEqual(len(self.rows),18)
        baseline={row['target_profit_margin']:row for row in self.rows
                  if row['design_id']=='benefit100k_max13_wait90'}
        self.assertEqual(baseline[.05]['monthly_gross_premium_rounded_up_krw'],2965)
        self.assertEqual(baseline[.10]['monthly_gross_premium_rounded_up_krw'],3141)
        self.assertEqual(self.stage['validation']['status'],'PASS')

    def test_design_conditions_and_target_margins(self):
        self.assertTrue(all(row['product_condition_match'] for row in self.rows))
        for row in self.rows:
            self.assertAlmostEqual(
                row['unrounded_achieved_profit_margin'],row['target_profit_margin'],places=12,
            )

    def test_unique_recipients_and_limits(self):
        for row in self.rows:
            self.assertLessEqual(row['exhausted'],row['recipients'])
            self.assertLessEqual(row['recipients'],row['total_starts'])
            self.assertLessEqual(
                row['total_paid_claims'],row['recipients']*row['max_claims']+1e-7,
            )

    def test_decision_horizon_keeps_13_and_26_start_paths_equal(self):
        by_key={(row['benefit_per_dispense_krw'],row['max_claims'],row['target_profit_margin']):row
                for row in self.rows}
        for benefit in (100000.,150000.,200000.):
            for margin in (.05,.10):
                cap13=by_key[(benefit,13,margin)]
                cap26=by_key[(benefit,26,margin)]
                self.assertEqual(cap13['start_probability_path_sha256'],cap26['start_probability_path_sha256'])
                self.assertEqual(cap13['ten_year_starter_ratio_initial'],cap26['ten_year_starter_ratio_initial'])
                self.assertGreater(cap26['total_paid_claims'],cap13['total_paid_claims'])


if __name__=='__main__':unittest.main()
