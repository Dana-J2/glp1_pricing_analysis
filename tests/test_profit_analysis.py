import unittest

from models.rider.cashflows import CashflowSettings,project_cashflows,summarize
from models.rider.pricing import achieved_profit_margin,calculate_premiums
from models.rider.profit_analysis import build_analysis,evaluate_projected_cashflows
from models.rider.start_integration import ROOT,read_mortality,read_starts


class ProfitAnalysisChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings=CashflowSettings()
        cls.rows=project_cashflows(
            read_starts(ROOT/'outputs/start_rate/start_probabilities.csv','base'),
            read_mortality(),cls.settings,
        )
        cls.summary=summarize(cls.rows,cls.settings.initial_policies)
        cls.premium_rows,cls.scenario_rows,cls.matrix,cls.validation=build_analysis()

    def test_zero_margin_reproduces_baseline(self):
        price=calculate_premiums(self.summary,0)
        self.assertAlmostEqual(price['monthly_gross_premium_krw'],2807.0470368896235,places=10)
        self.assertAlmostEqual(price['monthly_net_premium_krw'],2006.8642190787155,places=10)

    def test_target_margins_and_monotonic_premiums(self):
        prices=[calculate_premiums(self.summary,m) for m in (0,.05,.10)]
        self.assertEqual(prices[0]['monthly_net_premium_krw'],prices[1]['monthly_net_premium_krw'])
        self.assertEqual(prices[1]['monthly_net_premium_krw'],prices[2]['monthly_net_premium_krw'])
        self.assertLess(prices[0]['monthly_gross_premium_krw'],prices[1]['monthly_gross_premium_krw'])
        self.assertLess(prices[1]['monthly_gross_premium_krw'],prices[2]['monthly_gross_premium_krw'])
        for margin,price in zip((0,.05,.10),prices):
            self.assertAlmostEqual(achieved_profit_margin(
                self.summary,price['monthly_gross_premium_krw']),margin,places=12)

    def test_direct_cashflow_margin_and_proportional_items(self):
        for margin in (0,.05,.10):
            price=calculate_premiums(self.summary,margin)
            g=price['monthly_gross_premium_krw']
            result=evaluate_projected_cashflows(self.rows,self.settings,g)
            self.assertAlmostEqual(result['profit_margin'],margin,places=12)
            formula_profit=g*self.summary['pv_net_premium_units']-self.summary['pv_fixed_outflow_krw']
            self.assertAlmostEqual(result['profit_pv_krw'],formula_profit,places=6)
            self.assertAlmostEqual(result['refund_pv_krw'],g*self.summary['pv_refund_units'],places=6)
            self.assertAlmostEqual(result['collection_pv_krw'],g*self.summary['pv_collection_units'],places=6)
            self.assertAlmostEqual(result['acquisition_variable_pv_krw'],g*self.summary['pv_acquisition_units'],places=6)

    def test_invalid_target_or_denominator(self):
        with self.assertRaises(ValueError):
            calculate_premiums(self.summary,-.01)
        with self.assertRaisesRegex(ValueError,'분모가 0 이하'):
            calculate_premiums(dict(pv_premium_units=100,pv_refund_units=0,
                                    pv_net_premium_units=5),.05)

    def test_fixed_premium_scenario_grid_and_validation(self):
        self.assertEqual(len(self.premium_rows),3)
        self.assertEqual(len(self.scenario_rows),18)
        self.assertEqual(len(self.matrix),6)
        for margin in (0,.05,.10):
            fixed={r['fixed_monthly_gross_premium_krw'] for r in self.scenario_rows
                   if r['pricing_target_profit_margin']==margin}
            self.assertEqual(len(fixed),1)
        self.assertEqual(self.validation['status'],'PASS')


if __name__=='__main__':unittest.main()
