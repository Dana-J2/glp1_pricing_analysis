import unittest
from dataclasses import replace
from models.rider.cashflows import CashflowSettings,project_cashflows,summarize
from models.rider.start_integration import ROOT,read_starts,read_mortality
from models.rider.pricing import calculate_premiums

class PricingChecks(unittest.TestCase):
    def setUp(self):
        self.starts=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv')
        self.q=read_mortality()
    def test_cohort_scale_invariant(self):
        prices=[]
        for n in [1,1000]:
            s=CashflowSettings(initial_policies=n)
            prices.append(calculate_premiums(summarize(project_cashflows(self.starts,self.q,s),n)))
        for k in ['monthly_net_premium_krw','monthly_gross_premium_krw']:
            self.assertAlmostEqual(prices[0][k],prices[1][k],places=7)
    def test_zero_expenses_gross_equals_net(self):
        s=CashflowSettings(acquisition_fixed=0,alpha=0,beta=0,gamma=0,claim_cost=0)
        p=calculate_premiums(summarize(project_cashflows(self.starts,self.q,s),1000))
        self.assertAlmostEqual(p['monthly_gross_premium_krw'],p['monthly_net_premium_krw'],places=9)
    def test_invalid_denominator(self):
        with self.assertRaises(ValueError):
            calculate_premiums(dict(pv_premium_units=1,pv_refund_units=1,pv_net_premium_units=0))

if __name__=='__main__':unittest.main()
