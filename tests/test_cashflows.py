import unittest
from dataclasses import replace
from datetime import date,timedelta
import numpy as np
from models.rider.cashflows import *
from models.rider.contract_state import ContractState,BenefitRules,step_population,apply_exit
from models.treatment.model import DailyTransitionKernel

def make_rows(T=200,p=.01,start=date(2026,11,15)):
    return [dict(elapsed_day=i,date=(start+timedelta(days=i)).isoformat(),p_start_yes=p,p_start_no=p) for i in range(T)]

class CashflowChecks(unittest.TestCase):
    def test_matches_dictionary_states(self):
        T=200;rows=make_rows(T);q={(49,'여'):.02}
        s=CashflowSettings(initial_policies=1,max_claims=3)
        result=project_cashflows(rows,q,s)
        states={ContractState():1.};kernel=DailyTransitionKernel(Parameters(),T+1)
        rules=BenefitRules(max_claims=3,maturity_day=T)
        dq=1-(1-.02)**(1/365)
        for day in range(T):
            states,e=step_population(states,day,.01,kernel,rules)
            states=apply_exit(states,dq,'dead')
            for key in ['paid_claims','dispenses','restarts','new_starters']:
                self.assertAlmostEqual(result[day][key],e[key],places=10)
            active=sum(m for st,m in states.items() if st.status=='active')
            self.assertAlmostEqual(result[day]['active_close'],active,places=10)
    def test_no_start_analytic(self):
        rows=make_rows(365,0);s=CashflowSettings(initial_policies=1)
        r=project_cashflows(rows,{(49,'여'):.02},s)
        self.assertAlmostEqual(r[-1]['matured_today'],.98,places=10)
        self.assertEqual(sum(x['paid_claims'] for x in r),0)
        self.assertEqual(sum(x['acquisition_fixed_krw'] for x in r),10000)
        self.assertAlmostEqual(sum(x['acquisition_units'] for x in r),3.6)
        self.assertEqual(sum(x['premium_units']>0 for x in r),12)
    def test_waiting_cap_and_refund(self):
        rows=make_rows(500,1);s=CashflowSettings(initial_policies=1,death_enabled=False)
        r=project_cashflows(rows,{},s,Parameters(stop_reduction=1))
        self.assertEqual([x['elapsed_day'] for x in r if x['paid_claims']],list(range(112,449,28)))
        self.assertAlmostEqual(sum(x['paid_claims'] for x in r),13)
        self.assertAlmostEqual(sum(x['new_recipients'] for x in r),1)
        self.assertAlmostEqual(r[-1]['cumulative_recipients'],1)
        self.assertEqual(sum(x['benefit_krw'] for x in r),1300000)
        endday=date(2026,11,15)+timedelta(days=448)
        prev=date(endday.year,endday.month,15)
        if endday<prev:prev=month_date(date(2026,11,15),(endday.year-2026)*12+endday.month-11-1)
        nxt=month_date(date(2026,11,15),(prev.year-2026)*12+prev.month-11+1)
        self.assertAlmostEqual(r[448]['refund_units'],(nxt-endday).days/(nxt-prev).days)
        self.assertEqual(sum(x['premium_units'] for x in r[449:]),0)
    def test_same_day_refund_and_expense(self):
        r=project_cashflows(make_rows(40,1),{},CashflowSettings(initial_policies=1,death_enabled=False,waiting_days=0,max_claims=1))
        self.assertEqual(r[0]['premium_units'],1)
        self.assertEqual(r[0]['refund_units'],1)
        self.assertEqual(r[0]['collection_units'],.02)
        self.assertEqual(r[-1]['matured_today'],0)
    def test_month_end_and_maturity(self):
        self.assertEqual(month_date(date(2028,1,31),1),date(2028,2,29))
        self.assertEqual(month_date(date(2028,1,31),2),date(2028,3,31))
        r=project_cashflows(make_rows(60,0,start=date(2028,1,31)),{},CashflowSettings(initial_policies=1,death_enabled=False))
        self.assertEqual(r[-1]['premium_units'],0)
        self.assertEqual(r[-1]['benefit_krw'],0)
    def test_nonzero_lapse_rejected(self):
        with self.assertRaises(ValueError):project_cashflows(make_rows(),{},CashflowSettings(monthly_lapse=.01))
    def test_linear_premium_cashflow(self):
        r=project_cashflows(make_rows(100),{},CashflowSettings(initial_policies=1,death_enabled=False))
        for x in r:
            for g in [1000,2000]:
                direct=g*(x['premium_units']-x['refund_units']-x['collection_units']-x['acquisition_units'])-x['fixed_outflow_krw']
                self.assertAlmostEqual(direct,g*x['net_premium_units']-x['fixed_outflow_krw'])

    def test_full_term_premium_and_mass(self):
        rows=make_rows(3653,0)
        result=project_cashflows(rows,{},CashflowSettings(initial_policies=1,death_enabled=False))
        self.assertEqual(sum(x['premium_units'] for x in result),120)
        self.assertEqual(result[-1]['matured_today'],1)
        self.assertEqual(sum(x['refund_units'] for x in result),0)
    def test_full_term_baseline(self):
        rows=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv')
        r=project_cashflows(rows,read_mortality())
        self.assertLess(max(abs(x['mass_error']) for x in r),1e-7)
        self.assertLessEqual(r[-1]['cumulative_paid_claims'],13*r[-1]['cumulative_starts'])
        self.assertLessEqual(r[-1]['cumulative_exhausted'],r[-1]['cumulative_recipients'])
        self.assertLessEqual(r[-1]['cumulative_recipients'],r[-1]['cumulative_starts'])
        self.assertAlmostEqual(r[-1]['cumulative_exhausted']+r[-1]['cumulative_deaths']+r[-1]['matured_today'],1000,places=7)
        self.assertTrue(all(0<=x['refund_fraction']<=1 for x in r))
        self.assertEqual(r[-1]['fixed_outflow_krw'],0)

if __name__=='__main__':unittest.main()
