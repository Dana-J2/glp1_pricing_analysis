import unittest
from datetime import date
from models.rider.start_integration import *

class StartChecks(unittest.TestCase):
    def setUp(self):
        self.rows=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv');self.q=read_mortality()
    def test_reproduction(self):
        for path,target in [('no',.02455795483492669),('yes',.025044136443112053)]:
            r=project_starts(self.rows,self.q,path=path,mortality_enabled=False)
            self.assertAlmostEqual(sum(x['new_starters'] for x in r[:365])/1000,target,places=12)
    def test_mass_and_naive(self):
        r=project_starts(self.rows,self.q,monthly_lapse=.01)
        for x in r:
            self.assertLess(abs(x['mass_error']),1e-8)
            self.assertLessEqual(x['new_starters'],x['eligible_naive_before_start'])
            self.assertGreaterEqual(x['naive_close'],0)
        for a,b in zip(r,r[1:]):self.assertAlmostEqual(a['naive_close'],b['naive_open'])
    def test_anniversary(self):
        r=project_starts(self.rows,self.q)
        self.assertEqual(r[364]['insurance_age'],49)
        self.assertEqual(r[365]['insurance_age'],50)
    def test_lapse_on_payment_days(self):
        r=project_starts(self.rows,self.q,monthly_lapse=.01)
        for x in r:
            if x['lapse_naive']+x['lapse_started']>0:
                self.assertEqual(date.fromisoformat(x['date']).day,15)
        self.assertEqual(r[0]['lapse_naive'],0)
    def test_analytic_deaths(self):
        r=project_starts(self.rows,self.q)
        survival=1.
        for x in r:survival*=1-x['daily_death_q']
        self.assertAlmostEqual(r[-1]['naive_close']+r[-1]['started_alive_close'],1000*survival,places=8)
    def test_all_start_once(self):
        rows=[dict(r,p_start_yes=1.) for r in self.rows[:30]]
        r=project_starts(rows,self.q)
        self.assertEqual(r[0]['new_starters'],1000)
        self.assertEqual(sum(x['new_starters'] for x in r[1:]),0)
        self.assertGreater(r[0]['deaths_started'],0)
    def test_zero_response(self):
        rows=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv','response_zero')
        a=project_starts(rows,self.q,path='no');b=project_starts(rows,self.q,path='yes')
        self.assertEqual([x['new_starters'] for x in a],[x['new_starters'] for x in b])

if __name__=='__main__':unittest.main(verbosity=2)
