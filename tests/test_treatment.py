"""Reproducibility and invariant checks for the deterministic treatment module."""
from pathlib import Path
import json
import math
import unittest
import numpy as np
from models.treatment.model import DailyTransitionKernel, Parameters, reference_persistence, episode_survival, month_to_day, restart_hazard, treatment_pathway, validate_expected_path, first_starts_from_probabilities

class TestTreatmentPath(unittest.TestCase):
    def test_date_anchor(self):
        self.assertEqual([month_to_day(x) for x in [3,6,9,12,24]],[91,183,274,365,730])
    def test_clinical_anchors_and_ratio(self):
        p=Parameters()
        s=reference_persistence(800,p)
        for m,expected in [(0,1),(3,.87),(6,.73),(9,.61),(12,.51)]:
            self.assertAlmostEqual(s[month_to_day(m)],expected,places=12)
        self.assertAlmostEqual(s[730],.51*(.156/.352),places=12)
        self.assertLess(s[731],s[730])
    def test_restart_window_fraction(self):
        p=Parameters()
        q=restart_hazard(p)
        self.assertAlmostEqual(1-(1-q)**90,.25,places=12)
    def test_zero_restart_persistence_and_dispense_timing(self):
        p=Parameters(restart_in_window_fraction=0)
        x=np.zeros(800);x[0]=1
        r=treatment_pathway(x,p)
        self.assertTrue(validate_expected_path(r)['passed'])
        self.assertEqual(r['expected_dispenses'][0],1.0)
        self.assertAlmostEqual(float(r['expected_dispenses'][28]),float(reference_persistence(800,p)[28]),places=12)
        self.assertAlmostEqual(float(np.sum(r['expected_dispenses'][1:28])),0.0,places=12)
        self.assertTrue(np.allclose(r['expected_restarts'],0))
        for m,expected in [(3,.87),(6,.73),(9,.61),(12,.51)]:
            self.assertAlmostEqual(r['on_treatment'][month_to_day(m)],reference_persistence(800,p)[month_to_day(m)//28*28],places=10)
        self.assertAlmostEqual(r['on_treatment'][730],reference_persistence(800,p)[728],places=10)
        self.assertAlmostEqual(r['on_treatment'][365]+r['off_treatment'][365],1.0,places=12)
    def test_restart_grace_and_mass_and_no_spurious_restarts(self):
        p=Parameters()
        x=np.zeros(350);x[0]=1
        r=treatment_pathway(x,p)
        self.assertTrue(validate_expected_path(r)['passed'])
        self.assertAlmostEqual(float(np.sum(r['expected_restarts'][:88])),0.0,places=12)
        self.assertGreater(float(np.sum(r['expected_restarts'][88:])),0.0)
        self.assertTrue(np.all(r['expected_dispenses']>=0))
        self.assertTrue(np.all(r['expected_dispenses']<=1+1e-10))
    def test_multi_start_and_mass_conservation(self):
        p=Parameters()
        x=np.zeros(300);x[0]=2.0;x[4]=1.0;x[62]=.5
        r=treatment_pathway(x,p)
        self.assertTrue(validate_expected_path(r)['passed'])
        self.assertAlmostEqual(r['cumulative_first_starts'][-1],3.5,places=12)
        self.assertAlmostEqual(r['expected_dispenses'][0],2.,places=12)
    def test_effect_of_stop_reduction_and_restart_hazard_multiplier(self):
        x=np.zeros(365);x[0]=1
        base=treatment_pathway(x,Parameters(restart_in_window_fraction=0))
        less_stop=treatment_pathway(x,Parameters(restart_in_window_fraction=0,stop_reduction=.2))
        self.assertGreater(less_stop['on_treatment'][-1],base['on_treatment'][-1])
    def test_haein_first_year_starts(self):
        p=Path(__file__).resolve().parents[1]/'outputs/start_rate/start_probabilities.csv'
        start_no,_=first_starts_from_probabilities(p,'base','no',10000.)
        start_yes,_=first_starts_from_probabilities(p,'base','yes',10000.)
        self.assertAlmostEqual(start_no[:365].sum()/10000*100,2.455795,delta=.00002)
        self.assertAlmostEqual(start_yes[:365].sum()/10000*100,2.504414,delta=.00002)

    def test_daily_state_kernel_and_external_history(self):
        pp=Parameters(restart_in_window_fraction=0)
        kernel=DailyTransitionKernel(pp,200)
        states={kernel.initial_state(0,external_history=('paid-0',)):1.0}
        active_byday=[];disps=[]
        for day in range(100):
            active_byday.append(sum(w for st,w in states.items() if st.mode=='on'))
            new={};today_disp=0.
            for st,w in states.items():
                for b in kernel.advance(st,day):
                    amount=w*b.probability
                    new[b.next_state]=new.get(b.next_state,0.)+amount
                    today_disp+=amount*float(b.dispensed_today)
                    self.assertEqual(b.next_state.external_history,('paid-0',))
            disps.append(today_disp)
            states=new
        s=reference_persistence(100,pp)
        self.assertAlmostEqual(active_byday[91],s[84],places=10)
        self.assertAlmostEqual(disps[0],1,places=10)
        self.assertAlmostEqual(disps[28],s[28],places=10)
        self.assertAlmostEqual(disps[84],s[84],places=10)
        self.assertAlmostEqual(disps[85],0,places=10)


    def test_no_stop_before_runout(self):
        x=np.zeros(120);x[0]=1
        r=treatment_pathway(x)
        self.assertTrue(np.allclose(r['on_treatment'][:28],1))
        self.assertTrue(np.allclose(r['expected_stops'][:28],0))
        self.assertGreater(r['expected_stops'][28],0)
        self.assertAlmostEqual(r['expected_stops'][28]+r['expected_dispenses'][28],1)

    def test_probability_adjustment_formula(self):
        p=Parameters(stop_reduction=.2)
        ref=reference_persistence(400,p)
        q=1-ref[1:]/ref[:-1]
        actual=episode_survival(400,p,.75)
        self.assertTrue(np.allclose(1-actual[1:]/actual[:-1],q*.8*.75,atol=1e-14))
        with self.assertRaises(ValueError):episode_survival(400,p,10000)

    def test_restart_kernel_matches_aggregate(self):
        T=260;p=Parameters(stop_reduction=.2,postrestart_stop_hazard_multiplier=.75)
        kernel=DailyTransitionKernel(p,T+1)
        states={kernel.initial_state(0,('paid',0)):1.}
        ds=[];rs=[];ons=[]
        for day in range(T):
            nxt={};d=r=0.
            for st,mass in states.items():
                branches=kernel.advance(st,day)
                self.assertAlmostEqual(sum(b.probability for b in branches),1.)
                for b in branches:
                    mass2=mass*b.probability
                    nxt[b.next_state]=nxt.get(b.next_state,0)+mass2
                    d+=mass2*b.dispensed_today;r+=mass2*b.restarted_today
                    self.assertEqual(b.next_state.external_history,('paid',0))
            states=nxt
            self.assertAlmostEqual(sum(states.values()),1.)
            ds.append(d);rs.append(r);ons.append(sum(m for st,m in states.items() if st.mode=='on'))
        x=np.zeros(T);x[0]=1
        agg=treatment_pathway(x,p)
        for a,b in [(ds,agg['expected_dispenses']),(rs,agg['expected_restarts']),(ons,agg['on_treatment'])]:
            self.assertTrue(np.allclose(a,b,atol=1e-12))

if __name__=='__main__':unittest.main(verbosity=2)
