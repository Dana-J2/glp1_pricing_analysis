import unittest
from models.treatment.model import Parameters, DailyTransitionKernel, EpisodeState
from models.rider.contract_state import *

class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.k=DailyTransitionKernel(Parameters(stop_reduction=1),500)
    def test_waiting_and_exhaustion(self):
        states={ContractState():1.};days=[];disp=[]
        for day in range(500):
            states,e=step_population(states,day,1.,self.k,BenefitRules(maturity_day=500))
            if e['paid_claims']:days.append(day)
            if e['dispenses']:disp.append(day)
            self.assertAlmostEqual(sum(states.values()),1)
        self.assertEqual(days,list(range(112,449,28)))
        self.assertEqual(len(days),13)
        self.assertEqual(len(disp),17) # 대기 중 4회는 지급한도에서 제외
        self.assertTrue(all(s.status=='exhausted' and s.paid_claims==13 for s in states))
    def test_exact_waiting_boundary(self):
        for day,expect in [(89,False),(90,True)]:
            branches=advance_contract(ContractState(),day,1.,self.k)
            self.assertEqual(branches[0].paid_today,expect)
    def test_maturity_before_dispense(self):
        b=advance_contract(ContractState(),100,1.,self.k,BenefitRules(maturity_day=100))[0]
        self.assertEqual(b.state.status,'matured');self.assertFalse(b.paid_today)
    def test_restart_preserves_claim_count(self):
        k=DailyTransitionKernel(Parameters(),200)
        # 마지막 조제 0일, 재개 가능 88일. 지급 12회 이력인 가상 상태.
        s=ContractState('active',12,EpisodeState('off',-1,0,False))
        bs=advance_contract(s,90,1.,k)
        resumed=[b for b in bs if b.restarted_today]
        self.assertTrue(resumed)
        self.assertTrue(all(b.state.paid_claims==13 and b.state.status=='exhausted' for b in resumed))
    def test_different_histories_stay_separate(self):
        t=EpisodeState('on',0,0)
        ss={ContractState('active',1,t):.4,ContractState('active',2,t):.6}
        r,e=step_population(ss,28,0.,self.k,BenefitRules(waiting_days=0))
        self.assertEqual({s.paid_claims for s in r},{2,3})
    def test_stochastic_mass_and_repeat_starts(self):
        k=DailyTransitionKernel(Parameters(),200);s={ContractState():1.};new=paid=0.
        for d in range(150):
            s,e=step_population(s,d,.02,k,BenefitRules(max_claims=2,waiting_days=0))
            new+=e['new_starters'];paid+=e['paid_claims']
            self.assertAlmostEqual(sum(s.values()),1,places=10)
        self.assertLessEqual(new,1);self.assertLessEqual(paid,2)
    def test_terminal_and_exit(self):
        for status in ['dead','lapsed','exhausted','matured']:
            b=advance_contract(ContractState(status,0),10,1.,self.k)[0]
            self.assertFalse(b.dispensed_today)
        s=apply_exit({ContractState():1.},.1,'dead')
        self.assertAlmostEqual(sum(m for t,m in s.items() if t.status=='dead'),.1)
        self.assertAlmostEqual(sum(s.values()),1.)

if __name__=='__main__':unittest.main()
