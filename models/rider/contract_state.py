"""치료 전이에 지급 횟수와 특약 상태를 붙인다. 납입·반환·사업비는 별도 단계."""
from dataclasses import dataclass, replace
import math
from models.treatment.model import DailyTransitionKernel, EpisodeState

@dataclass(frozen=True)
class ContractState:
    status: str = 'active'  # active / exhausted / matured / dead / lapsed
    paid_claims: int = 0
    treatment: EpisodeState | None = None  # None = 최초 미시작

@dataclass(frozen=True)
class ContractBranch:
    probability: float
    state: ContractState
    started_today: bool = False
    restarted_today: bool = False
    dispensed_today: bool = False
    paid_today: bool = False
    exhausted_today: bool = False

@dataclass(frozen=True)
class BenefitRules:
    waiting_days: int = 90
    max_claims: int = 13
    maturity_day: int = 3653  # 이 날짜부터 지급 없음


def advance_contract(state, day, p_start, kernel, rules=BenefitRules()):
    """당일 급부까지 처리. 확률질량에 이 함수를 적용한 후 구간 사망을 한 번 반영한다."""
    if not math.isfinite(p_start) or not 0<=p_start<=1:raise ValueError('시작확률')
    if rules.max_claims<1 or rules.waiting_days<0 or rules.maturity_day<1:raise ValueError('지급 규칙')
    if state.paid_claims<0 or state.paid_claims>rules.max_claims:raise ValueError('지급 횟수')
    if state.status not in {'active','exhausted','matured','dead','lapsed'}:raise ValueError('계약 상태')
    if state.status!='active':return [ContractBranch(1.,state)]
    if state.paid_claims==rules.max_claims:raise ValueError('소진 계약이 active로 남아 있음')
    if day>=rules.maturity_day:
        return [ContractBranch(1.,replace(state,status='matured',treatment=None))]
    branches=[]
    if state.treatment is None:
        if state.paid_claims:raise ValueError('미시작 계약에 지급 이력 존재')
        branches.append(ContractBranch(1-p_start,state))
        origins=[(p_start,kernel.initial_state(day),True)]
    else:origins=[(1.,state.treatment,False)]
    for weight,treatment,started in origins:
        if not weight:continue
        for b in kernel.advance(treatment,day):
            paid=b.dispensed_today and day>=rules.waiting_days
            count=state.paid_claims+int(paid)
            exhausted=count==rules.max_claims
            # 지급 이력이 ContractState의 키에 포함되어 서로 다른 이력이 합쳐지지 않는다.
            next_state=ContractState('exhausted' if exhausted else 'active',count,
                                     None if exhausted else b.next_state)
            branches.append(ContractBranch(weight*b.probability,next_state,started,
                b.restarted_today,b.dispensed_today,paid,exhausted))
    return [b for b in branches if b.probability>0]


def apply_exit(states, probability, status):
    """납입 직전 해지 또는 급부 후 사망에 사용. 종료된 특약은 추가로 이동시키지 않는다."""
    if status not in ('dead','lapsed'):raise ValueError('이탈 종류')
    if not math.isfinite(probability) or not 0<=probability<=1:raise ValueError('이탈 확률')
    result={}
    for state,mass in states.items():
        if state.status!='active':
            result[state]=result.get(state,0)+mass;continue
        survivor=mass*(1-probability)
        exited=replace(state,status=status,treatment=None)
        result[state]=result.get(state,0)+survivor
        result[exited]=result.get(exited,0)+mass*probability
    return {s:m for s,m in result.items() if m>0}


def step_population(states,day,p_start,kernel,rules):
    result={};events=dict(new_starters=0.,restarts=0.,dispenses=0.,paid_claims=0.,exhaustions=0.)
    for state,mass in states.items():
        for b in advance_contract(state,day,p_start,kernel,rules):
            amount=mass*b.probability
            result[b.state]=result.get(b.state,0)+amount
            for key,flag in [('new_starters',b.started_today),('restarts',b.restarted_today),
                             ('dispenses',b.dispensed_today),('paid_claims',b.paid_today),
                             ('exhaustions',b.exhausted_today)]:
                events[key]+=amount*flag
    return result,events
