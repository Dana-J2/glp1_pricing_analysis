"""저장된 기준 보험료를 잠근 채 날짜별 현금흐름으로 수지상등을 검증한다."""
import csv
import hashlib
import json
import math
from datetime import date
from models.rider.cashflows import CashflowSettings, project_cashflows, month_date
from models.rider.start_integration import ROOT, read_starts, read_mortality
from models.treatment.model import Parameters


def reconstruct(rows,s,g):
    # 계산 엔진의 net_premium_units·fixed_outflow를 쓰지 않고 사건·노출에서 재구성한다.
    start=date.fromisoformat(rows[0]['date']);pay_index=0;parts=[]
    for r in rows:
        dt=date.fromisoformat(r['date']);d=(dt-start).days
        payday=dt==month_date(start,pay_index) and r['matured_today']==0 and d<len(rows)-1
        if payday:pay_index+=1
        exposure=r['active_open'] if payday else 0.
        discount=(1+s.interest)**(-d/365.)
        # 납입 주기의 실제 달력 일수로 반환 비율을 독립 계산한다.
        m=(dt.year-start.year)*12+dt.month-start.month
        if dt<month_date(start,m):m-=1
        prev=month_date(start,max(m,0));nxt=month_date(start,max(m,0)+1)
        refund=g*r['exhausted_today']*(nxt-dt).days/(nxt-prev).days if s.refund_unused else 0.
        parts.append(dict(date=dt.isoformat(),premium=g*exposure,
            benefit=s.benefit*r['paid_claims'],claim_expense=s.claim_cost*r['paid_claims'],
            maintenance=s.beta*exposure,acquisition_fixed=s.acquisition_fixed*s.initial_policies if d==0 else 0.,
            acquisition_variable=12*s.alpha*g*s.initial_policies if d==0 else 0.,
            collection=s.gamma*g*exposure,refund=refund,discount=discount))
    return parts


def pv_components(parts):
    keys=['premium','benefit','claim_expense','maintenance','acquisition_fixed','acquisition_variable','collection','refund']
    vals={k:math.fsum(r[k]*r['discount'] for r in parts) for k in keys}
    vals['total_outflow']=math.fsum(vals[k] for k in keys if k!='premium')
    vals['balance']=vals['premium']-vals['total_outflow']
    return vals


def run_validation():
    price=json.loads((ROOT/'outputs/pricing/base_premium.json').read_text())
    checks=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,passed=bool(ok),detail=detail))
    for path,expected in price['input_sha256'].items():
        check('입력 일치: '+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected)
    if not all(c['passed'] for c in checks):raise ValueError('보험료 산출 당시 입력과 현재 입력이 다릅니다')
    s=CashflowSettings(**price['contract_assumptions']);t=Parameters(**price['treatment_assumptions'])
    rows=project_cashflows(read_starts(ROOT/'outputs/start_rate/start_probabilities.csv',price['scenario']),read_mortality(),s,t)
    g=price['monthly_gross_premium_krw'];p=price['monthly_net_premium_krw']
    parts=reconstruct(rows,s,g);gross=pv_components(parts)
    check('영업보험료 수지상등',abs(gross['balance'])<.01,f"1,000명 기준 잔차 {gross['balance']:.10f}원")
    # 순보험료 자체에 비례한 반환이라는 정의로 점검. 실제 고객 납입은 G이다.
    pureparts=reconstruct(rows,s,p);pure=pv_components(pureparts)
    pure_balance=pure['premium']-pure['benefit']-pure['refund']
    check('순보험료 수지상등',abs(pure_balance)<.01,f'{pure_balance:.10f}원')
    expected_components={'benefit':'pv_benefit_krw','claim_expense':'pv_claim_expense_krw',
      'maintenance':'pv_maintenance_krw','acquisition_fixed':'pv_acquisition_fixed_krw'}
    for key,old in expected_components.items():
        check('지출 재구성: '+key,abs(gross[key]-price['cashflow_summary'][old])<.01)
    for key,old in [('premium','pv_premium_units'),('refund','pv_refund_units'),('acquisition_variable','pv_acquisition_units'),('collection','pv_collection_units')]:
        check('보험료 계수 재구성: '+key,abs(gross[key]-g*price['cashflow_summary'][old])<.01)
    # 나눗셈 산식 대신 수입·지출 균형점을 이분법으로 탐색한다.
    unit=pv_components(reconstruct(rows,s,1.));zero=pv_components(reconstruct(rows,s,0.))
    def balance(x):return zero['balance']+x*(unit['balance']-zero['balance'])
    lo,hi=0.,max(g*2,1.)
    for _ in range(80):
        mid=(lo+hi)/2
        if balance(mid)>0:hi=mid
        else:lo=mid
    solved=(lo+hi)/2
    check('재구성 현금흐름 균형점 비교',abs(solved-g)<1e-5,f'{solved:.8f}원')
    rounded=pv_components(reconstruct(rows,s,price['monthly_gross_premium_rounded_up_krw']))
    check('원 단위 올림 후 잔차 방향',rounded['balance']>=0)
    check('만기일 현금흐름 0',all(parts[-1][k]==0 for k in ['premium','benefit','refund','maintenance','collection','claim_expense','acquisition_fixed','acquisition_variable']))
    out=ROOT/'outputs/pricing_validation';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,
      monthly_gross_premium=g,monthly_net_premium=p,pv_gross=gross,pure_balance=pure_balance,
      independently_solved_gross=solved,rounded_pv_balance=rounded['balance'],
      scope='산출 검증; 경험 수지·민감도·자본·준비금 검증 제외')
    (out/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (out/'pv_breakdown.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['item','cohort_pv_krw','per_policy_pv_krw'])
        for k,v in gross.items():w.writerow([k,v,v/s.initial_policies])
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['status']!='PASS':raise AssertionError('산출 검증 실패')
    return report

if __name__=='__main__':run_validation()
