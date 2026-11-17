"""일별 계약 현금흐름. 보험료 G에 무관한 지출과 G의 계수를 따로 계산한다."""
from dataclasses import dataclass, asdict
from datetime import date, timedelta
import calendar
import json
import math
import numpy as np
from models.treatment.model import Parameters, episode_survival, restart_hazard
from models.rider.start_integration import ROOT, read_starts, read_mortality, save

@dataclass(frozen=True)
class CashflowSettings:
    initial_policies: float = 1000.
    entry_age: int = 49
    sex: str = '여'
    mortality_multiplier: float = 1.
    monthly_lapse: float = 0.
    death_enabled: bool = True
    waiting_days: int = 90
    max_claims: int = 13
    benefit: float = 100000.
    acquisition_fixed: float = 10000.
    alpha: float = .30
    beta: float = 500.
    gamma: float = .02
    claim_cost: float = 2000.
    interest: float = .0275
    refund_unused: bool = True

    def validate(self):
        for name in ('initial_policies','mortality_multiplier','benefit','acquisition_fixed','alpha','beta','gamma','claim_cost'):
            v=getattr(self,name)
            if not math.isfinite(v) or v<0:raise ValueError(name)
        if self.gamma>=1:raise ValueError('수금비율')
        if self.monthly_lapse!=0:raise ValueError('해약환급금 규칙 미확정: 현재 현금흐름은 해지율 0만 지원')
        if not math.isfinite(self.interest) or self.interest<=-1:raise ValueError('예정이율')
        if not isinstance(self.waiting_days,int) or self.waiting_days<0:raise ValueError('대기기간')
        if not isinstance(self.max_claims,int) or self.max_claims<1:raise ValueError('지급한도')
        if not isinstance(self.entry_age,int) or self.entry_age<0:raise ValueError('보험나이')

def month_date(start, offset):
    n=start.year*12+start.month-1+offset
    y,m=divmod(n,12);m+=1
    return date(y,m,min(start.day,calendar.monthrange(y,m)[1]))

def project_cashflows(rows, mortality, settings=CashflowSettings(), treatment=Parameters(), path='yes'):
    """행 마지막 다음날을 만기로 처리. 종료일 행을 추가해 잔여 active를 matured로 옮긴다.

    상태 배열: on[지급횟수, 치료경과일, 최초/재개], off[지급횟수, 중단경과일].
    치료 개시일 자체 대신 경과일로 묶되 지급횟수와 재개 여부를 보존한다.
    재개 창이 지난 off_expired도 특약은 유지되어 납입 대상이다.
    """
    settings.validate();treatment.validate()
    if path not in ('yes','no') or not rows:raise ValueError('시작 경로')
    start=date.fromisoformat(rows[0]['date']);T=len(rows);end=start+timedelta(days=T)
    for i,row in enumerate(rows):
        if int(row['elapsed_day'])!=i or date.fromisoformat(row['date'])!=start+timedelta(days=i):
            raise ValueError('날짜 경로')
        if not math.isfinite(row['p_start_'+path]) or not 0<=row['p_start_'+path]<=1:raise ValueError('시작확률')
    M=settings.max_claims;D=treatment.days_per_28day_dispense
    # 매 조제 후 유효 급부가 증가하므로 한도·대기기간으로 필요한 치료 나이를 제한할 수 있다.
    A=min(T+1, settings.waiting_days+D*(M+1)+2)
    K=treatment.gap_days_after_runout+treatment.restart_window_days
    on=np.zeros((M,A,2));off=np.zeros((M,K));expired=np.zeros(M)
    naive=settings.initial_policies;dead=exhausted=0.;paid_cum=starts_cum=recipients_cum=0.
    survival=[episode_survival(A,treatment,1.),episode_survival(A,treatment,treatment.postrestart_stop_hazard_multiplier)]
    due=np.arange(D,A,D)
    stop_q=np.stack([1-s[due]/s[due-D] for s in survival],axis=1)
    restart_q=restart_hazard(treatment)
    result=[];payment_index=0;last_pay=start;next_pay=month_date(start,1)
    for day,row in enumerate(rows):
        dt=start+timedelta(days=day)
        active_open=naive+on.sum()+off.sum()+expired.sum()
        payday=(dt==month_date(start,payment_index))
        if payday:
            last_pay=dt;next_pay=month_date(start,payment_index+1);payment_index+=1
        # Collect premiums before the final claim; refund unused days afterward.
        premium_units=active_open if payday else 0.
        maintenance=settings.beta*active_open if payday else 0.
        acquisition=settings.acquisition_fixed*settings.initial_policies if day==0 else 0.
        acquisition_units=12*settings.alpha*settings.initial_policies if day==0 else 0.
        # First starts apply only to naive policies; restarts are separate.
        new=naive*row['p_start_'+path];naive-=new;on[0,0,0]+=new
        starts_cum+=new
        # Stop at runout, before the next refill.
        stops=on[:,due,:]*stop_q[None,:,:]
        on[:,due,:]-=stops;off[:,0]+=stops.sum(axis=(1,2))
        eligible=off[:,treatment.gap_days_after_runout:]
        resumes=eligible*restart_q
        off[:,treatment.gap_days_after_runout:]-=resumes
        resumed=resumes.sum(axis=1);on[:,0,1]+=resumed
        # Dispense at episode start, restart and surviving 28-day boundaries.
        due_all=np.arange(0,A,D)
        dispense=float(on[:,due_all,:].sum());paid=exhaust_today=new_recipients=0.
        if day>=settings.waiting_days:
            fills=on[:,due_all,:].copy()
            paid=float(fills.sum());new_recipients=float(fills[0].sum())
            exhaust_today=float(fills[-1].sum())
            on[:,due_all,:]=0.
            on[1:,due_all,:]+=fills[:-1]
            exhausted+=exhaust_today;paid_cum+=paid;recipients_cum+=new_recipients
        # Refund [exhaustion day, next payment day); same-day exhaustion refunds all.
        frac=(next_pay-dt).days/(next_pay-last_pay).days
        refund_units=exhaust_today*frac if settings.refund_unused else 0.
        year=dt.year-start.year-((dt.month,dt.day)<(start.month,start.day))
        annual_q=mortality[(settings.entry_age+year,settings.sex)] if settings.death_enabled else 0.
        if not 0<=annual_q<1 or not math.isfinite(annual_q):raise ValueError('Invalid mortality probability')
        q=-math.expm1(math.log1p(-annual_q)*settings.mortality_multiplier/365.)
        active_after=naive+on.sum()+off.sum()+expired.sum()
        death_today=active_after*q;dead+=death_today
        naive*=1-q;on*=1-q;off*=1-q;expired*=1-q
        # Advance clocks; missing the restart window does not terminate coverage.
        if on[:,-1,:].sum()>1e-9:raise AssertionError('Treatment age array too short')
        on[:,1:,:]=on[:,:-1,:].copy();on[:,0,:]=0
        expired+=off[:,-1];off[:,1:]=off[:,:-1].copy();off[:,0]=0
        active_close=naive+on.sum()+off.sum()+expired.sum()
        benefit=settings.benefit*paid;claim_expense=settings.claim_cost*paid
        collection_units=settings.gamma*premium_units
        fixed_out=benefit+claim_expense+maintenance+acquisition
        net_units=premium_units-refund_units-collection_units-acquisition_units
        mass_error=settings.initial_policies-active_close-dead-exhausted
        if abs(mass_error)>1e-7*max(1,settings.initial_policies):raise AssertionError('인원 합계')
        result.append(dict(date=dt.isoformat(),elapsed_day=day,active_open=float(active_open),
            new_starters=float(new),restarts=float(resumed.sum()),dispenses=dispense,paid_claims=paid,
            new_recipients=new_recipients,exhausted_today=exhaust_today,
            deaths_today=float(death_today),active_close=float(active_close),
            naive_close=float(naive),cumulative_starts=float(starts_cum),cumulative_paid_claims=float(paid_cum),
            cumulative_recipients=float(recipients_cum),
            cumulative_exhausted=float(exhausted),cumulative_deaths=float(dead),matured_today=0.,
            premium_units=float(premium_units),refund_fraction=float(frac),refund_units=float(refund_units),
            acquisition_units=float(acquisition_units),collection_units=float(collection_units),
            benefit_krw=benefit,claim_expense_krw=claim_expense,maintenance_krw=float(maintenance),
            acquisition_fixed_krw=acquisition,fixed_outflow_krw=float(fixed_out),net_premium_units=float(net_units),
            discount=(1+settings.interest)**(-day/365.),mass_error=float(mass_error)))
    # No premium or benefit at maturity; close the remaining active policies.
    terminal={k:0. for k in result[-1]}
    terminal.update(date=end.isoformat(),elapsed_day=T,active_open=float(active_close),
        cumulative_starts=float(starts_cum),cumulative_paid_claims=float(paid_cum),
        cumulative_recipients=float(recipients_cum),
        cumulative_exhausted=float(exhausted),cumulative_deaths=float(dead),matured_today=float(active_close),
        discount=(1+settings.interest)**(-T/365.),mass_error=float(settings.initial_policies-active_close-dead-exhausted))
    result.append(terminal)
    return result


def summarize(rows,initial_policies):
    keys=['premium_units','refund_units','acquisition_units','collection_units','benefit_krw',
          'claim_expense_krw','maintenance_krw','acquisition_fixed_krw','fixed_outflow_krw','net_premium_units']
    out={f'pv_{k}':sum(r[k]*r['discount'] for r in rows) for k in keys}
    out.update(initial_policies=initial_policies,total_starts=rows[-1]['cumulative_starts'],
        recipients=rows[-1]['cumulative_recipients'],total_paid_claims=rows[-1]['cumulative_paid_claims'],
        exhausted=rows[-1]['cumulative_exhausted'],
        deaths=rows[-1]['cumulative_deaths'],matured=rows[-1]['matured_today'],
        max_mass_error=max(abs(r['mass_error']) for r in rows))
    return out


def main():
    cfg=json.loads((ROOT/'config/cashflows.json').read_text())
    s=CashflowSettings(**cfg);t=Parameters(**json.loads((ROOT/'config/treatment.json').read_text()))
    rows=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv','base')
    result=project_cashflows(rows,read_mortality(),s,t)
    out=ROOT/'outputs/cashflows';out.mkdir(parents=True,exist_ok=True)
    save(out/'base_daily.csv',result)
    summary=summarize(result,s.initial_policies)
    (out/'base_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
