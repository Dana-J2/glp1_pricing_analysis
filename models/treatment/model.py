"""Treatment/refill pathway module for GLP-1 pricing academic project.

Clinical anchors: Mailhac et al. JAMA Netw Open 2026 (first 12m semaglutide),
Rodriguez et al. JAMA Netw Open 2025 (12-24m conditional persistence,
multiple GLP-1 RAs). Model assumptions stated in README.md and params.json.

Outputs are expected patient counts/fractions, NOT actual patient trajectories.
No claim/benefit, death, lapse, underwriting, or insurer profitability here.

Design:
* Continuous daily persistence hazards calibrated to survival anchors.
* One dispensing at each episode start, then every 28 days if still on course.
* Discontinuation occurs at runout, before the next scheduled refill.
* No refill after discontinuation. Last fill's 28-day possession completes
  before the 60-day grace clock and 90-day restart opportunity start.
* Restart resets the episode persistence clock. Repeated restarts allowed.
* Deterministic expected cohorts: each episode birth date is a distinct cohort,
  but paths are tracked in aggregate, not Monte Carlo.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import date, timedelta
import csv
import math
import json
from pathlib import Path
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class Parameters:
    days_per_28day_dispense: int = 28
    first_year_3m_persistence: float = 0.87
    first_year_6m_persistence: float = 0.73
    first_year_9m_persistence: float = 0.61
    first_year_12m_persistence: float = 0.51
    us_nondiabetic_persistence_12m: float = 0.352
    us_nondiabetic_persistence_24m: float = 0.156
    gap_days_after_runout: int = 60
    restart_window_days: int = 90
    restart_in_window_fraction: float = 0.25
    stop_reduction: float = 0.0
    postrestart_stop_hazard_multiplier: float = 1.0  # legacy key; now DAILY PROBABILITY multiplier

    def validate(self):
        vals=[self.first_year_3m_persistence,self.first_year_6m_persistence,
              self.first_year_9m_persistence,self.first_year_12m_persistence]
        assert 1.0>vals[0]>vals[1]>vals[2]>vals[3]>0, "Invalid first-year persistence"
        assert 0<self.us_nondiabetic_persistence_24m<self.us_nondiabetic_persistence_12m<=1
        assert self.days_per_28day_dispense>0
        assert self.gap_days_after_runout>=1
        assert self.restart_window_days>=1
        assert 0<=self.restart_in_window_fraction<1
        assert 0<=self.stop_reduction<=1
        assert self.postrestart_stop_hazard_multiplier>=0


def month_to_day(month: int) -> int:
    """Study-time mapping: months*365/12 rounded HALF-UP to whole days.

    Month 3->91, 6->183, 9->274, 12->365, 24->730 days.
    Model treatment time, not dateutil/calendar-month anniversary.
    """
    return int(math.floor(month * 365.0/12.0 + 0.5))


def reference_persistence(days: int, params: Parameters) -> np.ndarray:
    """S(t) at all integer days 0...days, 0-12m linearly interpolated,
    12-24m constant conditional hazard (Rodriguez ratio), beyond copied hazard.
    """
    params.validate()
    anchors=np.asarray([0,month_to_day(3),month_to_day(6),
                         month_to_day(9),month_to_day(12)],dtype=np.float64)
    fractions=np.asarray([1.0,params.first_year_3m_persistence,
                          params.first_year_6m_persistence,params.first_year_9m_persistence,
                          params.first_year_12m_persistence],dtype=np.float64)
    all_days=np.arange(days+1,dtype=np.float64)
    out=np.interp(all_days,anchors,fractions)
    m12=month_to_day(12)
    after=all_days>m12
    r=params.us_nondiabetic_persistence_24m/params.us_nondiabetic_persistence_12m
    out[after]=fractions[-1]*np.power(r,(all_days[after]-m12)/365.0)
    # This copies the 12-24-month daily hazard beyond 24 months.
    return out


def episode_survival(days:int, params:Parameters, multiplier:float=1.0)->np.ndarray:
    """Adjust DAILY conditional stop probabilities, then accumulate survival."""
    s=reference_persistence(days,params)
    q=1.0-s[1:]/s[:-1]
    adjusted=q*(1.0-params.stop_reduction)*multiplier
    if not np.all(np.isfinite(adjusted)) or np.any((adjusted<0)|(adjusted>=1)):
        raise ValueError("Adjusted daily stop probability must be in [0,1); no silent clipping")
    out=np.empty(days+1,dtype=float)
    out[0]=1
    out[1:]=np.cumprod(1.0-adjusted)
    return out


def restart_hazard(params:Parameters)->float:
    return 1.0 - (1.0-params.restart_in_window_fraction)**(1.0/params.restart_window_days)


def treatment_pathway(first_starts:Iterable[float], params:Parameters=Parameters(),
                       initial_date:date=date(2026,11,15))->dict[str,np.ndarray]:
    """Deterministic expected treatment path for an arbitrary daily start stream.

    Expected-person mass (including fractional cohorts) is conserved. If one
    person starts on day0, on_treatment+off_treatment=1 each day. For multiple
    daily first starts, the sum equals cumulative ever-started by that day.

    Complexity O(number_of_days^2), memory O(number_of_days).
    Inputs first_starts[t] are already estimated first initiators, NOT hazards.
    """
    params.validate()
    starts0=np.array(list(first_starts),dtype=float)
    if starts0.ndim!=1 or len(starts0)<1 or np.any(starts0<0) or not np.all(np.isfinite(starts0)):
        raise ValueError('first_starts must be a finite nonnegative 1D daily stream')
    T=len(starts0)
    surv0=episode_survival(T+1,params,1.0)
    surv1=episode_survival(T+1,params,params.postrestart_stop_hazard_multiplier)
    starts1=np.zeros(T,dtype=float)
    eligible_inflow=np.zeros(T+1,dtype=float)
    eligible_expiry=np.zeros(T+1,dtype=float)
    p_restart=restart_hazard(params)
    pool_after=0.0
    decay_to_expiry=(1-p_restart)**params.restart_window_days
    on0=np.zeros(T)
    on1=np.zeros(T)
    off=np.zeros(T)
    restarts=np.zeros(T)
    dispenses=np.zeros(T)
    stops=np.zeros(T)
    eligible_mass=np.zeros(T)
    cumulative_original=np.cumsum(starts0)
    for t in range(T):
        if p_restart>0:
            pool_start=pool_after+eligible_inflow[t]-eligible_expiry[t]
            if pool_start<-1e-10:
                raise AssertionError(f'negative eligible pool at day {t}: {pool_start}')
            pool_start=max(pool_start,0.0)
            starts1[t]=p_restart*pool_start
            pool_after=pool_start-starts1[t]
            restarts[t]=starts1[t]
            eligible_mass[t]=pool_start
        age=np.arange(t,-1,-1,dtype=int)  # index i: age=t-i
        c0=starts0[:t+1]
        c1=starts1[:t+1]
        grid_age=(age//params.days_per_28day_dispense)*params.days_per_28day_dispense
        z0=c0*surv0[grid_age]
        z1=c1*surv1[grid_age]
        on0[t]=z0.sum()
        on1[t]=z1.sum()
        off[t]=cumulative_original[t]-on0[t]-on1[t]
        # Refill exactly on the episode's day 0,28,56,... while active.
        due=(age%params.days_per_28day_dispense)==0
        dispenses[t]=z0[due].sum()+z1[due].sum()
        # Runout at the START of the refill day; a stopped branch gets no refill.
        boundary=(age>0)&due
        previous=np.maximum(grid_age-params.days_per_28day_dispense,0)
        total_stops=np.where(boundary,
            c0*(surv0[previous]-surv0[grid_age])+c1*(surv1[previous]-surv1[grid_age]),0.0)
        stops[t]=total_stops.sum()
        eligible_day=t+params.gap_days_after_runout
        if p_restart>0 and eligible_day<T:
            if params.gap_days_after_runout==0:
                raise ValueError("Zero-day gap requires a separate same-day restart rule")
            eligible_inflow[eligible_day]+=stops[t]
            expiry=eligible_day+params.restart_window_days
            if expiry<T:
                eligible_expiry[expiry]+=stops[t]*decay_to_expiry
        if off[t]<-1e-8*max(1,cumulative_original[t]):
            raise AssertionError(f'probability mass violation: day {t} off {off[t]}')
    cumulative_dispenses=np.cumsum(dispenses)
    return dict(
       elapsed_day=np.arange(T,dtype=int),
       date=np.asarray([(initial_date+timedelta(days=i)).isoformat() for i in range(T)]),
       first_starts=starts0,
       first_course_survival=np.asarray([surv0[min(i,len(surv0)-1)] for i in range(T)]),
       first_episode_on_treatment=on0,
       restarted_episode_on_treatment=on1,
       on_treatment=on0+on1,
       off_treatment=off,
       expected_stops=stops,
       expected_restarts=restarts,
       expected_dispenses=dispenses,
       cumulative_dispenses=cumulative_dispenses,
       cumulative_first_starts=cumulative_original,
       eligible_to_restart=eligible_mass,
    )


def first_starts_from_probabilities(csv_path:str|Path, scenario:str='base',
                                   path:str='yes', N:float=10000)->tuple[np.ndarray,date]:
    """Import Haein's daily conditional probability CSV; mortality/lapse excluded.

    Each day: first starts = yet-unstarted eligible people * p_start_path.
    This is just a start-cohort interface. Insurance mortality/lapse must be
    inserted by the integrator, and stop/restarts do not return to 'unstarted'.
    """
    if path not in ('yes','no'):raise ValueError('path must be yes or no')
    if N<0:raise ValueError('N must be nonnegative')
    rows=[]
    with open(csv_path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['scenario']==scenario:rows.append(r)
    if not rows:raise ValueError('Scenario not found '+scenario)
    rows.sort(key=lambda r:int(r['elapsed_day']))
    if [int(r['elapsed_day']) for r in rows]!=list(range(len(rows))):
        raise ValueError('Missing/duplicate elapsed_day')
    p=np.array([float(r['p_start_'+path]) for r in rows])
    if np.any((p<0)|(p>1)):raise ValueError('Daily hazard outside [0,1]')
    remaining=float(N)
    starts=[]
    for prob in p:
        x=remaining*prob
        starts.append(x)
        remaining-=x
    return np.array(starts),date.fromisoformat(rows[0]['date'])


def write_csv(path:Path, data:dict[str,np.ndarray], columns:list[str]|None=None):
    cols=columns or list(data)
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(cols)
        for i in range(len(data[cols[0]])):
            w.writerow([f'{float(data[c][i]):.12g}' if isinstance(data[c][i],(float,np.floating)) else data[c][i] for c in cols])


def validate_expected_path(r:dict[str,np.ndarray],atol=1e-6)->dict:
    target=r['cumulative_first_starts']
    residue=target-r['on_treatment']-r['off_treatment']
    return {
      'mass_conservation_max_absolute_error':float(np.max(np.abs(residue))),
      'minimum_on_treatment':float(np.min(r['on_treatment'])),
      'minimum_off_treatment':float(np.min(r['off_treatment'])),
      'minimum_expected_dispenses':float(np.min(r['expected_dispenses'])),
      'minimum_expected_restarts':float(np.min(r['expected_restarts'])),
      'total_expected_first_starts':float(r['first_starts'].sum()),
      'total_expected_restarts':float(r['expected_restarts'].sum()),
      'total_expected_dispenses':float(r['expected_dispenses'].sum()),
      'end_on_treatment':float(r['on_treatment'][-1]),
      'end_off_treatment':float(r['off_treatment'][-1]),
      'passed': bool(np.max(np.abs(residue))<atol and min(np.min(r[k]) for k in ('on_treatment','off_treatment','expected_dispenses','expected_restarts'))>=-atol)
    }


def params_from_json(path:str|Path)->Parameters:
    with open(path,encoding='utf-8') as f:return Parameters(**json.load(f))



@dataclass(frozen=True)
class EpisodeState:
    """Hashable treatment state. Keep `external_history` distinct for later payout integration.

    Current scope handles treatment only; insurance integrator can branch this
    external tuple after a refill without merging different paid-claim histories.
    """
    mode: str
    episode_start_day: int = -1
    last_dispense_day: int = -1
    after_restart: bool = False
    external_history: tuple = ()


@dataclass(frozen=True)
class EpisodeBranch:
    """One possible next-day state and its fractional transition probability."""
    probability: float
    next_state: EpisodeState
    dispensed_today: bool
    restarted_today: bool


class DailyTransitionKernel:
    """Fine-grained state API for later payment-history-aware integration.

    State enters at START of day t, branches into states at START of day t+1.
    At runout, stop is evaluated BEFORE refill. Output state is after today events.
    Use external_history key for payment histories, DO NOT drop it on merge.
    """
    def __init__(self,params:Parameters=Parameters(),max_age:int=3654):
        params.validate()
        self.params=params
        self.s0=episode_survival(max_age+1,params,1.0)
        self.s1=episode_survival(max_age+1,params,params.postrestart_stop_hazard_multiplier)
        self.q=restart_hazard(params)

    def initial_state(self,first_day:int,external_history:tuple=()):
        return EpisodeState('on',first_day,first_day,False,external_history)

    def _on_branches(self,s:EpisodeState,day:int,outer_prob:float=1.0,restarted_today:bool=False):
        age=day-s.episode_start_day
        if age<0:raise ValueError('episode cannot begin in future')
        surv=self.s1 if s.after_restart else self.s0
        if age>=len(surv):raise ValueError('max_age too small')
        interval=self.params.days_per_28day_dispense
        if age==0:
            return [EpisodeBranch(outer_prob,s,True,restarted_today)]
        if age%interval:
            return [EpisodeBranch(outer_prob,s,False,restarted_today)]
        # Product of adjusted daily survival probabilities over the previous supply.
        q_stop=1.0-surv[age]/surv[age-interval]
        stay=EpisodeState('on',s.episode_start_day,day,s.after_restart,s.external_history)
        leave=EpisodeState('off',-1,s.last_dispense_day,s.after_restart,s.external_history)
        return [EpisodeBranch(outer_prob*(1-q_stop),stay,True,restarted_today),
                EpisodeBranch(outer_prob*q_stop,leave,False,restarted_today)]

    def advance(self,state:EpisodeState,day:int)->list[EpisodeBranch]:
        if state.mode=='on':return self._on_branches(state,day)
        if state.mode!='off':raise ValueError('mode must be on or off')
        eligible=state.last_dispense_day+self.params.days_per_28day_dispense+self.params.gap_days_after_runout
        if not(eligible<=day<eligible+self.params.restart_window_days) or self.q==0:
            return [EpisodeBranch(1.0,state,False,False)]
        stayed=EpisodeBranch(1.0-self.q,state,False,False)
        resumed=EpisodeState('on',day,day,True,state.external_history)
        return [stayed]+self._on_branches(resumed,day,outer_prob=self.q,restarted_today=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='GLP-1 treatment pathway after first initiation')
    parser.add_argument('--params',default='params.json')
    parser.add_argument('--mode',choices=['one','haein'],default='one')
    parser.add_argument('--days',type=int,default=3653)
    parser.add_argument('--scenario',default='base')
    parser.add_argument('--path',choices=['yes','no'],default='yes')
    parser.add_argument('--N',type=float,default=10000)
    parser.add_argument('--starts-csv',default='input_reference/start_probabilities.csv')
    parser.add_argument('--output',default='outputs/treatment_example.csv')
    args=parser.parse_args()
    pp=params_from_json(args.params)
    if args.mode=='one':
        starts=np.zeros(args.days);starts[0]=1
        start_date=date(2026,11,15)
    else:
        starts,start_date=first_starts_from_probabilities(args.starts_csv,args.scenario,args.path,args.N)
    result=treatment_pathway(starts,pp,start_date)
    cols=['elapsed_day','date','on_treatment','off_treatment','expected_restarts','expected_dispenses','cumulative_dispenses',
          'first_course_survival','expected_stops','first_episode_on_treatment','restarted_episode_on_treatment','cumulative_first_starts']
    write_csv(Path(args.output),result,cols)
    print(json.dumps(validate_expected_path(result),ensure_ascii=False,indent=2))
