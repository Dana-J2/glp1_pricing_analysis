"""Filled research scenarios for first Wegovy dispensing, version 2.0.

All market counts, off-label shares, prior-use fractions, enrollment selection,
price response and forward trends are explicitly CONDITIONAL assumptions.
This is not an estimated Korean uptake model or a premium/claims engine.
The normalized cohort has no death/lapse/eligibility exits. Daily initiation
hazards are exported for a later contract model to add competing events.
"""
from dataclasses import dataclass,asdict,replace
from datetime import date,timedelta
from math import exp,expm1,log,log1p,isfinite
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parents[2]
POP_DETAIL=json.loads((ROOT/'data/reference/nondiabetic_population.json').read_text(encoding='utf-8'))['populations']
PROXY={k:v['all'] for k,v in POP_DETAIL.items()}
TARGET={k:v['nonDM'] for k,v in POP_DETAIL.items()}

@dataclass(frozen=True)
class Scenario:
    name:str='baseline'
    status:str='conditional_research_assumption_not_estimate'
    annual_first_people:int=120000
    outside_eligibility_fraction:float=.20
    newly_eligible_fraction_of_valid_market_starts:float=0.
    ever_used_fraction:float=.05
    population_proxy:str='diagnosis_proxy'
    selection_ratios:tuple=(1.5,)*10
    nondiabetic_relative_start:float=1.
    annual_market_odds_change:float=0.
    price_per_28days:float=400000.
    benefit_per_28days:float=100000.
    waiting_days:int=90
    lifetime_payment_limit:int=13
    decision_horizon_units:int=13
    perceived_benefit_fraction:float=.75
    initiation_hazard_price_response:float=.1
    anticipation_before_waiting:bool=True
    decision_discount_annual:float=0.
    N:float=10000.
    contract_start:str='2026-11-15'
    contract_years:int=10
    death_lapse_eligibility_exit_in_this_normalization:float=0.
    other_behavior_channels_in_this_module:float=0.
    poststart_persistence_effect_in_this_module:float=0.

    def validate(self):
        for key,value in asdict(self).items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and not isfinite(value):raise ValueError(key)
        for key in ['outside_eligibility_fraction','newly_eligible_fraction_of_valid_market_starts','ever_used_fraction','perceived_benefit_fraction']:
            if not 0<=getattr(self,key)<=1:raise ValueError(key)
        if self.ever_used_fraction>=1:raise ValueError('No naive risk set')
        if self.nondiabetic_relative_start<=0:raise ValueError('Positive relative nonDM initiation needed')
        if self.population_proxy not in PROXY:raise ValueError('Unknown population proxy')
        if len(self.selection_ratios)!=self.contract_years or any(not isfinite(x) or x<=0 for x in self.selection_ratios):raise ValueError('Selection profile')
        if self.annual_market_odds_change<=-1:raise ValueError('Market trend')
        if self.price_per_28days<=0 or self.benefit_per_28days<0:raise ValueError('Price/benefit')
        if self.initiation_hazard_price_response<0 or self.decision_discount_annual<0:raise ValueError('Response/discount')
        for key in ['waiting_days','lifetime_payment_limit','decision_horizon_units','contract_years']:
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError(key)
        if self.decision_horizon_units<1 or self.contract_years<1 or self.N<0 or self.annual_first_people<0:raise ValueError('Cohort/horizon/count')
        if not 0<=self.national_reference_probability()<1:raise ValueError('Implied annual probability')
        for key in ['death_lapse_eligibility_exit_in_this_normalization','other_behavior_channels_in_this_module','poststart_persistence_effect_in_this_module']:
            if getattr(self,key)!=0:raise ValueError('This module excludes '+key+'; use the full contract engine')
        date.fromisoformat(self.contract_start)
        return self

    def national_reference_probability(self):
        pop=POP_DETAIL[self.population_proxy]
        # s_nonDM / s_DM = r; unknown DM status is assigned s_DM as a scenario
        # convention, not declared non-diabetic. Prior-use fraction common by group.
        effective=pop['nonDM']+(pop['DM']+pop['unknown_DM'])/self.nondiabetic_relative_start
        market=self.annual_first_people*(1-self.outside_eligibility_fraction)*(1-self.newly_eligible_fraction_of_valid_market_starts)
        return market/(effective*(1-self.ever_used_fraction))

def boundaries(s):
    start=date.fromisoformat(s.contract_start)
    if start.month==2 and start.day==29:raise ValueError('Use a non-February-29 reference start date')
    return [start.replace(year=start.year+y) for y in range(s.contract_years+1)]

def annual_reference(s,year):
    p=s.national_reference_probability()
    odds=p/(1-p)*(1+s.annual_market_odds_change)**year*s.selection_ratios[year]
    return odds/(1+odds)

def decision_support(s,elapsed_days):
    """Perceived subsidy / planned course cost; NOT expected actual claims.

    For price perception only, the customer plans H consecutive 28-day units.
    No assertion that all H are actually dispensed. Reinitiation is not used to
    create another first start. Actual continuation must be modeled separately.
    """
    b=boundaries(s);end=(b[-1]-b[0]).days
    cost=0.;subsidy=0.;paid=0
    for k in range(s.decision_horizon_units):
        d=elapsed_days+28*k
        discount=(1+s.decision_discount_annual)**(-(28*k)/365.25)
        cost+=s.price_per_28days*discount
        if d>=s.waiting_days and d<end and paid<s.lifetime_payment_limit:
            subsidy+=s.benefit_per_28days*discount;paid+=1
    perceived=subsidy*s.perceived_benefit_fraction
    if elapsed_days<s.waiting_days and not s.anticipation_before_waiting:perceived=0.
    share=perceived/cost
    if not 0<=share<1:raise ValueError('Perceived effective price must remain positive')
    return {'planned_cost_KRW':cost,'planned_payable_units':paid,'planned_subsidy_KRW':subsidy,
            'perceived_subsidy_KRW':perceived,'effective_support_share':share,
            'hazard_multiplier':(1-share)**(-s.initiation_hazard_price_response)}

def run(s,keep_daily=False):
    s.validate();b=boundaries(s);n0=n1=s.N;years=[];daily=[];cum0=cum1=0.
    for y in range(s.contract_years):
        days=(b[y+1]-b[y]).days;a0=annual_reference(s,y);h0=-log1p(-a0)/365.
        A=-expm1(-h0*days)
        open0,open1=n0,n1;starts0=starts1=0.;sum_mult=0.
        for k in range(days):
            dt=b[y]+timedelta(days=k);elapsed=(dt-b[0]).days
            support=decision_support(s,elapsed);mult=support['hazard_multiplier'];sum_mult+=mult
            q0=-expm1(-h0);q1=-expm1(-h0*mult)
            new0=n0*q0;new1=n1*q1;n0-=new0;n1-=new1;starts0+=new0;starts1+=new1
            if keep_daily:daily.append({'scenario':s.name,'date':dt.isoformat(),'elapsed_days':elapsed,'policy_year':y+1,
              'daily_hazard_without':h0,'daily_hazard_with':h0*mult,'conditional_daily_q_without':q0,'conditional_daily_q_with':q1,
              'a_without_annual365':a0,'a_price_with_annual365':-expm1(-365*h0*mult),
              'reference_A_calendar_year_no_exit':A,'eligible_naive_remaining_without':n0,'eligible_naive_remaining_with':n1,
              'first_starts_without':new0,'first_starts_with':new1,**support})
        p1=-expm1(-h0*sum_mult)
        if abs(p1-A)<1e-14:p1=A
        cum0+=starts0;cum1+=starts1
        years.append({'scenario':s.name,'year':y+1,'N_open_without':open0,'N_open_with':open1,
          'a_without_annual365_percent':a0*100,'a_with_equivalent_annual365_percent':-expm1(-365*h0*sum_mult/days)*100,
          'conditional_A_percent':A*100,'conditional_with_percent':p1*100,'conditional_B_pp':(p1-A)*100,
          'first_starts_without':starts0,'first_starts_with':starts1,'difference_in_year_starts':starts1-starts0,
          'cumulative_without':cum0,'cumulative_with':cum1,'cumulative_net_extra':cum1-cum0,
          'N_close_without':n0,'N_close_with':n1,
          'scope':'same initial eligible-naive cohort; no contract exits; no actual claims or mortality effects'})
    summary={'scenario':s.name,'status':s.status,'reference_population_nonDM':TARGET[s.population_proxy],
      'reference_population_all_eligible_proxy':PROXY[s.population_proxy],
      'reference_naive_population_nonDM':TARGET[s.population_proxy]*(1-s.ever_used_fraction),
      'assumed_eligible_start_people_all_DM_groups':s.annual_first_people*(1-s.outside_eligibility_fraction),
      'assumed_nonDM_first_people':s.national_reference_probability()*TARGET[s.population_proxy]*(1-s.ever_used_fraction),
      'national_reference_A_percent':s.national_reference_probability()*100,
      'year1_A_percent':years[0]['conditional_A_percent'],'year1_with_percent':years[0]['conditional_with_percent'],
      'year1_B_pp':years[0]['conditional_B_pp'],'year1_without':years[0]['first_starts_without'],
      'year1_with':years[0]['first_starts_with'],'year1_net_extra':years[0]['difference_in_year_starts'],
      'year10_cumulative_without':cum0,'year10_cumulative_with':cum1,'year10_cumulative_net_extra':cum1-cum0,
      'price_per_28days':s.price_per_28days,'response':s.initiation_hazard_price_response,'perception':s.perceived_benefit_fraction,
      'all_values_conditional_not_confidence_bounds':True}
    return summary,years,daily

