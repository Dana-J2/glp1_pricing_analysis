"""기준 가정의 보험료. 목표이익은 영업보험료에만 반영한다."""
import argparse
import json
import math
import hashlib
from models.rider.cashflows import CashflowSettings, project_cashflows, summarize
from models.rider.start_integration import ROOT, read_starts, read_mortality
from models.treatment.model import Parameters


def achieved_profit_margin(summary, gross_premium):
    """보험료 수입 현가를 분모로 한 연구용 현금흐름 이익률."""
    premium_income=gross_premium*summary['pv_premium_units']
    if premium_income<=0:
        raise ValueError('보험료 수입 현가가 0 이하입니다')
    profit=gross_premium*summary['pv_net_premium_units']-summary['pv_fixed_outflow_krw']
    return profit/premium_income


def calculate_premiums(summary,target_profit_margin=0.):
    if not math.isfinite(target_profit_margin) or not 0<=target_profit_margin<1:
        raise ValueError('목표이익률은 0 이상 1 미만이어야 합니다')
    pure_denominator=summary['pv_premium_units']-summary['pv_refund_units']
    # A=총 납입현가 계수, D=비례 지출 차감 후 순수입현가 계수.
    # 목표식 (G*D-C)/(G*A)=m 에서 G=C/(D-m*A).
    gross_denominator=(summary['pv_net_premium_units']
                       -target_profit_margin*summary['pv_premium_units'])
    if min(pure_denominator,gross_denominator)<=0:
        raise ValueError('목표이익 반영 보험료 분모가 0 이하입니다')
    pure=summary['pv_benefit_krw']/pure_denominator
    gross=summary['pv_fixed_outflow_krw']/gross_denominator
    rounded=math.ceil(gross)
    return {'monthly_net_premium_krw':pure,'monthly_gross_premium_krw':gross,
            'monthly_gross_premium_rounded_up_krw':rounded,
            'target_profit_margin':target_profit_margin,
            'achieved_profit_margin':achieved_profit_margin(summary,gross),
            'rounded_achieved_profit_margin':achieved_profit_margin(summary,rounded),
            'pure_denominator':pure_denominator,'gross_denominator':gross_denominator,
            'pv_premium_units':summary['pv_premium_units'],
            'pv_net_premium_units':summary['pv_net_premium_units'],
            'pv_fixed_outflow_krw':summary['pv_fixed_outflow_krw'],
            'pricing_basis':'baseline research assumptions; target cashflow profit margin; no tax/capital loading',
            'net_premium_definition':'benefits and refund proportional to net premium, excluding expenses',
            'status':'baseline price calculated with explicit target cashflow profit margin'}


def main(target_profit_margin=None):
    if target_profit_margin is None:
        parser=argparse.ArgumentParser(description='기준 가정 월 보험료 산출')
        parser.add_argument('--target-profit-margin',type=float,default=0.,
                            help='보험료 수입 현가 대비 목표이익률 (기본 0)')
        target_profit_margin=parser.parse_args().target_profit_margin
    cfg=json.loads((ROOT/'config/cashflows.json').read_text())
    params=json.loads((ROOT/'config/treatment.json').read_text())
    s=CashflowSettings(**cfg);t=Parameters(**params)
    starts=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv','base')
    summary=summarize(project_cashflows(starts,read_mortality(),s,t),s.initial_policies)
    result=calculate_premiums(summary,target_profit_margin)
    inputs=['config/cashflows.json','config/treatment.json','outputs/start_rate/start_probabilities.csv','data/reference/mortality.csv']
    result.update(scenario='base',contract_assumptions=cfg,treatment_assumptions=params,
        input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},cashflow_summary=summary)
    out=ROOT/'outputs/pricing';out.mkdir(parents=True,exist_ok=True)
    if target_profit_margin==0:
        filename='base_premium.json'
    else:
        label=f'{target_profit_margin*100:g}'.replace('.','p')
        filename=f'base_premium_target_profit_{label}pct.json'
    (out/filename).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','contract_assumptions','treatment_assumptions','cashflow_summary']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
