"""기준 가정의 보험료. 수지 검증·민감도는 별도 단계에서 수행한다."""
import json
import math
import hashlib
from dataclasses import asdict
from models.rider.cashflows import CashflowSettings, project_cashflows, summarize
from models.rider.start_integration import ROOT, read_starts, read_mortality
from models.treatment.model import Parameters


def calculate_premiums(summary):
    pure_denominator=summary['pv_premium_units']-summary['pv_refund_units']
    gross_denominator=summary['pv_net_premium_units']
    if min(pure_denominator,gross_denominator)<=0:
        raise ValueError('보험료 분모가 0 이하입니다')
    pure=summary['pv_benefit_krw']/pure_denominator
    gross=summary['pv_fixed_outflow_krw']/gross_denominator
    return {'monthly_net_premium_krw':pure,'monthly_gross_premium_krw':gross,
            'monthly_gross_premium_rounded_up_krw':math.ceil(gross),
            'pure_denominator':pure_denominator,'gross_denominator':gross_denominator,
            'pricing_basis':'baseline research assumptions; no separate profit/capital loading',
            'net_premium_definition':'benefits and refund proportional to net premium, excluding expenses',
            'status':'baseline price calculated; profit test and scenario comparison pending'}


def main():
    cfg=json.loads((ROOT/'config/cashflows.json').read_text(encoding="utf-8"))
    params=json.loads((ROOT/'config/treatment.json').read_text(encoding="utf-8"))
    s=CashflowSettings(**cfg);t=Parameters(**params)
    starts=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv','base')
    summary=summarize(project_cashflows(starts,read_mortality(),s,t),s.initial_policies)
    result=calculate_premiums(summary)
    inputs=['config/cashflows.json','config/treatment.json','outputs/start_rate/start_probabilities.csv','data/reference/mortality.csv']
    result.update(scenario='base',contract_assumptions=cfg,treatment_assumptions=params,
        input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},cashflow_summary=summary)
    out=ROOT/'outputs/pricing';out.mkdir(parents=True,exist_ok=True)
    (out/'base_premium.json').write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','contract_assumptions','treatment_assumptions','cashflow_summary']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
