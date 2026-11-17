"""목표이익 보험료와 확정 보험료 기준 경험 시나리오 수지를 비교한다.

가격은 base 가정에서 한 번만 산출한다. 경험 시나리오 비교에서는 원 단위
올림 보험료를 고정하고 시작률 경로만 교체한다.
"""
import csv
import json
import math

from models.rider.cashflows import CashflowSettings, project_cashflows, summarize
from models.rider.pricing import calculate_premiums
from models.rider.start_integration import ROOT, read_mortality, read_starts
from models.rider.validate_pricing import pv_components, reconstruct
from models.treatment.model import Parameters


EXPENSE_KEYS=(
    'claim_expense','maintenance','acquisition_fixed',
    'acquisition_variable','collection',
)


def evaluate_projected_cashflows(projected_rows,settings,gross_premium):
    """사건·노출에서 보험료별 현금흐름을 재구성해 현가를 직접 합산한다."""
    pv=pv_components(reconstruct(projected_rows,settings,gross_premium))
    premium_income=pv['premium']
    if premium_income<=0:
        raise ValueError('보험료 수입 현가가 0 이하입니다')
    expenses=math.fsum(pv[key] for key in EXPENSE_KEYS)
    return {
        'premium_income_pv_krw':premium_income,
        'benefit_pv_krw':pv['benefit'],
        'expenses_pv_krw':expenses,
        'refund_pv_krw':pv['refund'],
        'total_outflow_pv_krw':pv['total_outflow'],
        'profit_pv_krw':pv['balance'],
        'profit_margin':pv['balance']/premium_income,
        **{f'{key}_pv_krw':pv[key] for key in EXPENSE_KEYS},
    }


def _write_csv(path,rows):
    if not rows:
        raise ValueError(f'{path.name}에 저장할 결과가 없습니다')
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _check(checks,name,condition,detail=''):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})


def build_analysis(config=None):
    cfg=config or json.loads((ROOT/'config/profit_analysis.json').read_text())
    cashflow_cfg=json.loads((ROOT/'config/cashflows.json').read_text())
    treatment_cfg=json.loads((ROOT/'config/treatment.json').read_text())
    settings=CashflowSettings(**cashflow_cfg)
    treatment=Parameters(**treatment_cfg)
    mortality=read_mortality()
    margins=[float(x) for x in cfg['target_profit_margins']]
    scenarios=list(cfg['experience_scenarios'])
    pricing_scenario=cfg['pricing_scenario']
    if pricing_scenario not in scenarios:
        raise ValueError('가격 산출 시나리오가 경험 시나리오 목록에 없습니다')
    if len(set(scenarios))!=len(scenarios):
        raise ValueError('경험 시나리오가 중복되었습니다')

    projected={}
    summaries={}
    for scenario in scenarios:
        starts=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv',scenario)
        rows=project_cashflows(starts,mortality,settings,treatment)
        projected[scenario]=rows
        summaries[scenario]=summarize(rows,settings.initial_policies)

    base_summary=summaries[pricing_scenario]
    prices=[calculate_premiums(base_summary,margin) for margin in margins]
    zero_price=next(p for p in prices if p['target_profit_margin']==0)
    zero_g=zero_price['monthly_gross_premium_krw']
    premium_rows=[]
    for price in prices:
        exact_g=price['monthly_gross_premium_krw']
        rounded_g=price['monthly_gross_premium_rounded_up_krw']
        exact=evaluate_projected_cashflows(projected[pricing_scenario],settings,exact_g)
        rounded=evaluate_projected_cashflows(projected[pricing_scenario],settings,rounded_g)
        increase=exact_g-zero_g
        premium_rows.append({
            'pricing_scenario':pricing_scenario,
            'target_profit_margin':price['target_profit_margin'],
            'target_profit_margin_percent':100*price['target_profit_margin'],
            'monthly_net_premium_krw':price['monthly_net_premium_krw'],
            'monthly_gross_premium_unrounded_krw':exact_g,
            'monthly_gross_premium_rounded_up_krw':rounded_g,
            'increase_vs_zero_krw':increase,
            'increase_vs_zero_percent':100*increase/zero_g,
            'unrounded_premium_income_pv_krw':exact['premium_income_pv_krw'],
            'unrounded_total_outflow_pv_krw':exact['total_outflow_pv_krw'],
            'unrounded_profit_pv_krw':exact['profit_pv_krw'],
            'unrounded_achieved_profit_margin':exact['profit_margin'],
            'rounded_premium_income_pv_krw':rounded['premium_income_pv_krw'],
            'rounded_total_outflow_pv_krw':rounded['total_outflow_pv_krw'],
            'rounded_profit_pv_krw':rounded['profit_pv_krw'],
            'rounded_achieved_profit_margin':rounded['profit_margin'],
            'annual_discount_rate':settings.interest,
            'initial_policies':settings.initial_policies,
            'cohort_basis':cfg['cohort_basis'],
            'rounding':'ceil to whole KRW',
        })

    scenario_rows=[]
    for price in prices:
        fixed_g=price['monthly_gross_premium_rounded_up_krw']
        for scenario in scenarios:
            evaluation=evaluate_projected_cashflows(projected[scenario],settings,fixed_g)
            summary=summaries[scenario]
            scenario_rows.append({
                'pricing_target_profit_margin':price['target_profit_margin'],
                'pricing_target_profit_margin_percent':100*price['target_profit_margin'],
                'fixed_monthly_gross_premium_krw':fixed_g,
                'premium_rounding':'ceil to whole KRW',
                'pricing_scenario':pricing_scenario,
                'experience_scenario':scenario,
                'premium_income_pv_krw':evaluation['premium_income_pv_krw'],
                'benefit_pv_krw':evaluation['benefit_pv_krw'],
                'expenses_pv_krw':evaluation['expenses_pv_krw'],
                'refund_pv_krw':evaluation['refund_pv_krw'],
                'total_outflow_pv_krw':evaluation['total_outflow_pv_krw'],
                'profit_pv_krw':evaluation['profit_pv_krw'],
                'profit_margin':evaluation['profit_margin'],
                'total_starts':summary['total_starts'],
                'total_paid_claims':summary['total_paid_claims'],
                'annual_discount_rate':settings.interest,
                'initial_policies':settings.initial_policies,
                'cohort_basis':cfg['cohort_basis'],
                'experience_scope':'start-rate scenario only; common treatment persistence',
            })

    matrix=[]
    for scenario in scenarios:
        row={'experience_scenario':scenario}
        for result in (x for x in scenario_rows if x['experience_scenario']==scenario):
            label=f"target_{result['pricing_target_profit_margin_percent']:g}pct".replace('.','p')
            row[f'{label}_fixed_premium_krw']=result['fixed_monthly_gross_premium_krw']
            row[f'{label}_profit_pv_krw']=result['profit_pv_krw']
            row[f'{label}_profit_margin']=result['profit_margin']
        matrix.append(row)

    checks=[]
    saved=json.loads((ROOT/'reports/base_premium.json').read_text())
    _check(checks,'0% 기존 영업보험료 재현',
           abs(zero_g-saved['monthly_gross_premium_krw'])<1e-10,
           f'{zero_g:.12f}원')
    max_target_error=max(abs(r['unrounded_achieved_profit_margin']-r['target_profit_margin'])
                         for r in premium_rows)
    _check(checks,'올림 전 목표이익률 달성',max_target_error<1e-12,
           f'최대 오차 {max_target_error:.3g}')
    _check(checks,'목표이익률 증가 시 보험료 증가',
           all(a['monthly_gross_premium_krw']<b['monthly_gross_premium_krw']
               for a,b in zip(prices,prices[1:])))
    base_experience=[r for r in scenario_rows if r['experience_scenario']==pricing_scenario]
    behavior_unchanged=(len({r['total_starts'] for r in base_experience})==1
                        and len({r['total_paid_claims'] for r in base_experience})==1
                        and len({r['benefit_pv_krw'] for r in base_experience})==1)
    _check(checks,'목표이익률 변경 시 행동·급부 불변',behavior_unchanged)
    proportional_error=0.
    for price in prices:
        g=price['monthly_gross_premium_krw']
        pv=pv_components(reconstruct(projected[pricing_scenario],settings,g))
        for component,unit_key in [('refund','pv_refund_units'),('collection','pv_collection_units'),
                                   ('acquisition_variable','pv_acquisition_units')]:
            proportional_error=max(proportional_error,abs(pv[component]-g*base_summary[unit_key]))
    _check(checks,'보험료 연동 사업비·반환금 재계산',proportional_error<.01,
           f'최대 현가 차이 {proportional_error:.6f}원')
    fixed_ok=all(len({r['fixed_monthly_gross_premium_krw'] for r in scenario_rows
                      if r['pricing_target_profit_margin']==margin})==1 for margin in margins)
    _check(checks,'경험 시나리오별 확정 보험료 고정',fixed_ok)
    formula_error=0.
    for row in scenario_rows:
        summary=summaries[row['experience_scenario']];g=row['fixed_monthly_gross_premium_krw']
        formula_profit=g*summary['pv_net_premium_units']-summary['pv_fixed_outflow_krw']
        formula_error=max(formula_error,abs(row['profit_pv_krw']-formula_profit))
    _check(checks,'직접 합산 손익과 G×D−C 일치',formula_error<.01,
           f'최대 현가 차이 {formula_error:.6f}원')
    try:
        calculate_premiums({'pv_premium_units':100.,'pv_refund_units':0.,
                            'pv_net_premium_units':5.},.05)
    except ValueError as exc:
        invalid_denominator='분모가 0 이하' in str(exc)
    else:
        invalid_denominator=False
    _check(checks,'목표이익 보험료 분모 0 이하 거부',invalid_denominator)
    _check(checks,'3개 보험료×6개 경험 시나리오',len(scenario_rows)==18,
           f'{len(scenario_rows)}개 조합')
    validation={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL',
                'checks':checks,'formula':'G=C/(D-m*A)',
                'profit_margin_definition':cfg['profit_margin_definition'],
                'scope':cfg['scope']}
    return premium_rows,scenario_rows,matrix,validation


def main():
    premium_rows,scenario_rows,matrix,validation=build_analysis()
    out=ROOT/'outputs/profit_analysis';out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/'premium_comparison.csv',premium_rows)
    _write_csv(out/'fixed_premium_scenario_results.csv',scenario_rows)
    _write_csv(out/'scenario_profit_summary.csv',matrix)
    for name,value in [('premium_comparison.json',premium_rows),
                       ('fixed_premium_scenario_results.json',scenario_rows),
                       ('validation.json',validation)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    print(json.dumps({'premium_comparison':premium_rows,
                      'scenario_profit_summary':matrix,
                      'validation':validation},ensure_ascii=False,indent=2))
    if validation['status']!='PASS':
        raise AssertionError('목표이익·고정 보험료 수지 비교 검증 실패')


if __name__=='__main__':main()
