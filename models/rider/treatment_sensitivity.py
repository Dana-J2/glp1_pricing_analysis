"""치료 지속·재개 경험 시나리오를 확정 보험료로 평가한다.

상품별 기준 치료 행태로 가격을 한 번 산출하고 원 단위 올림 보험료를
고정한다. 이후 시작률은 그대로 둔 채 치료 설정만 바꿔 공통 현금흐름
엔진을 시나리오당 한 번 실행하고, 같은 경로를 여러 보험료로 평가한다.
"""
import csv
import json
import math

import numpy as np

from models.rider.cashflows import CashflowSettings,project_cashflows,summarize
from models.rider.pricing import calculate_premiums
from models.rider.profit_analysis import evaluate_projected_cashflows
from models.rider.start_integration import ROOT,read_mortality,read_starts
from models.treatment.model import Parameters,episode_survival,restart_hazard


SENSITIVITY_KEYS={
    'stop_reduction','restart_in_window_fraction',
    'postrestart_stop_hazard_multiplier',
}


def _margin_label(margin):
    return f'{100*margin:g}'.replace('.','p')


def _write_csv(path,rows):
    if not rows:
        raise ValueError(f'{path.name}에 저장할 결과가 없습니다')
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _check(checks,name,condition,detail=''):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})


def build_parameter_cases(base_treatment_config,scenario_definitions):
    """각 시나리오를 기준 설정의 새 복사본에서 생성한다."""
    ids=[item['scenario_id'] for item in scenario_definitions]
    if len(ids)!=len(set(ids)):
        raise ValueError('치료 행태 시나리오 ID가 중복되었습니다')
    cases=[]
    for item in scenario_definitions:
        overrides=dict(item['overrides'])
        unexpected=set(overrides)-SENSITIVITY_KEYS
        if unexpected:
            raise ValueError(f'허용되지 않은 치료 민감도 키: {sorted(unexpected)}')
        resolved=dict(base_treatment_config)
        resolved.update(overrides)
        params=Parameters(**resolved);params.validate()
        cases.append({
            'scenario_id':item['scenario_id'],
            'display_name_ko':item['display_name_ko'],
            'family':item['family'],
            'overrides':overrides,
            'resolved_config':resolved,
            'parameters':params,
        })
    baseline=[case for case in cases if case['family']=='baseline']
    if len(baseline)!=1 or baseline[0]['overrides']:
        raise ValueError('변경값이 없는 기준 치료 행태 시나리오가 하나 필요합니다')
    return cases


def analyze_treatment_scenarios(*,settings,start_rows,mortality,
                                base_treatment_config,scenario_definitions,
                                target_profit_margins,product_id,
                                pricing_start_scenario='base',
                                start_probability_path='yes'):
    """상품·시작률을 외부에서 받아 재사용 가능한 치료 민감도를 계산한다."""
    cases=build_parameter_cases(base_treatment_config,scenario_definitions)
    margins=[float(x) for x in target_profit_margins]
    if margins!=sorted(set(margins)) or 0 not in margins:
        raise ValueError('목표이익률은 중복 없이 오름차순이며 0%를 포함해야 합니다')

    paths={};summaries={};path_stats={}
    for case in cases:
        scenario_id=case['scenario_id']
        projected=project_cashflows(
            start_rows,mortality,settings,case['parameters'],path=start_probability_path,
        )
        paths[scenario_id]=projected
        summaries[scenario_id]=summarize(projected,settings.initial_policies)
        path_stats[scenario_id]={
            'total_dispenses':math.fsum(row['dispenses'] for row in projected),
            'total_restarts':math.fsum(row['restarts'] for row in projected),
        }

    baseline_case=next(case for case in cases if case['family']=='baseline')
    baseline_id=baseline_case['scenario_id']
    prices=[calculate_premiums(summaries[baseline_id],margin) for margin in margins]
    pricing_rows=[]
    for price in prices:
        pricing_rows.append({
            'product_id':product_id,
            'pricing_start_scenario':pricing_start_scenario,
            'start_probability_path':start_probability_path,
            'pricing_treatment_scenario_id':baseline_id,
            'target_profit_margin':price['target_profit_margin'],
            'monthly_net_premium_krw':price['monthly_net_premium_krw'],
            'monthly_gross_premium_unrounded_krw':price['monthly_gross_premium_krw'],
            'monthly_gross_premium_rounded_up_krw':price['monthly_gross_premium_rounded_up_krw'],
            'unrounded_achieved_profit_margin':price['achieved_profit_margin'],
            'rounded_achieved_profit_margin':price['rounded_achieved_profit_margin'],
            'annual_discount_rate':settings.interest,
        })

    results=[]
    for case in cases:
        scenario_id=case['scenario_id'];summary=summaries[scenario_id]
        for price in prices:
            margin=price['target_profit_margin']
            fixed_g=price['monthly_gross_premium_rounded_up_krw']
            evaluation=evaluate_projected_cashflows(paths[scenario_id],settings,fixed_g)
            results.append({
                'analysis_type':'treatment_behavior_sensitivity',
                'product_id':product_id,
                'benefit_per_dispense_krw':settings.benefit,
                'max_claims':settings.max_claims,
                'waiting_days':settings.waiting_days,
                'pricing_start_scenario':pricing_start_scenario,
                'experience_start_scenario':pricing_start_scenario,
                'start_probability_path':start_probability_path,
                'treatment_scenario_id':scenario_id,
                'treatment_scenario_display_name_ko':case['display_name_ko'],
                'treatment_scenario_family':case['family'],
                'stop_reduction':case['parameters'].stop_reduction,
                'restart_in_window_fraction':case['parameters'].restart_in_window_fraction,
                'postrestart_stop_probability_multiplier':case['parameters'].postrestart_stop_hazard_multiplier,
                'pricing_target_profit_margin':margin,
                'fixed_monthly_gross_premium_krw':fixed_g,
                'premium_rounding':'ceil to whole KRW',
                'premium_income_pv_krw':evaluation['premium_income_pv_krw'],
                'benefit_pv_krw':evaluation['benefit_pv_krw'],
                'claim_expense_pv_krw':evaluation['claim_expense_pv_krw'],
                'maintenance_pv_krw':evaluation['maintenance_pv_krw'],
                'acquisition_fixed_pv_krw':evaluation['acquisition_fixed_pv_krw'],
                'acquisition_variable_pv_krw':evaluation['acquisition_variable_pv_krw'],
                'collection_pv_krw':evaluation['collection_pv_krw'],
                'expenses_pv_krw':evaluation['expenses_pv_krw'],
                'refund_pv_krw':evaluation['refund_pv_krw'],
                'total_outflow_pv_krw':evaluation['total_outflow_pv_krw'],
                'profit_pv_krw':evaluation['profit_pv_krw'],
                'scenario_profit_margin':evaluation['profit_margin'],
                'total_starts':summary['total_starts'],
                'total_dispenses':path_stats[scenario_id]['total_dispenses'],
                'total_restarts':path_stats[scenario_id]['total_restarts'],
                'total_paid_claims':summary['total_paid_claims'],
                'exhausted':summary['exhausted'],
                'deaths':summary['deaths'],
                'matured':summary['matured'],
                'pv_premium_units':summary['pv_premium_units'],
                'pv_net_premium_units':summary['pv_net_premium_units'],
                'pv_fixed_outflow_krw':summary['pv_fixed_outflow_krw'],
                'max_mass_error':summary['max_mass_error'],
                'annual_discount_rate':settings.interest,
                'initial_policies':settings.initial_policies,
                'cohort_basis':f'initial {settings.initial_policies:g} policies',
                'experience_scope':'base start rate fixed; treatment persistence and restart only',
            })

    summary_rows=[]
    for case in cases:
        scenario_id=case['scenario_id'];summary=summaries[scenario_id]
        row={
            'product_id':product_id,
            'treatment_scenario_id':scenario_id,
            'treatment_scenario_display_name_ko':case['display_name_ko'],
            'treatment_scenario_family':case['family'],
            'stop_reduction':case['parameters'].stop_reduction,
            'restart_in_window_fraction':case['parameters'].restart_in_window_fraction,
            'postrestart_stop_probability_multiplier':case['parameters'].postrestart_stop_hazard_multiplier,
            'total_starts':summary['total_starts'],
            'total_dispenses':path_stats[scenario_id]['total_dispenses'],
            'total_restarts':path_stats[scenario_id]['total_restarts'],
            'total_paid_claims':summary['total_paid_claims'],
            'exhausted':summary['exhausted'],
            'max_mass_error':summary['max_mass_error'],
        }
        for result in (x for x in results if x['treatment_scenario_id']==scenario_id):
            label=_margin_label(result['pricing_target_profit_margin'])
            row[f'target_{label}pct_fixed_premium_krw']=result['fixed_monthly_gross_premium_krw']
            row[f'target_{label}pct_profit_pv_krw']=result['profit_pv_krw']
            row[f'target_{label}pct_scenario_profit_margin']=result['scenario_profit_margin']
        summary_rows.append(row)
    return {
        'cases':cases,'paths':paths,'summaries':summaries,'prices':prices,
        'pricing_rows':pricing_rows,'results':results,'summary_rows':summary_rows,
        'baseline_id':baseline_id,
    }


def compare_team_reference(analysis,reference_rows):
    by_id={row['treatment_scenario_id']:row for row in analysis['summary_rows']}
    comparisons=[]
    for reference in reference_rows:
        scenario_id=reference['scenario_id'];current=by_id[scenario_id]
        row={
            'treatment_scenario_id':scenario_id,
            'team_display_name':reference['display_name'],
            'current_display_name':current['treatment_scenario_display_name_ko'],
            'team_total_starts':float(reference['total_starts']),
            'current_total_starts':current['total_starts'],
            'starts_difference':current['total_starts']-float(reference['total_starts']),
            'team_total_restarts':float(reference['total_restarts']),
            'current_total_restarts':current['total_restarts'],
            'restarts_difference':current['total_restarts']-float(reference['total_restarts']),
            'team_total_paid_claims':float(reference['total_paid_claims']),
            'current_total_paid_claims':current['total_paid_claims'],
            'paid_claims_difference':current['total_paid_claims']-float(reference['total_paid_claims']),
            'team_exhausted':float(reference['exhausted']),
            'current_exhausted':current['exhausted'],
            'exhausted_difference':current['exhausted']-float(reference['exhausted']),
        }
        for margin in (0.,.05,.10):
            label=_margin_label(margin)
            team_profit=float(reference[f'team_profit_target_{label}pct_krw'])
            team_rate=float(reference[f'team_scenario_margin_target_{label}pct'])
            current_profit=current[f'target_{label}pct_profit_pv_krw']
            current_rate=current[f'target_{label}pct_scenario_profit_margin']
            row[f'team_profit_target_{label}pct_krw']=team_profit
            row[f'current_profit_target_{label}pct_krw']=current_profit
            row[f'profit_difference_target_{label}pct_krw']=current_profit-team_profit
            row[f'team_scenario_margin_target_{label}pct']=team_rate
            row[f'current_scenario_margin_target_{label}pct']=current_rate
            row[f'scenario_margin_difference_target_{label}pct']=current_rate-team_rate
        count_differences=[row['starts_difference'],row['restarts_difference'],
                           row['paid_claims_difference'],row['exhausted_difference']]
        profit_differences=[row[f'profit_difference_target_{_margin_label(m)}pct_krw']
                            for m in (0.,.05,.10)]
        margin_differences=[row[f'scenario_margin_difference_target_{_margin_label(m)}pct']
                            for m in (0.,.05,.10)]
        row['reproduction_status']='PASS' if (
            max(map(abs,count_differences))<1e-9
            and max(map(abs,profit_differences))<.01
            and max(map(abs,margin_differences))<1e-10
        ) else 'FAIL'
        comparisons.append(row)
    return comparisons


def validate_analysis(analysis,base_treatment_config,scenario_definitions,
                      reference_comparisons,reference_rounded_premiums):
    checks=[];results=analysis['results'];summaries=analysis['summaries']
    prices=analysis['prices'];cases=analysis['cases']
    fixed=[p['monthly_gross_premium_rounded_up_krw'] for p in prices]
    expected=[reference_rounded_premiums[_margin_label(p['target_profit_margin'])] for p in prices]
    _check(checks,'기준 상품 0%·5%·10% 보험료 재현',fixed==expected,
           f'{fixed}')

    base_comparison=next(r for r in reference_comparisons
                         if r['treatment_scenario_id']==analysis['baseline_id'])
    max_base_profit_diff=max(
        abs(base_comparison[f'profit_difference_target_{_margin_label(margin)}pct_krw'])
        for margin in (0.,.05,.10)
    )
    _check(checks,'치료 기준 행과 기존 base 수지 일치',max_base_profit_diff<.01,
           f'최대 손익 차이 {max_base_profit_diff:.9f}원')

    claims_unchanged=all(len({r['total_paid_claims'] for r in results
                              if r['treatment_scenario_id']==case['scenario_id']})==1
                         for case in cases)
    _check(checks,'동일 치료 경험의 보험료별 지급 건수 불변',claims_unchanged)
    starts={summary['total_starts'] for summary in summaries.values()}
    _check(checks,'치료 행태 변경 시 최초 시작자 수 불변',len(starts)==1,
           f'{next(iter(starts)):.12f}명')

    probability_ok=True
    for case in cases:
        params=case['parameters']
        for multiplier in (1.,params.postrestart_stop_hazard_multiplier):
            survival=episode_survival(3654,params,multiplier)
            probability_ok &= bool(np.all(np.isfinite(survival))
                                   and np.all((survival>=0)&(survival<=1))
                                   and np.all(np.diff(survival)<=1e-15))
        q_restart=restart_hazard(params)
        probability_ok &= 0<=q_restart<1
    mass_cap_ok=all(summary['max_mass_error']<1e-7
                    and summary['total_paid_claims']<=summary['total_starts']*results[0]['max_claims']+1e-7
                    for summary in summaries.values())
    _check(checks,'전이확률 범위·인원 보존·지급한도',probability_ok and mass_cap_ok,
           f"최대 인원오차 {max(s['max_mass_error'] for s in summaries.values()):.3g}")

    settings_isolated=True
    for case,item in zip(cases,scenario_definitions):
        expected_config=dict(base_treatment_config);expected_config.update(item['overrides'])
        settings_isolated &= case['resolved_config']==expected_config
        for key in SENSITIVITY_KEYS-set(item['overrides']):
            settings_isolated &= case['resolved_config'][key]==base_treatment_config[key]
    _check(checks,'시나리오별 설정 독립 복사',settings_isolated)

    proportional_error=0.;formula_error=0.
    for result in results:
        summary=summaries[result['treatment_scenario_id']]
        g=result['fixed_monthly_gross_premium_krw']
        for field,unit in [('refund_pv_krw','pv_refund_units'),
                           ('collection_pv_krw','pv_collection_units'),
                           ('acquisition_variable_pv_krw','pv_acquisition_units')]:
            proportional_error=max(proportional_error,abs(result[field]-g*summary[unit]))
        formula=g*summary['pv_net_premium_units']-summary['pv_fixed_outflow_krw']
        formula_error=max(formula_error,abs(result['profit_pv_krw']-formula))
    _check(checks,'보험료 연동 비용·반환금 재계산',proportional_error<.01,
           f'최대 차이 {proportional_error:.9f}원')
    _check(checks,'직접 합산 손익과 G×D_s−C_s 일치',formula_error<.01,
           f'최대 차이 {formula_error:.9f}원')

    max_claim_diff=max(abs(r['paid_claims_difference']) for r in reference_comparisons)
    profit_diff_fields=[f'profit_difference_target_{_margin_label(m)}pct_krw' for m in (0.,.05,.10)]
    max_profit_diff=max(abs(r[field]) for r in reference_comparisons for field in profit_diff_fields)
    _check(checks,'팀원 원정밀도 결과 재현',max_claim_diff<1e-9 and max_profit_diff<.01,
           f'지급건수 최대 차이 {max_claim_diff:.3g}, 손익 최대 차이 {max_profit_diff:.9f}원')
    _check(checks,'13개 치료 행태×3개 고정 보험료',len(results)==39,
           f'{len(results)}개 조합; 치료 경로 계산 {len(cases)}회')
    return {
        'status':'PASS' if all(check['passed'] for check in checks) else 'FAIL',
        'checks':checks,
        'analysis_type':'treatment_behavior_sensitivity',
        'scenario_profit_margin_definition':'profit PV / premium income PV',
        'formula':'profit PV = G*D_s-C_s',
        'scope':'base start rate fixed; treatment persistence and restart only; no 6x13 cross stress',
    }


def _read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def _validate_default_product_basis(config,settings):
    basis=config['product_basis']
    actual={'benefit':settings.benefit,'max_claims':settings.max_claims,
            'waiting_days':settings.waiting_days}
    expected={key:basis[key] for key in actual}
    if actual!=expected:
        raise ValueError(f'기준 상품 설정 불일치: {actual} != {expected}')
    start_config=json.loads((ROOT/'config/start_scenarios.json').read_text())
    start_inputs=next(item['inputs'] for item in start_config['scenarios']
                      if item['inputs']['name']==config['pricing_start_scenario'])
    start_basis={
        'benefit':start_inputs['benefit_per_28days'],
        'max_claims':start_inputs['lifetime_payment_limit'],
        'waiting_days':start_inputs['waiting_days'],
        'decision_horizon_units':start_inputs['decision_horizon_units'],
    }
    expected_start={key:basis[key] for key in start_basis}
    if start_basis!=expected_start:
        raise ValueError(f'시작률 상품 조건 불일치: {start_basis} != {expected_start}')


def build_analysis(config=None):
    cfg=config or json.loads((ROOT/'config/treatment_scenarios.json').read_text())
    cashflow_config=json.loads((ROOT/'config/cashflows.json').read_text())
    base_treatment_config=json.loads((ROOT/'config/treatment.json').read_text())
    settings=CashflowSettings(**cashflow_config)
    _validate_default_product_basis(cfg,settings)
    start_rows=read_starts(
        ROOT/'outputs/start_rate/start_probabilities.csv',cfg['pricing_start_scenario'],
    )
    analysis=analyze_treatment_scenarios(
        settings=settings,start_rows=start_rows,mortality=read_mortality(),
        base_treatment_config=base_treatment_config,
        scenario_definitions=cfg['scenarios'],
        target_profit_margins=cfg['target_profit_margins'],
        product_id=cfg['product_basis']['product_id'],
        pricing_start_scenario=cfg['pricing_start_scenario'],
        start_probability_path=cfg['start_probability_path'],
    )
    reference=_read_csv(ROOT/'data/reference/treatment_sensitivity_team_results.csv')
    comparisons=compare_team_reference(analysis,reference)
    validation=validate_analysis(
        analysis,base_treatment_config,cfg['scenarios'],comparisons,
        cfg['product_basis']['reference_rounded_premiums_by_target'],
    )
    analysis.update(reference_comparisons=comparisons,validation=validation)
    return analysis


def main():
    analysis=build_analysis()
    out=ROOT/'outputs/treatment_sensitivity';out.mkdir(parents=True,exist_ok=True)
    outputs=[
        ('pricing_basis',analysis['pricing_rows']),
        ('treatment_sensitivity_39_results',analysis['results']),
        ('treatment_sensitivity_summary',analysis['summary_rows']),
        ('team_reproduction',analysis['reference_comparisons']),
    ]
    for name,rows in outputs:
        _write_csv(out/f'{name}.csv',rows)
        (out/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'validation.json').write_text(json.dumps(analysis['validation'],ensure_ascii=False,indent=2))
    print(json.dumps({
        'pricing_basis':analysis['pricing_rows'],
        'treatment_sensitivity_summary':analysis['summary_rows'],
        'validation':analysis['validation'],
    },ensure_ascii=False,indent=2))
    if analysis['validation']['status']!='PASS':
        raise AssertionError('치료 지속·재개 민감도 검증 실패')


if __name__=='__main__':main()
