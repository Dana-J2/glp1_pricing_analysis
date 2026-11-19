"""지급액·지급한도 9개 설계의 가격과 분리 스트레스를 비교한다."""
import argparse
import csv
import hashlib
import json
import math

from models.rider.cashflows import CashflowSettings,project_cashflows,summarize
from models.rider.pricing import calculate_premiums
from models.rider.profit_analysis import evaluate_projected_cashflows
from models.rider.start_integration import ROOT,read_mortality
from models.rider.treatment_sensitivity import (
    analyze_treatment_scenarios,build_parameter_cases,
)
from models.start_rate.model import Scenario,decision_support,run as run_start_rate
from models.treatment.model import Parameters


def _margin_label(margin):
    return f'{100*margin:g}'.replace('.','p')


def _write_csv(path,rows):
    if not rows:
        raise ValueError(f'{path.name}에 저장할 결과가 없습니다')
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _write_rows(out,name,rows):
    _write_csv(out/f'{name}.csv',rows)
    (out/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))


def _check(checks,name,condition,detail=''):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})


def _design_id(benefit,max_claims,waiting_days):
    return f'benefit{int(benefit/1000)}k_max{max_claims}_wait{waiting_days}'


def build_product_design(*,benefit,max_claims,waiting_days,decision_horizon_units,
                         base_cashflow_config,product_id=None):
    """하나의 상품조건을 기존 현금흐름 설정과 연결한다."""
    settings_config=dict(base_cashflow_config)
    settings_config.update(
        benefit=float(benefit),max_claims=int(max_claims),
        waiting_days=int(waiting_days),
    )
    design={
        'design_id':product_id or _design_id(benefit,max_claims,waiting_days),
        'benefit':float(benefit),
        'max_claims':int(max_claims),
        'waiting_days':int(waiting_days),
        'decision_horizon_units':int(decision_horizon_units),
        'maximum_support_krw':float(benefit)*int(max_claims),
        'settings_config':settings_config,
        'settings':CashflowSettings(**settings_config),
    }
    if design['maximum_support_krw']<=0:
        raise ValueError('최대 지원금은 양수여야 합니다')
    return design


def build_designs(config,base_cashflow_config):
    designs=[]
    for benefit in config['benefit_amounts_krw']:
        for max_claims in config['max_claims_values']:
            designs.append(build_product_design(
                benefit=benefit,max_claims=max_claims,
                waiting_days=config['waiting_days'],
                decision_horizon_units=config['decision_horizon_units'],
                base_cashflow_config=base_cashflow_config,
            ))
    if len(designs)!=9 or len({d['design_id'] for d in designs})!=9:
        raise ValueError('지급액 3개×지급한도 3개의 고유 설계가 필요합니다')
    return designs


def load_start_templates():
    config=json.loads((ROOT/'config/start_scenarios.json').read_text())
    return {item['inputs']['name']:item for item in config['scenarios']}


def generate_start_rows(template,design):
    args=dict(template['inputs'])
    args.update(
        benefit_per_28days=design['benefit'],
        lifetime_payment_limit=design['max_claims'],
        waiting_days=design['waiting_days'],
        decision_horizon_units=design['decision_horizon_units'],
    )
    args['selection_ratios']=tuple(args['selection_ratios'])
    scenario=Scenario(**args).validate()
    start_summary,years,daily=run_start_rate(scenario,keep_daily=True)
    rows=[{
        'scenario':scenario.name,
        'elapsed_day':int(row['elapsed_days']),
        'date':row['date'],
        'p_start_no':float(row['conditional_daily_q_without']),
        'p_start_yes':float(row['conditional_daily_q_with']),
    } for row in daily]
    digest=hashlib.sha256('\n'.join(
        f"{row['elapsed_day']},{row['p_start_yes']:.17g}" for row in rows
    ).encode()).hexdigest()
    condition_match=(
        scenario.benefit_per_28days==design['benefit']
        and scenario.lifetime_payment_limit==design['max_claims']
        and scenario.waiting_days==design['waiting_days']
        and scenario.decision_horizon_units==design['decision_horizon_units']
    )
    return {
        'scenario':scenario,'rows':rows,'summary':start_summary,'years':years,
        'path_sha256':digest,'condition_match':condition_match,
        'inception_support':decision_support(scenario,0),
    }


def _path_metrics(projected,summary,design):
    n=design['settings'].initial_policies
    recipients=summary['recipients']
    one_year_starts=math.fsum(row['new_starters'] for row in projected[:365])
    return {
        'expected_benefit_nominal_per_initial_policy_krw':
            design['benefit']*summary['total_paid_claims']/n,
        'expected_benefit_pv_per_initial_policy_krw':summary['pv_benefit_krw']/n,
        'expected_paid_claims_per_initial_policy':summary['total_paid_claims']/n,
        'first_year_starter_ratio_initial':one_year_starts/n,
        'ten_year_starter_ratio_initial':summary['total_starts']/n,
        'recipient_ratio_initial':recipients/n,
        'recipient_ratio_among_starters':recipients/summary['total_starts'],
        'exhaustion_ratio_initial':summary['exhausted']/n,
        'exhaustion_ratio_among_recipients':summary['exhausted']/recipients,
        'total_starts':summary['total_starts'],
        'recipients':recipients,
        'total_paid_claims':summary['total_paid_claims'],
        'exhausted':summary['exhausted'],
    }


def build_pricing_stage(config=None):
    config=config or json.loads((ROOT/'config/product_comparison.json').read_text())
    base_cashflow=json.loads((ROOT/'config/cashflows.json').read_text())
    base_treatment=json.loads((ROOT/'config/treatment.json').read_text())
    designs=build_designs(config,base_cashflow)
    templates=load_start_templates();base_template=templates[config['pricing_start_scenario']]
    mortality=read_mortality();treatment=Parameters(**base_treatment)
    caches={};pricing_rows=[];start_summaries=[]
    for design in designs:
        start=generate_start_rows(base_template,design)
        projected=project_cashflows(start['rows'],mortality,design['settings'],treatment)
        summary=summarize(projected,design['settings'].initial_policies)
        metrics=_path_metrics(projected,summary,design)
        prices=[calculate_premiums(summary,m) for m in config['target_profit_margins']]
        caches[design['design_id']]={
            'start':start,'projected':projected,'summary':summary,
            'metrics':metrics,'prices':prices,
        }
        start_summaries.append({
            'design_id':design['design_id'],'start_scenario':'base',
            'benefit_per_dispense_krw':design['benefit'],'max_claims':design['max_claims'],
            'waiting_days':design['waiting_days'],
            'decision_horizon_units':design['decision_horizon_units'],
            'planned_payable_units_at_inception':start['inception_support']['planned_payable_units'],
            'effective_support_share_at_inception':start['inception_support']['effective_support_share'],
            'first_year_start_percent_start_model':start['summary']['year1_with_percent'],
            'ten_year_starts_start_model_N10000':start['summary']['year10_cumulative_with'],
            'start_probability_path_sha256':start['path_sha256'],
            'product_condition_match':start['condition_match'],
        })
        for price in prices:
            pricing_rows.append({
                'design_id':design['design_id'],
                'benefit_per_dispense_krw':design['benefit'],
                'max_claims':design['max_claims'],
                'waiting_days':design['waiting_days'],
                'decision_horizon_units':design['decision_horizon_units'],
                'maximum_support_krw':design['maximum_support_krw'],
                'target_profit_margin':price['target_profit_margin'],
                'monthly_net_premium_krw':price['monthly_net_premium_krw'],
                'monthly_gross_premium_unrounded_krw':price['monthly_gross_premium_krw'],
                'monthly_gross_premium_rounded_up_krw':price['monthly_gross_premium_rounded_up_krw'],
                'unrounded_achieved_profit_margin':price['achieved_profit_margin'],
                'rounded_achieved_profit_margin':price['rounded_achieved_profit_margin'],
                **metrics,
                'starter_denominator':'initial policies',
                'recipient_denominator':'initial policies; unique policies with >=1 paid claim',
                'exhaustion_denominator':'initial policies',
                'start_probability_path_sha256':start['path_sha256'],
                'product_condition_match':start['condition_match'],
            })
        print(f"pricing stage: {design['design_id']}",flush=True)

    baseline=config['baseline_design_id']
    baseline_by_margin={r['target_profit_margin']:r for r in pricing_rows if r['design_id']==baseline}
    for row in pricing_rows:
        reference=baseline_by_margin[row['target_profit_margin']]
        row['premium_difference_vs_baseline_same_target_krw']=(
            row['monthly_gross_premium_unrounded_krw']-reference['monthly_gross_premium_unrounded_krw']
        )
        row['premium_change_vs_baseline_same_target_percent']=(
            100*row['premium_difference_vs_baseline_same_target_krw']
            / reference['monthly_gross_premium_unrounded_krw']
        )

    checks=[]
    regression=config['baseline_regression']
    base5=baseline_by_margin[.05];base10=baseline_by_margin[.10]
    _check(checks,'기준 상품 5%·10% 확정 보험료',
           base5['monthly_gross_premium_rounded_up_krw']==regression['target_5pct_rounded_premium_krw']
           and base10['monthly_gross_premium_rounded_up_krw']==regression['target_10pct_rounded_premium_krw'])
    target_error=max(abs(r['unrounded_achieved_profit_margin']-r['target_profit_margin'])
                     for r in pricing_rows)
    _check(checks,'올림 전 목표이익률 달성',target_error<1e-12,
           f'최대 오차 {target_error:.3g}')
    _check(checks,'9개 설계의 시작률·급부 상품조건 일치',
           all(r['product_condition_match'] for r in pricing_rows))
    _check(checks,'수급자·시작자·소진자·지급한도 관계',all(
        r['exhausted']<=r['recipients']<=r['total_starts']<=1000+1e-7
        and r['total_paid_claims']<=r['recipients']*r['max_claims']+1e-7
        for r in pricing_rows
    ))
    _check(checks,'9개 설계×2개 목표이익률',len(pricing_rows)==18)
    validation={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks}
    return {
        'config':config,'designs':designs,'templates':templates,
        'base_cashflow_config':base_cashflow,'base_treatment_config':base_treatment,
        'mortality':mortality,'caches':caches,'pricing_rows':pricing_rows,
        'start_summaries':start_summaries,'validation':validation,
    }


def _stress_row(*,design,price,analysis_group,scenario_id,scenario_name,
                projected,summary,evaluation,scenario_family):
    return {
        'design_id':design['design_id'],
        'benefit_per_dispense_krw':design['benefit'],
        'max_claims':design['max_claims'],
        'waiting_days':design['waiting_days'],
        'decision_horizon_units':design['decision_horizon_units'],
        'target_profit_margin':price['target_profit_margin'],
        'fixed_monthly_gross_premium_krw':price['monthly_gross_premium_rounded_up_krw'],
        'analysis_group':analysis_group,
        'experience_scenario_id':scenario_id,
        'experience_scenario_display_name':scenario_name,
        'experience_scenario_family':scenario_family,
        'is_analysis_group_baseline':scenario_id in ('base','treatment_base'),
        'baseline_duplicated_across_analysis_groups':scenario_id in ('base','treatment_base'),
        'premium_income_pv_krw':evaluation['premium_income_pv_krw'],
        'benefit_pv_krw':evaluation['benefit_pv_krw'],
        'expenses_pv_krw':evaluation['expenses_pv_krw'],
        'refund_pv_krw':evaluation['refund_pv_krw'],
        'total_outflow_pv_krw':evaluation['total_outflow_pv_krw'],
        'profit_pv_krw':evaluation['profit_pv_krw'],
        'scenario_profit_margin':evaluation['profit_margin'],
        'profit_change_vs_same_group_baseline_krw':0.,
        'total_starts':summary['total_starts'],
        'recipients':summary['recipients'],
        'total_paid_claims':summary['total_paid_claims'],
        'exhausted':summary['exhausted'],
        'pv_premium_units':summary['pv_premium_units'],
        'pv_net_premium_units':summary['pv_net_premium_units'],
        'pv_fixed_outflow_krw':summary['pv_fixed_outflow_krw'],
        'max_mass_error':summary['max_mass_error'],
        'cohort_basis':'initial 1000 policies',
    }


def build_full_analysis(pricing_stage=None):
    stage=pricing_stage or build_pricing_stage()
    config=stage['config'];treatment=Parameters(**stage['base_treatment_config'])
    start_results=[];treatment_results=[];generated_start_summaries=[]
    role_by_name={name:item['role_ko'] for name,item in stage['templates'].items()}

    for design in stage['designs']:
        prices=stage['caches'][design['design_id']]['prices']
        for scenario_name in config['start_rate_scenarios']:
            if scenario_name=='base':
                bundle=stage['caches'][design['design_id']]['start']
                projected=stage['caches'][design['design_id']]['projected']
                summary=stage['caches'][design['design_id']]['summary']
            else:
                bundle=generate_start_rows(stage['templates'][scenario_name],design)
                projected=project_cashflows(
                    bundle['rows'],stage['mortality'],design['settings'],treatment,
                )
                summary=summarize(projected,design['settings'].initial_policies)
            generated_start_summaries.append({
                'design_id':design['design_id'],'start_scenario':scenario_name,
                'role_ko':role_by_name[scenario_name],
                'benefit_per_dispense_krw':design['benefit'],'max_claims':design['max_claims'],
                'waiting_days':design['waiting_days'],
                'decision_horizon_units':design['decision_horizon_units'],
                'start_probability_path_sha256':bundle['path_sha256'],
                'total_starts_contract_model':summary['total_starts'],
                'product_condition_match':bundle['condition_match'],
            })
            for price in prices:
                g=price['monthly_gross_premium_rounded_up_krw']
                evaluation=evaluate_projected_cashflows(projected,design['settings'],g)
                start_results.append(_stress_row(
                    design=design,price=price,analysis_group='start_rate',
                    scenario_id=scenario_name,scenario_name=role_by_name[scenario_name],
                    projected=projected,summary=summary,evaluation=evaluation,
                    scenario_family='start_rate',
                ))

        treatment_analysis=analyze_treatment_scenarios(
            settings=design['settings'],
            start_rows=stage['caches'][design['design_id']]['start']['rows'],
            mortality=stage['mortality'],
            base_treatment_config=stage['base_treatment_config'],
            scenario_definitions=json.loads((ROOT/config['treatment_scenario_config']).read_text())['scenarios'],
            target_profit_margins=config['target_profit_margins'],
            product_id=design['design_id'],pricing_start_scenario='base',
            start_probability_path=config['start_probability_path'],
        )
        treatment_case={c['scenario_id']:c for c in treatment_analysis['cases']}
        for result in treatment_analysis['results']:
            scenario_id=result['treatment_scenario_id']
            price=next(p for p in prices if p['target_profit_margin']==result['pricing_target_profit_margin'])
            summary=treatment_analysis['summaries'][scenario_id]
            projected=treatment_analysis['paths'][scenario_id]
            evaluation=evaluate_projected_cashflows(
                projected,design['settings'],price['monthly_gross_premium_rounded_up_krw'],
            )
            treatment_results.append(_stress_row(
                design=design,price=price,analysis_group='treatment_behavior',
                scenario_id=scenario_id,
                scenario_name=treatment_case[scenario_id]['display_name_ko'],
                projected=projected,summary=summary,evaluation=evaluation,
                scenario_family=treatment_case[scenario_id]['family'],
            ))
        print(f"stress stage: {design['design_id']}",flush=True)

    all_results=start_results+treatment_results
    for group in ('start_rate','treatment_behavior'):
        group_baseline='base' if group=='start_rate' else 'treatment_base'
        for design in stage['designs']:
            for margin in config['target_profit_margins']:
                rows=[r for r in all_results if r['analysis_group']==group
                      and r['design_id']==design['design_id']
                      and r['target_profit_margin']==margin]
                baseline=next(r for r in rows if r['experience_scenario_id']==group_baseline)
                for row in rows:
                    row['profit_change_vs_same_group_baseline_krw']=row['profit_pv_krw']-baseline['profit_pv_krw']

    decisions=[]
    pricing_lookup={(r['design_id'],r['target_profit_margin']):r for r in stage['pricing_rows']}
    for design in stage['designs']:
        for margin in config['target_profit_margins']:
            price=pricing_lookup[(design['design_id'],margin)]
            start_rows=[r for r in start_results if r['design_id']==design['design_id']
                        and r['target_profit_margin']==margin]
            treatment_rows=[r for r in treatment_results if r['design_id']==design['design_id']
                            and r['target_profit_margin']==margin]
            response=next(r for r in start_rows if r['experience_scenario_id']=='response_high')
            start_combined=next(r for r in start_rows if r['experience_scenario_id']=='combined_stress')
            treatment_combined=next(r for r in treatment_rows if r['experience_scenario_id']=='treatment_combined_stress')
            worst_start=min(start_rows,key=lambda r:r['profit_pv_krw'])
            worst_treatment=min(treatment_rows,key=lambda r:r['profit_pv_krw'])
            decisions.append({
                'design_id':design['design_id'],'target_profit_margin':margin,
                'monthly_gross_premium_rounded_up_krw':price['monthly_gross_premium_rounded_up_krw'],
                'maximum_support_krw':design['maximum_support_krw'],
                'recipient_ratio_initial':price['recipient_ratio_initial'],
                'expected_benefit_nominal_per_initial_policy_krw':price['expected_benefit_nominal_per_initial_policy_krw'],
                'expected_benefit_pv_per_initial_policy_krw':price['expected_benefit_pv_per_initial_policy_krw'],
                'response_high_profit_pv_krw':response['profit_pv_krw'],
                'response_high_nonnegative_reference':response['profit_pv_krw']>=0,
                'start_combined_stress_profit_pv_krw':start_combined['profit_pv_krw'],
                'treatment_combined_stress_profit_pv_krw':treatment_combined['profit_pv_krw'],
                'worst_start_rate_scenario_id':worst_start['experience_scenario_id'],
                'worst_start_rate_profit_pv_krw':worst_start['profit_pv_krw'],
                'worst_treatment_scenario_id':worst_treatment['experience_scenario_id'],
                'worst_treatment_profit_pv_krw':worst_treatment['profit_pv_krw'],
                'interpretation_note':'reference indicators only; no automatic optimal-product decision',
            })

    validation=validate_full(stage,start_results,treatment_results,generated_start_summaries)
    return {
        **stage,'start_results':start_results,'treatment_results':treatment_results,
        'stress_results':all_results,'decision_summary':decisions,
        'generated_start_summaries':generated_start_summaries,
        'full_validation':validation,
    }


def validate_full(stage,start_results,treatment_results,start_summaries):
    config=stage['config'];checks=[];all_results=start_results+treatment_results
    baseline=config['baseline_design_id'];reg=config['baseline_regression']
    def find(rows,margin,scenario):
        return next(r for r in rows if r['design_id']==baseline
                    and r['target_profit_margin']==margin
                    and r['experience_scenario_id']==scenario)
    _check(checks,'기준 상품 기존 시작률 수지 재현',all(
        abs(value-reference)<.01 for value,reference in [
            (find(start_results,.10,'base')['profit_pv_krw'],reg['target_10pct_start_base_profit_pv_krw']),
            (find(start_results,.10,'response_high')['profit_pv_krw'],reg['target_10pct_start_response_high_profit_pv_krw']),
            (find(start_results,.10,'combined_stress')['profit_pv_krw'],reg['target_10pct_start_combined_stress_profit_pv_krw']),
        ]
    ))
    _check(checks,'기준 상품 기존 치료 행태 수지 재현',
           abs(find(treatment_results,.10,'treatment_combined_stress')['profit_pv_krw']
               -reg['target_10pct_treatment_combined_stress_profit_pv_krw'])<.01)
    _check(checks,'시작률·급부 모형 상품조건 일치',
           all(row['product_condition_match'] for row in start_summaries))
    fixed_ok=all(len({r['fixed_monthly_gross_premium_krw'] for r in all_results
                      if r['design_id']==design['design_id']
                      and r['target_profit_margin']==margin})==1
                 for design in stage['designs'] for margin in config['target_profit_margins'])
    _check(checks,'설계별 확정 보험료의 경험 시나리오 고정',fixed_ok)
    target_error=max(abs(r['unrounded_achieved_profit_margin']-r['target_profit_margin'])
                     for r in stage['pricing_rows'])
    _check(checks,'올림 전 기준 목표이익률 달성',target_error<1e-12,
           f'최대 오차 {target_error:.3g}')
    path_invariant=all(len({r['total_paid_claims'] for r in all_results
                            if r['design_id']==row['design_id']
                            and r['analysis_group']==row['analysis_group']
                            and r['experience_scenario_id']==row['experience_scenario_id']})==1
                       for row in all_results)
    _check(checks,'목표이익률 변경 시 치료 경로·지급 건수 불변',path_invariant)
    hierarchy=all(r['exhausted']<=r['recipients']<=r['total_starts']<=1000+1e-7
                  and r['total_paid_claims']<=r['recipients']*r['max_claims']+1e-7
                  for r in all_results)
    _check(checks,'수급자·시작자·소진자·지급한도 관계',hierarchy)
    formula_error=max(abs(r['profit_pv_krw']-(
        r['fixed_monthly_gross_premium_krw']*r['pv_net_premium_units']-r['pv_fixed_outflow_krw']))
        for r in all_results)
    _check(checks,'직접 합산 손익과 G×D−C 일치',formula_error<.01,
           f'최대 차이 {formula_error:.9f}원')
    _check(checks,'시나리오 설정 독립·비누적',
           all(row['product_condition_match'] for row in start_summaries))
    _check(checks,'요청한 스트레스 행 수',len(start_results)==108 and len(treatment_results)==234,
           f'시작률 {len(start_results)}행, 치료 행태 {len(treatment_results)}행')
    return {
        'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL',
        'checks':checks,
        'limitations':'research assumptions; no premium behavior response, tax, capital cost or separate investment income; start-rate and treatment stresses applied separately',
    }


def save_pricing_stage(stage,out):
    out.mkdir(parents=True,exist_ok=True)
    _write_rows(out,'design_pricing_benefit_comparison',stage['pricing_rows'])
    _write_rows(out,'design_base_start_summary',stage['start_summaries'])
    (out/'pricing_stage_validation.json').write_text(json.dumps(stage['validation'],ensure_ascii=False,indent=2))
    (out/'run_config_snapshot.json').write_text(json.dumps(stage['config'],ensure_ascii=False,indent=2))


def save_full_analysis(analysis,out):
    save_pricing_stage(analysis,out)
    _write_rows(out,'start_rate_stress_108_results',analysis['start_results'])
    _write_rows(out,'treatment_stress_234_results',analysis['treatment_results'])
    _write_rows(out,'stress_detail_342_results',analysis['stress_results'])
    _write_rows(out,'decision_summary',analysis['decision_summary'])
    _write_rows(out,'design_start_rate_generation_summary',analysis['generated_start_summaries'])
    (out/'validation.json').write_text(json.dumps(analysis['full_validation'],ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description='상품조건 9개 가격·스트레스 비교')
    parser.add_argument('--stage',choices=('pricing','full'),default='full')
    args=parser.parse_args();out=ROOT/'outputs/product_comparison'
    pricing=build_pricing_stage();save_pricing_stage(pricing,out)
    if pricing['validation']['status']!='PASS':
        raise AssertionError('상품조건 가격·혜택 단계 검증 실패')
    if args.stage=='pricing':
        print(json.dumps({'pricing_rows':pricing['pricing_rows'],
                          'validation':pricing['validation']},ensure_ascii=False,indent=2))
        return
    analysis=build_full_analysis(pricing);save_full_analysis(analysis,out)
    print(json.dumps({'decision_summary':analysis['decision_summary'],
                      'validation':analysis['full_validation']},ensure_ascii=False,indent=2))
    if analysis['full_validation']['status']!='PASS':
        raise AssertionError('상품조건 전체 스트레스 검증 실패')


if __name__=='__main__':main()
