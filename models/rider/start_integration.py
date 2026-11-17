"""최초 시작확률을 살아 있는 미시작 계약에 연결한다. 급부·보험료는 다음 단계."""
from pathlib import Path
from datetime import date, timedelta
import calendar
import csv
import json
import math

ROOT=Path(__file__).resolve().parents[2]

def read_starts(path,scenario='base'):
    with open(path,encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['scenario']==scenario]
    if not rows:raise ValueError('시나리오 없음')
    rows.sort(key=lambda r:int(r['elapsed_day']))
    start=date.fromisoformat(rows[0]['date'])
    for i,r in enumerate(rows):
        if int(r['elapsed_day'])!=i or date.fromisoformat(r['date'])!=start+timedelta(days=i):
            raise ValueError('날짜 누락 또는 중복')
        for key in ('p_start_no','p_start_yes'):
            r[key]=float(r[key])
            if not math.isfinite(r[key]) or not 0<=r[key]<=1:raise ValueError('확률 범위')
    return rows

def read_mortality():
    with (ROOT/'data/reference/mortality.csv').open(encoding='utf-8-sig',newline='') as f:
        return {(int(r['age']),r['sex']):float(r['annual_q']) for r in csv.DictReader(f)}

def project_starts(rows,mortality,*,path='yes',initial_policies=1000.,entry_age=49,sex='여',
                   mortality_multiplier=1.,monthly_lapse=0.,mortality_enabled=True):
    if path not in ('yes','no'):raise ValueError('path')
    if not 0<=monthly_lapse<=1 or not math.isfinite(monthly_lapse):raise ValueError('해지율')
    if initial_policies<0 or not math.isfinite(initial_policies):raise ValueError('가입자 수')
    if mortality_multiplier<0 or not math.isfinite(mortality_multiplier):raise ValueError('사망력 배수')
    first=date.fromisoformat(rows[0]['date'])
    naive=float(initial_policies);started=0.;death_total=lapse_total=starts_total=0.;out=[]
    for r in rows:
        dt=date.fromisoformat(r['date'])
        year=dt.year-first.year-((dt.month,dt.day)<(first.month,first.day))
        age=entry_age+year
        # 보험나이는 계약 응당일에 증가. 당일 이탈은 납입 직전 해지만 처리한다.
        payday=dt.day==min(first.day,calendar.monthrange(dt.year,dt.month)[1])
        w=monthly_lapse if payday and dt>first else 0.
        opening_naive,opening_started=naive,started
        lapse_n,lapse_s=naive*w,started*w
        naive-=lapse_n;started-=lapse_s
        risk_set=naive
        new=naive*r['p_start_'+path]
        naive-=new;started+=new;starts_total+=new
        q=mortality[(age,sex)] if mortality_enabled else 0.
        if not math.isfinite(q) or not 0<=q<1:raise ValueError('기초 사망확률')
        daily_q=-math.expm1(math.log1p(-q)*mortality_multiplier/365.)
        # 시작·당일 조제 이후 하루 동안의 사망. 신규 시작자도 해당 구간에 포함한다.
        dn,ds=naive*daily_q,started*daily_q
        naive-=dn;started-=ds
        death_total+=dn+ds;lapse_total+=lapse_n+lapse_s
        out.append(dict(date=dt.isoformat(),elapsed_day=int(r['elapsed_day']),policy_year=year+1,
          insurance_age=age,path=path,naive_open=opening_naive,started_open=opening_started,
          lapse_naive=lapse_n,lapse_started=lapse_s,eligible_naive_before_start=risk_set,
          p_start=r['p_start_'+path],new_starters=new,annual_q=q,daily_death_q=daily_q,
          deaths_naive=dn,deaths_started=ds,naive_close=naive,started_alive_close=started,
          cumulative_starts=starts_total,cumulative_deaths=death_total,cumulative_lapses=lapse_total,
          mass_error=initial_policies-naive-started-death_total-lapse_total))
    return out

def save(path,rows):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    (ROOT/"outputs/rider").mkdir(parents=True,exist_ok=True)
    settings=json.loads((ROOT/'config/rider.json').read_text())
    rows=read_starts(ROOT/'outputs/start_rate/start_probabilities.csv')
    mortality=read_mortality();summaries=[]
    for enabled in (False,True):
        for path in ('no','yes'):
            data=project_starts(rows,mortality,path=path,mortality_enabled=enabled,**settings)
            tag=('mortality' if enabled else 'no_exit')+'_'+path
            save(ROOT/'outputs/rider'/f'{tag}.csv',data)
            summaries.append(dict(case=tag,first_365_days_starters=sum(r['new_starters'] for r in data[:365]),
              ten_year_starters=data[-1]['cumulative_starts'],alive_at_end=data[-1]['naive_close']+data[-1]['started_alive_close'],
              deaths=data[-1]['cumulative_deaths'],max_mass_error=max(abs(r['mass_error']) for r in data)))
    save(ROOT/'outputs/rider/start_summary.csv',summaries)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
