import json
import unittest
from datetime import date,timedelta

from models.rider.cashflows import CashflowSettings
from models.rider.start_integration import ROOT
from models.rider.treatment_sensitivity import (
    analyze_treatment_scenarios,build_analysis,build_parameter_cases,
)


class TreatmentSensitivityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis=build_analysis()
        cls.config=json.loads((ROOT/'config/treatment_scenarios.json').read_text())
        cls.base_treatment=json.loads((ROOT/'config/treatment.json').read_text())

    def test_39_fixed_premium_results_and_validation(self):
        self.assertEqual(len(self.analysis['summary_rows']),13)
        self.assertEqual(len(self.analysis['results']),39)
        self.assertEqual(
            [p['monthly_gross_premium_rounded_up_krw'] for p in self.analysis['prices']],
            [2808,2965,3141],
        )
        self.assertEqual(self.analysis['validation']['status'],'PASS')
        for scenario in self.analysis['summary_rows']:
            rows=[r for r in self.analysis['results']
                  if r['treatment_scenario_id']==scenario['treatment_scenario_id']]
            self.assertEqual(len({r['total_paid_claims'] for r in rows}),1)

    def test_team_reported_results_reproduced(self):
        by_id={row['treatment_scenario_id']:row for row in self.analysis['summary_rows']}
        expected={
            'treatment_base':(2265.413157979289,88800.88512969017,14718684.050201476,31119062.63015455),
            'treatment_stop_reduction_30':(2403.68978104941,-14271913.435543478,219729.68873256445,16465138.541424274),
            'treatment_restart_50':(2419.2992566081634,-15732300.182334512,-1250479.1225417256,14983919.008181393),
            'treatment_combined_stress':(2436.46785366697,-17639259.34828168,-3180507.6027347445,13028029.385903955),
        }
        for scenario_id,values in expected.items():
            row=by_id[scenario_id]
            self.assertAlmostEqual(row['total_paid_claims'],values[0],places=9)
            self.assertAlmostEqual(row['target_0pct_profit_pv_krw'],values[1],places=5)
            self.assertAlmostEqual(row['target_5pct_profit_pv_krw'],values[2],places=5)
            self.assertAlmostEqual(row['target_10pct_profit_pv_krw'],values[3],places=5)
        self.assertAlmostEqual(
            by_id['treatment_combined_stress']['target_10pct_scenario_profit_margin'],
            .04244631343578966,places=12,
        )

    def test_scenario_settings_start_from_baseline(self):
        cases=build_parameter_cases(self.base_treatment,self.config['scenarios'])
        by_id={case['scenario_id']:case for case in cases}
        stop=by_id['treatment_stop_reduction_30']['resolved_config']
        restart=by_id['treatment_restart_50']['resolved_config']
        self.assertEqual(stop['stop_reduction'],.3)
        self.assertEqual(stop['restart_in_window_fraction'],.25)
        self.assertEqual(restart['stop_reduction'],0.)
        self.assertEqual(restart['restart_in_window_fraction'],.5)
        self.assertEqual(restart['postrestart_stop_hazard_multiplier'],1.)

    def test_start_rate_and_claim_limit_checks(self):
        starts={row['total_starts'] for row in self.analysis['summary_rows']}
        self.assertEqual(len(starts),1)
        for row in self.analysis['summary_rows']:
            self.assertLess(row['max_mass_error'],1e-7)
            self.assertLessEqual(row['total_paid_claims'],13*row['total_starts']+1e-7)

    def test_core_accepts_separate_product_context_without_baseline_prices(self):
        start=date(2026,11,15)
        rows=[{'date':(start+timedelta(days=i)).isoformat(),'elapsed_day':i,
               'p_start_yes':.01,'p_start_no':.01} for i in range(200)]
        settings=CashflowSettings(initial_policies=10,benefit=150000,max_claims=6,
                                  waiting_days=0,death_enabled=False)
        scenarios=[self.config['scenarios'][0],self.config['scenarios'][11]]
        analysis=analyze_treatment_scenarios(
            settings=settings,start_rows=rows,mortality={},
            base_treatment_config=self.base_treatment,
            scenario_definitions=scenarios,target_profit_margins=[0.,.05],
            product_id='test_benefit150k_max6_wait0',
        )
        self.assertEqual(len(analysis['results']),4)
        self.assertTrue(all(r['product_id']=='test_benefit150k_max6_wait0'
                            for r in analysis['results']))
        self.assertNotEqual(
            analysis['prices'][0]['monthly_gross_premium_rounded_up_krw'],2808,
        )


if __name__=='__main__':unittest.main()
