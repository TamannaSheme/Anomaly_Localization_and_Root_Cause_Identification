"""Meaningful checks on data generation, leakage, alignment and ranking metrics."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"analysis"))
from synthetic_system import CaseConfig,simulate_case,simulate_components
from baselines import zscores,magnitude_ranking,onset_time_ranking
from run_smd import ranking_metrics


class PipelineChecks(unittest.TestCase):
    def test_intervention_has_exact_delayed_support_and_gain(self):
        cfg=CaseConfig(delays=(1,3,5),amplification=(.5,2.,1.5))
        b,r=simulate_components(cfg)
        for j,(arrival,height) in enumerate([(100,2.5),(101,1.25),(104,2.5),(109,3.75)]):
            np.testing.assert_array_equal(r[:arrival,j],0)
            np.testing.assert_allclose(r[arrival:,j],height)
        _,zero=simulate_components(replace(cfg,fault_size=0))
        np.testing.assert_array_equal(zero,0)

    def test_reproducible_and_paired_backgrounds(self):
        a=CaseConfig(); b=replace(a,delays=(10,10,10),amplification=(2.,2.,2.))
        np.testing.assert_array_equal(simulate_case(a).values,simulate_case(a).values)
        np.testing.assert_array_equal(simulate_components(a)[0],simulate_components(b)[0])
        self.assertFalse(np.array_equal(simulate_components(a)[0],simulate_components(replace(a,seed=1))[0]))

    def test_prefault_reference_cannot_see_fault_window(self):
        df=simulate_case(CaseConfig()); changed=df.copy(); changed.iloc[100:]*=1e6
        np.testing.assert_array_equal(zscores(df,100).iloc[:100],zscores(changed,100).iloc[:100])
        expected=(df-df.iloc[:100].mean())/df.iloc[:100].std(ddof=1)
        np.testing.assert_allclose(zscores(df,100),expected)

    def test_onset_sustain_rejects_single_spike(self):
        columns=["pump","flow","pressure","temperature"]
        normal=np.tile([-1.,1.],10)[:,None]*np.ones((1,4))
        data=np.vstack([normal,np.zeros((10,4))]); data[20,3]=100; data[22:25,1]=10
        df=pd.DataFrame(data,columns=columns)
        self.assertEqual(magnitude_ranking(df,20),"temperature")
        self.assertEqual(onset_time_ranking(df,20),"flow")

    def test_multilabel_ranking_metrics(self):
        m=ranking_metrics([2,0,1,3],{0,3})
        self.assertEqual(m['hit1'],0); self.assertEqual(m['hit3'],1)
        self.assertAlmostEqual(m['recall3'],.5)
        self.assertAlmostEqual(m['ndcg3'],(1/np.log2(3))/(1+1/np.log2(3)))
        perfect=ranking_metrics([0,3,1,2],{0,3})
        self.assertAlmostEqual(perfect['ndcg3'],1)

    def test_saved_sweep_matches_actual_data(self):
        out=ROOT/"analysis/results"
        rows=pd.read_csv(out/"sweep_cases.csv")
        data=np.load(out/"sweep_data.npz")['signals']
        self.assertEqual(data.shape,(320,200,4))
        self.assertEqual(len(rows.query("noise_profile == 'equal_noise'")),160)
        self.assertTrue(np.isfinite(data).all())
        for i,row in rows.iterrows():
            df=pd.DataFrame(data[i],columns=["pump","flow","pressure","temperature"])
            self.assertEqual(magnitude_ranking(df,100),row.magnitude_pred)
            self.assertEqual(onset_time_ranking(df,100),row.onset_pred)

    def test_smd_download_hashes_and_annotation_alignment(self):
        data=ROOT/"analysis/data/smd"
        meta=json.loads((data/"sources.json").read_text())
        for f in meta['files']:
            p=ROOT/f['file']
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),f['sha256'])
        labels=np.loadtxt(data/"test_label.txt",delimiter=',')
        events=pd.read_csv(ROOT/"analysis/results/smd_events.csv")
        for row in events.itertuples():
            self.assertTrue(labels[row.start:row.end_exclusive].all())
            self.assertEqual(labels[row.start-1],0)

    def test_aerca_oracle_score_alignment_and_summary(self):
        out=ROOT/"analysis/results"
        cases=pd.read_csv(out/"aerca_cases.csv")
        self.assertEqual(len(cases),300)
        for seed in range(3):
            data=np.load(out/f"aerca_seed{seed}/oracle_scores.npz")
            self.assertEqual(data['scores'].shape,data['labels'].shape)
            self.assertEqual(data['scores'].shape,(100,498,4))
            self.assertTrue(np.isfinite(data['scores']).all())
            self.assertTrue(((data['labels'].sum(axis=2)>0).sum(axis=1)==1).all())


if __name__ == "__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PipelineChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful())
    (ROOT/"analysis/results/validation.json").write_text(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
