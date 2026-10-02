"""Unit tests for reusable QC metrics; synthetic data only, no hardware."""
import unittest
import numpy as np
from qc_signal import parse_task,finite_values,timestamp_check,lag_estimate,length_outliers
from qc_video import fingerprints,duplicate_stats
from qc_dataset import episode_summary


class MetricsTests(unittest.TestCase):
    def test_duplicates(self):
        rgb=np.full((16,16,3),80,dtype=np.uint8)
        self.assertEqual(fingerprints(rgb),fingerprints(rgb.copy()))
        other=rgb.copy(); other[0,0]=0
        self.assertNotEqual(fingerprints(rgb)[0],fingerprints(other)[0])
        stats=duplicate_stats([0,0,0,20],[False,True,True,False],[False,True,True,False],[0,0,0,10],[0,1,2,3])
        self.assertAlmostEqual(stats['exact_duplicate_ratio'],2/3)
        self.assertEqual(stats['longest_exact_duplicate_run_frames'],3)

    def test_nan_action(self):
        result=finite_values([[1,2],[np.nan,np.inf]])
        self.assertFalse(result['ok']); self.assertEqual(result['nonfinite_count'],2)
        self.assertTrue(finite_values(np.zeros((3,6)))['ok'])

    def test_timestamps(self):
        self.assertTrue(timestamp_check(np.arange(10)/30,30)['grid_ok'])
        self.assertFalse(timestamp_check([0,.1,.1],10)['strictly_monotonic'])
        self.assertFalse(timestamp_check([0,np.nan],30)['finite'])

    def test_lag(self):
        rng=np.random.default_rng(0); action=rng.normal(size=500)
        state=np.r_[np.zeros(4),action[:-4]]
        result=lag_estimate(action,state,30)
        self.assertEqual(result['best_lag_frames'],4)
        self.assertAlmostEqual(result['RMSE'],0)
        self.assertAlmostEqual(result['corr'],1)
        constant=lag_estimate(np.zeros(20),np.zeros(20),30)
        self.assertFalse(constant['identifiable']); self.assertIsNone(constant['corr'])

    def test_tasks(self):
        self.assertEqual(parse_task('Stack the red block on top of the blue block.'),('red','blue'))
        self.assertIsNone(parse_task('Stack the red block on top of the red block.'))
        self.assertIsNone(parse_task('Stack the purple block on top of the red block.'))
        self.assertIsNone(parse_task('unrelated instruction'))

    def test_episode_summary(self):
        ep=dict(episode=0,task='task',frames=90,duration_s=3,fps=30,state_dim=6,action_dim=6,
                auto_status='PASS',recommendation='KEEP',warnings=[],errors=[],joints={},
                cameras={name:dict(decode_ok=True,frame_count_matches=True,decoded_frames=90) for name in ('front','wrist')})
        row=episode_summary(ep)
        self.assertEqual(row['duration_s'],3); self.assertTrue(row['decode_ok']); self.assertEqual(row['front_frames'],90)
        ep['cameras'].pop('wrist'); self.assertFalse(episode_summary(ep)['decode_ok'])

    def test_length_MAD(self):
        self.assertEqual(length_outliers([10,11,12,13,14,100])['episodes'][-1]['flag'],'too_long')
        self.assertTrue(all(r['flag'] is None for r in length_outliers([10]*6)['episodes']))


if __name__=='__main__':
    unittest.main(verbosity=2)
