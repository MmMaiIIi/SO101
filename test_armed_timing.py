"""Hardware-free onset, camera-thread and aligned real LeRobot sidecar regression tests."""
import copy
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from collector import EpisodeWriter, JOINTS, RecordingState, TaskScheduler, make_dataset_factory
from quality_telemetry import camera_snapshot, timing_summary, write_episode_timing
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.feature_utils import hw_to_dataset_features


def features():
    positions={f'{j}.pos':float for j in JOINTS}
    return {**hw_to_dataset_features(positions,'action'),
            **hw_to_dataset_features({**positions,'front':(48,64,3),'wrist':(48,64,3)},'observation')}


class InlineWriter:
    """Deterministic in-memory dataset producer; no hardware or background thread."""
    def __init__(self):
        self.features=features(); self.error=None; self.frames=[]; self.results=queue.SimpleQueue()
    def frame(self,frame):
        self.frames.append(copy.deepcopy(frame)); return True
    def barrier(self,kind):
        self.results.put(('saved',1) if kind=='save' else ('discarded',None))


def feed(state, index, delta=0., repeated=False, image=None, gripper_delta=0.):
    ns=10_000_000_000+index*33_333_333
    action={f'{j}.pos':delta if j=='shoulder_pan' else 0. for j in JOINTS}
    action['gripper.pos']=gripper_delta
    image=np.full((48,64,3),index%255,np.uint8) if image is None else image
    obs={**action,'front':image,'wrist':image}
    seq=index-1 if repeated else index
    update=ns-50_000_000 if repeated else ns-5_000_000
    cameras={name:SimpleNamespace(latest_timestamp=time.perf_counter(),last_read_frame_metadata=(seq,update)) for name in ('front','wrist')}
    timing={'tick_start_monotonic_ns':ns-4_000_000,'tick_monotonic_ns':ns,
            'state_timestamp_ns':ns-3_000_000,'state_read_done_ns':ns-3_000_000,
            'action_read_done_ns':ns-2_000_000,'action_send_timestamp_ns':ns-1_000_000,
            'action_send_done_ns':ns-1_000_000,'control_dt_ms':1000/30}
    state.record(obs,action,cameras,ns/1e9,1/30,timing)


class ArmedTimingTests(unittest.TestCase):
    def make_state(self):
        writer=InlineWriter()
        state=RecordingState(writer,TaskScheduler(3),{f'{j}.pos':100. for j in JOINTS},.005,12)
        feed(state,0)
        state.key(' ',time.monotonic())
        return state,writer

    def test_camera_thread_new_and_repeated_updates(self):
        camera=OpenCVCamera(OpenCVCameraConfig(index_or_path=0,width=64,height=48,fps=30))
        camera.stop_event=threading.Event()
        camera.thread=SimpleNamespace(is_alive=lambda:True)
        image=np.zeros((48,64,3),np.uint8)
        def capture():
            camera.stop_event.set()
            return image
        camera._read_from_hardware=capture
        camera._postprocess_image=lambda value:value
        read=OpenCVCamera.read_latest.__wrapped__
        with patch('lerobot.cameras.opencv.camera_opencv.time.monotonic_ns',return_value=1_000_000_000):
            camera._read_loop()
        read(camera)
        first=camera_snapshot({'front':camera},1_010_000_000)
        read(camera)
        repeated=camera_snapshot({'front':camera},1_030_000_000)
        self.assertEqual(first['front_frame_seq'],1)
        self.assertEqual(repeated['front_frame_seq'],1)
        self.assertEqual(first['front_update_monotonic_ns'],repeated['front_update_monotonic_ns'])
        self.assertEqual(repeated['front_age_ms'],30.)
        camera.stop_event.clear()
        with patch('lerobot.cameras.opencv.camera_opencv.time.monotonic_ns',return_value=1_040_000_000):
            camera._read_loop()
        read(camera)
        new=camera_snapshot({'front':camera},1_045_000_000)
        self.assertEqual(new['front_frame_seq'],2)
        self.assertEqual(new['front_age_ms'],5.)

    def test_noise_and_single_tick_do_not_trigger(self):
        state,writer=self.make_state()
        for i in range(1,60):
            feed(state,i,.05 if i%2 else -.05,gripper_delta=.1 if i%2 else -.1)
        feed(state,60,1.)
        feed(state,61,0.)
        self.assertTrue(state.armed); self.assertFalse(state.recording)
        self.assertEqual(writer.frames,[])

    def test_two_ticks_trigger_and_keep_pre_roll(self):
        state,writer=self.make_state()
        for i in range(1,61):
            feed(state,i)
        feed(state,61,1.)
        self.assertFalse(state.recording)
        feed(state,62,1.2)
        self.assertTrue(state.recording); self.assertFalse(state.armed)
        self.assertEqual(len(writer.frames),12)
        self.assertEqual([f['_collector_qc']['frame_index'] for f in writer.frames],list(range(12)))
        self.assertEqual([float(f['action'][0]) for f in writer.frames][-2:],[1.,np.float32(1.2)])
        # 10 static context frames + both confirmation movement frames are retained.
        self.assertTrue(all(float(f['action'][0])==0 for f in writer.frames[:-2]))
        self.assertAlmostEqual(state.started,writer.frames[0]['_collector_qc']['tick_monotonic_ns']/1e9)
        self.assertAlmostEqual((writer.frames[-1]['_collector_qc']['tick_monotonic_ns']-writer.frames[0]['_collector_qc']['tick_monotonic_ns'])/1e9,11/30,places=6)

    def test_cancel_armed_no_episode_and_same_task(self):
        state,writer=self.make_state(); task=state.task
        for i in range(1,20):
            feed(state,i)
        state.key('\x7f',time.monotonic())
        self.assertFalse(state.active); self.assertIsNone(state.task)
        self.assertEqual(writer.frames,[]); self.assertEqual(state.saved,0)
        state.key(' ',time.monotonic()); self.assertEqual(state.task,task)

    def test_timeout_stays_armed(self):
        state,writer=self.make_state()
        feed(state,400,0.)  # >10 s host timeline; no fallback recording
        state.key('\r',time.monotonic())
        self.assertTrue(state.armed); self.assertEqual(writer.frames,[])
        self.assertIn('Waiting for first movement',state.notice)

    def test_repeated_seq_is_not_new_and_high_age_is_only_telemetry(self):
        state,writer=self.make_state()
        feed(state,1,1.); feed(state,2,1.)
        feed(state,3,1.,repeated=True)
        last=writer.frames[-1]['_collector_qc']
        self.assertFalse(last['front_is_new']); self.assertFalse(last['wrist_is_new'])
        self.assertEqual(last['front_age_ms'],50.)
        cameras={name:SimpleNamespace(latest_timestamp=time.perf_counter(),last_read_frame_metadata=(2,1)) for name in ('front','wrist')}
        obs={**{f'{j}.pos':1. for j in JOINTS},'front':np.zeros((48,64,3),np.uint8),'wrist':np.zeros((48,64,3),np.uint8)}
        state.record(obs,{f'{j}.pos':1. for j in JOINTS},cameras,time.monotonic(),1/30)
        self.assertTrue(state.recording)  # sidecar age itself never triggers discard

    def test_pre_roll_detaches_images(self):
        state,writer=self.make_state()
        image=np.zeros((48,64,3),np.uint8)
        feed(state,1,image=image)
        image[:]=255
        feed(state,2,1.); feed(state,3,1.)
        self.assertEqual(int(writer.frames[0]['observation.images.front'].max()),0)

    def test_real_dataset_sidecar_alignment_and_reset_control(self):
        with tempfile.TemporaryDirectory(prefix='so101_armed_timing_') as directory:
            root=Path(directory)/'dataset'
            writer=EpisodeWriter(make_dataset_factory(root,'local/armed_timing_test',30,features()))
            state=RecordingState(writer,TaskScheduler(3),{f'{j}.pos':100. for j in JOINTS},.005,12)
            try:
                feed(state,0); state.key(' ',time.monotonic())
                for i in range(1,61):
                    feed(state,i)
                feed(state,61,1.); feed(state,62,1.)
                for i in range(63,68):
                    feed(state,i,1.5)
                    time.sleep(.005)
                state.key('\r',time.monotonic())
                deadline=time.monotonic()+30
                reset_ticks=0
                while state.busy and time.monotonic()<deadline:
                    feed(state,68+reset_ticks,0.)  # unconditional teleop continues in RESET
                    reset_ticks+=1; state.poll(); time.sleep(.005)
                self.assertFalse(state.busy); self.assertEqual(state.saved,1)
                self.assertGreater(reset_ticks,0); self.assertFalse(state.active)
                writer.close()
                self.assertIsNone(writer.error)
                info=json.loads((root/'meta/info.json').read_text())
                self.assertEqual(info['total_frames'],17)
                data=pq.read_table(next((root/'data').rglob('*.parquet'))).to_pandas()
                sidecar=pq.read_table(root/'telemetry/episode_000.parquet').to_pandas()
                self.assertEqual(len(data),len(sidecar))
                self.assertEqual(list(data.frame_index),list(sidecar.frame_index))
                self.assertEqual(list(sidecar.episode_index),list(data.episode_index))
                self.assertTrue(np.allclose(data.timestamp,np.arange(17)/30))
                self.assertTrue(np.allclose(data.action.iloc[10],np.array([1.,0,0,0,0,0])))
                self.assertIn('camera timing:',state.notice)
                import sys
                sys.path.insert(0,str(Path(__file__).parent/'tools'))
                from qc_dataset import host_camera_telemetry, run_qc
                timing,errors=host_camera_telemetry(root)
                self.assertFalse(errors); self.assertEqual(timing[0]['sample_count'],17)
                self.assertFalse(timing[0]['errors'])
                report=run_qc(root,Path(directory)/'qc',width=64,height=48,make_review=False)
                self.assertFalse(report['global_errors'])
                self.assertFalse(report['episodes'][0]['errors'])
                self.assertTrue(report['raw_dataset_unchanged'])
                self.assertEqual(report['episodes'][0]['host_camera_telemetry']['sample_count'],17)
                self.assertGreaterEqual(report['episodes'][0]['boundary']['start_idle_s'],.3)
                self.assertLessEqual(report['episodes'][0]['boundary']['start_idle_s'],.5)
            finally:
                if writer.thread.is_alive():
                    writer.close()


if __name__=='__main__':
    unittest.main(verbosity=2)
