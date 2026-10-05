"""Two-worker, deadline-paced control with native LeRobot timestep-aligned chunk replacement.

Uses native predict_action_chunk (50 predictions), never synchronous select_action's ten-prefix queue.
Consumption budget is a runtime option; late prefix indices are skipped, not replayed.
The control thread owns all actuator access. Inference performs no serial or control I/O.
"""
from dataclasses import dataclass
import math
import threading
import time
import numpy as np
from .config import MODEL_SHA256
from .protocol import CONTRACT_VERSION
from .action_adapter import filter_action,SafetyRejection


@dataclass(frozen=True)
class ObservationPacket:
    observation_seq: int
    observation_timestamp: float
    capture_complete_timestamp: float
    observation: dict


@dataclass(frozen=True)
class Candidate:
    observation_seq: int
    observation_timestamp: float
    inference_seq: int
    checkpoint_hash: str
    contract_version: str
    created_timestamp: float
    inference_start: float
    inference_end: float
    actions: np.ndarray
    trace: dict


class AlignedQueue:
    """Native latest-wins timestep semantics; wall control ticks advance even during SAFE HOLD."""
    def __init__(self, actions_per_chunk=10):
        if not 1<=actions_per_chunk<=50: raise ValueError('Invalid consumption horizon')
        self.actions_per_chunk=actions_per_chunk
        self.candidate=None; self.latest_seq=-1; self.latest_observation=-1
        self.selected=0; self.last_tick=-1; self.first_selection=True

    def accept(self,candidate,tick):
        if candidate.checkpoint_hash!=MODEL_SHA256 or candidate.contract_version!=CONTRACT_VERSION:
            raise ValueError('Candidate checkpoint/contract mismatch')
        if candidate.inference_seq<=self.latest_seq or candidate.observation_seq<=self.latest_observation:
            raise ValueError('Sequence regression')
        if candidate.actions.shape!=(50,6) or not np.isfinite(candidate.actions).all():
            raise ValueError('Expected finite native 50-step prediction')
        self.latest_seq=candidate.inference_seq; self.latest_observation=candidate.observation_seq
        index=tick-candidate.observation_seq
        if not 0<=index<50: return False
        self.candidate=candidate; self.selected=0; self.first_selection=True
        return True

    def pop(self,tick):
        if tick<=self.last_tick: raise ValueError('Control timestep regression/replay')
        self.last_tick=tick
        c=self.candidate
        if c is None: return None
        i=tick-c.observation_seq
        if not 0<=i<50 or self.selected>=self.actions_per_chunk:
            self.candidate=None; return None
        first=self.first_selection; self.first_selection=False; self.selected+=1
        return c.actions[i].copy(),c,i,first

    def clear(self): self.candidate=None


class AsyncRuntime:
    def __init__(self,infer,observe,bounds,*,motor_write_enabled=False,send_target=None,
                 shadow_safety=False,hold_timeout_s=.5,clock=time.perf_counter,sleep=time.sleep,
                 percentile_slew_warning=False,tick_sink=None,actions_per_chunk=10,chunk_size_threshold=.6):
        if motor_write_enabled and (send_target is None or shadow_safety):
            raise ValueError('Motor mode requires a real guarded sink and strict safety')
        self.infer=infer; self.observe=observe; self.bounds=bounds
        self.motor_write_enabled=motor_write_enabled; self.send_target=send_target
        self.shadow_safety=shadow_safety; self.hold_timeout_s=hold_timeout_s
        self.clock=clock; self.sleep=sleep
        self.percentile_slew_warning=percentile_slew_warning; self.tick_sink=tick_sink
        if not 1<=actions_per_chunk<=50 or not 0<chunk_size_threshold<=1: raise ValueError('Invalid async queue options')
        self.actions_per_chunk=actions_per_chunk; self.chunk_size_threshold=chunk_size_threshold
        self.inflight=False; self.request_ready=True
        self.stop_event=threading.Event(); self.condition=threading.Condition()
        self.latest=None; self.pending=None; self.worker_error=None; self.requests=[]
        self.worker=threading.Thread(target=self._inference_worker,name='so101-inference',daemon=True)

    def _inference_worker(self):
        last=-1; seq=0
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(lambda:self.stop_event.is_set() or (self.request_ready and self.pending is None and self.latest is not None and self.latest.observation_seq>last))
                if self.stop_event.is_set(): return
                packet=self.latest
                self.inflight=True; self.request_ready=False
            last=packet.observation_seq; start=self.clock()
            try:
                actions,trace=self.infer(packet.observation)
                end=self.clock(); immutable=np.asarray(actions,dtype=np.float32).copy(); immutable.setflags(write=False)
                c=Candidate(packet.observation_seq,packet.observation_timestamp,seq,MODEL_SHA256,CONTRACT_VERSION,
                            end,start,end,immutable,trace)
                row={'observation_seq':last,'observation_timestamp':packet.observation_timestamp,
                     'capture_complete_timestamp':packet.capture_complete_timestamp,'inference_seq':seq,
                     'inference_start':start,'inference_end':end,'created_timestamp':end,
                     'action_chunk_origin_timestamp':packet.observation_timestamp,
                     'latency_ms':(end-start)*1000,'checkpoint_hash':MODEL_SHA256,'contract_version':CONTRACT_VERSION}
                with self.condition:
                    self.inflight=False
                    if not self.stop_event.is_set(): self.pending=c
                self.requests.append(row); seq+=1
            except Exception as exc:
                with self.condition: self.worker_error=f'{type(exc).__name__}: {exc}'
                return

    def run(self,ticks,*,action_budget=None):
        """Control worker runs on calling thread; no HTTP, CUDA, encoding, or inference joins inside it."""
        epoch=self.clock(); queue=AlignedQueue(self.actions_per_chunk); previous=None; last_send=None
        rows=[]; rejection_reports=[]; first_executions=[]; selected_keys=set(); execution_records={}
        underflow=0; initial_underflow=0; arrivals=0; sent=0; accepted=0; watchdog=0; reason=None
        last_fresh=epoch; tick=0; observed_overlap=0
        control_ident=threading.get_ident()
        self.worker.start()
        def save_row(row):
            rows.append(row)
            if self.tick_sink is not None: self.tick_sink(row)
        try:
            for iteration in range(ticks):
                deadline=epoch+tick/30
                self.sleep(max(0,deadline-self.clock()))
                now=self.clock()
                # Never catch up by issuing multiple back-to-back overdue commands.
                tick=max(tick,int(math.floor((now-epoch)*30)))
                deadline=epoch+tick/30; begin=self.clock()
                obs=self.observe(tick)
                if obs is None:
                    # Failed bounded serial read: no cached-state publication and no motor write.
                    save_row({'iteration':iteration,'tick':tick,'control_timestamp':begin,
                              'jitter_ms':(begin-deadline)*1000,'mode':'STATE_READ_FAILED_HOLD',
                              'motor_write_enabled':self.motor_write_enabled,'underflow':False,
                              'state_valid':False,'goal_target':None if previous is None else previous.tolist(),
                              'control_work_ms':(self.clock()-begin)*1000})
                    tick+=1
                    continue
                state=np.asarray(obs['observation.state'],dtype=np.float32)
                capture_done=self.clock()
                packet=ObservationPacket(tick,capture_done,capture_done,obs)
                with self.condition:
                    self.latest=packet
                    pending=self.pending; self.pending=None; error=self.worker_error
                if error: reason=error; watchdog+=1; break
                if pending is not None and queue.accept(pending,tick):
                    arrivals+=1; last_fresh=self.clock()
                value=queue.pop(tick)
                remaining=0 if queue.candidate is None else max(0,min(self.actions_per_chunk-queue.selected,49-(tick-queue.candidate.observation_seq)))
                with self.condition:
                    self.request_ready=remaining<=self.actions_per_chunk*self.chunk_size_threshold
                    self.condition.notify()
                row={'iteration':iteration,'tick':tick,'virtual_time_s':tick/30,'control_timestamp':begin,
                     'jitter_ms':(begin-deadline)*1000,'mode':'SAFE_HOLD','motor_write_enabled':self.motor_write_enabled,
                     'robot_state':state.tolist(),'goal_target':None if previous is None else previous.tolist(),
                     'tracking_error':None if previous is None else (state-previous).tolist(),
                     'underflow':value is None,'observation_age_ms':(self.clock()-packet.observation_timestamp)*1000,
                     'queue_depth':remaining}
                if value is None:
                    underflow+=1
                    if not arrivals: initial_underflow+=1
                    if self.hold_timeout_s is not None and self.clock()-last_fresh>self.hold_timeout_s:
                        reason='SAFE_HOLD_WATCHDOG'; watchdog+=1; save_row(row); break
                else:
                    target,c,index,first=value
                    key=(c.inference_seq,index)
                    if key in selected_keys: raise RuntimeError('Stale action replay')
                    selected_keys.add(key)
                    age=self.clock()-c.observation_timestamp
                    if age>=50/30:
                        queue.clear(); row['mode']='EXPIRED_CHUNK_HOLD'; save_row(row); tick+=1; continue
                    if age<0: raise RuntimeError('Negative prediction age')
                    row.update(mode='CANDIDATE',observation_seq=c.observation_seq,inference_seq=c.inference_seq,
                               selected_action_index=index,prediction_age_ms=age*1000,
                               action_chunk_origin_timestamp=c.observation_timestamp,raw_target=target.tolist(),
                               inference_latency_ms=(c.inference_end-c.inference_start)*1000)
                    if first:
                        record={**row,'selected_start_index':index,'first_selection_timestamp':self.clock(),
                                'first_execution_timestamp':None,'first_execution_kind':None}
                        first_executions.append(record); execution_records[c.inference_seq]=record
                    trace=c.trace
                    context={'observation_state':state.tolist(),'chunk_anchor':trace.get('observation_state'),
                        'normalized_model_output':trace.get('normalized_model_output',[None]*50)[index],
                        'denormalized_model_output':trace.get('denormalized_model_output',[None]*50)[index]}
                    try:
                        dt=1/30 if last_send is None else self.clock()-last_send
                        reference=state if previous is None else previous
                        row.update(command_delta=(target-reference).tolist(),
                                   command_velocity=((target-reference)/dt).tolist(),
                                   tracking_error=None if previous is None else (state-previous).tolist(),
                                   transition_type='STARTUP_TRANSITION' if previous is None else 'CONTINUOUS_COMMAND')
                        try:
                            filtered,_=filter_action(target,state if previous is None else previous,self.bounds,
                                dt_s=dt,context=context,measured_state=state,startup=previous is None)
                        except SafetyRejection as slew:
                            if not self.percentile_slew_warning or any(v['category'] not in ('DELTA_LIMIT','VELOCITY_LIMIT') for v in slew.report['violations']): raise
                            row['slew_warning']=slew.report
                            filtered=target.copy()
                        if self.motor_write_enabled: self.send_target(filtered); sent+=1
                        previous=filtered; last_send=self.clock(); accepted+=1; row['mode']='ACCEPTED'; row['goal_target']=filtered.tolist()
                        record=execution_records[c.inference_seq]
                        if record['first_execution_timestamp'] is None:
                            record.update(first_execution_timestamp=last_send,execution_start_index=index,
                                          first_execution_kind='motor' if self.motor_write_enabled else 'virtual_sink')
                    except SafetyRejection as exc:
                        rejection_reports.append({**exc.report,'tick':tick,'inference_seq':c.inference_seq,'index':index})
                        row['mode']='SAFETY_REJECTED'
                        if not self.shadow_safety: reason='SAFETY_REJECTION'; save_row(row); break
                row['control_work_ms']=(self.clock()-begin)*1000
                save_row(row)
                tick+=1
                if action_budget is not None and accepted>=action_budget: break
        finally:
            queue.clear(); self.stop_event.set()
            with self.condition: self.condition.notify_all()
            self.worker.join(timeout=0)  # Never delay actuator-safe cleanup waiting for a remote response.
        for row in rows:
            if any(r['inference_start']<=row['control_timestamp']<r['inference_end'] for r in self.requests): observed_overlap+=1
        def percentiles(values):
            return dict(zip(['P50','P95','P99'],np.percentile(values,[50,95,99]).tolist())) if values else None
        latency=percentiles([r['latency_ms'] for r in self.requests])
        ages=percentiles([r['prediction_age_ms'] for r in rows if r['mode']=='ACCEPTED'])
        summary={'control_ticks':len(rows),'control_hz':30,'native_prediction_horizon':50,'maximum_actions_selected_per_chunk':self.actions_per_chunk,'chunk_size_threshold':self.chunk_size_threshold,
                 'queue_underflow':underflow,'initial_queue_underflow':initial_underflow,'steady_queue_underflow':underflow-initial_underflow,
                 'fresh_chunk_arrivals':arrivals,'selected_actions':len(selected_keys),'safety_accepted':accepted,'safety_rejected':len(rejection_reports),
                 'prediction_age_ms':ages,'inference_latency_ms':latency,'control_jitter_ms':percentiles([r['jitter_ms'] for r in rows]),
                 'selected_prediction_age_ms':percentiles([r['prediction_age_ms'] for r in rows if 'prediction_age_ms' in r]),
                 'control_work_ms':percentiles([r['control_work_ms'] for r in rows if 'control_work_ms' in r]),
                 'watchdog_activations':watchdog,'stop_reason':reason,'motor_commands_sent':sent,
                 'stale_replays':0,'sequence_regressions':0,'control_ticks_overlapping_inference':observed_overlap,
                 'worker_join_completed':not self.worker.is_alive(),'prediction_horizon_ms':50/30*1000,
                 'control_thread_ident':control_ident,'inference_thread_ident':self.worker.ident,
                 'shadow_safety':self.shadow_safety}
        return {'summary':summary,'ticks':rows,'requests':self.requests,'first_executions':first_executions,'rejections':rejection_reports}
