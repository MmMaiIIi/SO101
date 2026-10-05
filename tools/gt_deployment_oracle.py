"""Real-data oracle through saved action processors, absolute queue, and unchanged safety caps."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
import lerobot.policies.pi05.processor_pi05  # registers saved steps
from lerobot.processor.factory import load_pretrained_policy_processors
from lerobot.lerobot_types import TransitionKey as K
from deployment.pi05_stack2.action_adapter import JointBounds, filter_action, SafetyRejection, append_rejection
from deployment.pi05_stack2.action_queue import SafeActionQueue


def saved_action_roundtrip(pre, post, state, targets):
    """Run the saved relative+normalization steps and entire paired postprocessor (no model)."""
    transition = {K.OBSERVATION:{"observation.state":torch.tensor(state).float()[None]},
                  K.ACTION:torch.tensor(targets).float()[None]}
    relative = pre.steps[2](transition)
    normalized = pre.steps[3](relative)[K.ACTION]
    restored = post(normalized).detach().cpu().numpy()[0]
    return restored, normalized.detach().cpu().numpy()[0], relative[K.ACTION].numpy()[0]


def stats(a):
    return dict(zip(["min","p0_1","p1","p50","p99","p99_9","max"],
                    np.percentile(a,[0,.1,1,50,99,99.9,100],axis=0).tolist()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('artifacts'))
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    profile=json.loads(args.profile.read_text())
    bounds=JointBounds(tuple(profile['lower']),tuple(profile['upper']),tuple(profile['max_delta_per_step']),
                       tuple(profile['max_velocity_per_s']))
    pre,post=load_pretrained_policy_processors(args.checkpoint,
                      preprocessor_overrides={'device_processor':{'device':'cpu'}})
    rows=[]
    for path in sorted((args.dataset_root/'data').rglob('*.parquet')):
        rows.extend(pq.read_table(path,columns=['episode_index','frame_index','observation.state','action']).to_pylist())
    rows.sort(key=lambda r:(r['episode_index'],r['frame_index']))
    s=np.array([r['observation.state'] for r in rows],dtype=np.float32)
    a=np.array([r['action'] for r in rows],dtype=np.float32)
    ep=np.array([r['episode_index'] for r in rows])
    same=ep[1:]==ep[:-1]; diffs=np.diff(a,axis=0)[same]
    counts=Counter(); joints=Counter(); passed=0; tested=0; errors=[]; startup=Counter()
    continuous=0; continuous_passed=0; command_failed=0; tracking=[]; discontinuities=[]
    path=args.output/'safety_rejections.jsonl'
    # Preserve prior runtime diagnostics; the summary tool deduplicates identical oracle records.
    path.touch(exist_ok=True)
    for episode in sorted(set(ep)):
        ids=np.flatnonzero(ep==episode)
        previous_command=None
        for offset in range(0,len(ids),10):
            chunk_ids=ids[offset:offset+10]; first=chunk_ids[0]
            decoded,normalized,relative=saved_action_roundtrip(pre,post,s[first],a[chunk_ids])
            errors.append(float(np.max(np.abs(decoded-a[chunk_ids]))))
            queue=SafeActionQueue(10); queue.push_chunk(offset,decoded)
            for k,i in enumerate(chunk_ids):
                target=queue.pop(offset); tested+=1
                # Continuous replay has a known previous *command*, not measured lagging state.
                is_startup=previous_command is None
                previous=s[i] if is_startup else previous_command
                if not is_startup:
                    continuous+=1; tracking.append(s[i]-previous)
                context={'source':'GT_ORACLE','episode':int(episode),'frame':rows[i]['frame_index'],
                         'observation_state':s[i].tolist(),'chunk_anchor_state':s[first].tolist(),
                         'normalized_model_output':normalized[k].tolist(),
                         'normalized_output_origin':'GT encoded by saved processor; model bypassed',
                         'denormalized_model_output':relative[k].tolist(),'relative_arm_action':relative[k,:5].tolist()}
                try:
                    filter_action(target,previous,bounds,context=context,measured_state=s[i],startup=is_startup)
                    passed+=1
                    if not is_startup: continuous_passed+=1
                except SafetyRejection as exc:
                    append_rejection(exc.report,path)
                    for v in exc.report['violations']:
                        counts[v['category']]+=1; joints[v.get('joint','unknown')]+=1
                    if is_startup:
                        for v in exc.report['violations']: startup[v['category']]+=1
                    elif any(v['category']=='DELTA_LIMIT' for v in exc.report['violations']):
                        command_failed+=1
                        discontinuities.append({'episode':int(episode),'frame':rows[i]['frame_index'],
                            'previous_command':previous.tolist(),'next_command':target.tolist(),
                            'measured_state':s[i].tolist(),'tracking_error':(s[i]-previous).tolist(),
                            'command_delta':(target-previous).tolist(),
                            'violations':exc.report['violations'],
                            'raw_gt_delta':(a[i]-a[i-1]).tolist()})
                # Replay what was actually recorded even after a rejected transition; a real runtime stops instead.
                previous_command=target.copy()
    unexplained=sum(not any(abs(d['raw_gt_delta'][v['dimension']])>profile['max_delta_per_step'][v['dimension']]
                           for v in d['violations'] if v['category']=='DELTA_LIMIT') for d in discontinuities)
    report={'frames':tested,'episodes':len(set(ep)),'passed':passed,'failed':tested-passed,
            'unexplained_software_violations':unexplained,
            'GT_ACTION_PASS_RATE':passed/tested,'roundtrip_max_error':max(errors),
            'representation_pass':max(errors)<3e-5,'failure_categories':dict(counts),'failure_joints':dict(joints),
            'startup_transition_violations':dict(startup),
            'continuous_transitions':continuous,
            'continuous_gt_command_transition_pass_rate':continuous_passed/continuous,
            'continuous_command_slew_pass_rate':(continuous-command_failed)/continuous,
            'remaining_original_delta_violations':command_failed,
            'ROOT_CAUSE':'RECORDED_COMMAND_SLEW_EXCEEDS_UNCHANGED_PERCENTILE_CAP' if command_failed else 'incorrect safety-oracle reference state',
            'tracking_error_definition':'measured_state[t] - previous_command[t-1]',
            'tracking_error_distribution':stats(np.asarray(tracking)),
            'tracking_error_absolute_distribution':stats(np.abs(tracking)),
            'tracking_error_limit':'UNRESOLVED_HARDWARE_EVIDENCE; diagnostic only, never substituted into command slew',
            'split':'recorded Stack2 data; exact training membership unavailable',
            'previous_target':'previous recorded command within episode; measured state only at episode start',
            'hardware_commands_sent':0,'limits_changed':False}
    (args.output/'gt_oracle_report.json').write_text(json.dumps(report,indent=2))
    (args.output/'gt_command_discontinuities.json').write_text(json.dumps(discontinuities,indent=2))
    envelope={'state':stats(s),'absolute_action_target':stats(a),'signed_per_step_action_delta':stats(diffs),
              'absolute_per_step_action_delta':stats(np.abs(diffs))}
    layered={**profile,'hardware_limit':{'status':'REQUIRES_HARDWARE_VERIFICATION','reason':'No mechanical workspace/limit artifact'},
             'calibration_limit':{'lower':profile['lower'],'upper':profile['upper'],
                                  'meaning':'saved calibration encoder endpoint envelope, NOT certified physical stops',
                                  'live_match':'REQUIRES_HARDWARE_VERIFICATION'},
             'data_operating_envelope':envelope,'runtime_delta_limit':profile['max_delta_per_step'],
             'runtime_velocity_limit':profile['max_velocity_per_s'],
             'software_gate':'PASS' if max(errors)<3e-5 and unexplained==0 else 'BLOCKED',
             'hardware_safety_status':'HARDWARE_SAFETY_UNVERIFIED',
             'absolute_position_limit':{'lower':profile['lower'],'upper':profile['upper']},
             'command_delta_limit':profile['max_delta_per_step'],
             'command_velocity_limit':profile['max_velocity_per_s'],
             'tracking_error_limit':None,
             'reason':'GT safety conflicts require resolution; no bounds automatically widened'}
    (args.output/'safety_profile_v2.json').write_text(json.dumps(layered,indent=2))
    info=json.loads((args.dataset_root/'meta/info.json').read_text())
    audit={'formula':'absolute[k,:5] = state_t[:5] + denormalized_relative[k,:5]; absolute[k,5] = denormalized[k,5]',
           'cumulative_integration':False,'saved_processor_gripper_exclusion':True,
           'data_split':report['split'],'dataset_features':info['features'],
           'source_evidence':['relative_action_processor.to_relative_actions/to_absolute_actions',
                              'motors_bus._normalize/_unnormalize DEGREES branch',
                              'collector use_degrees=True; saved action_feature_names'],
           'dimensions':[]}
    for j,name in enumerate(profile['joint_order']):
        audit['dimensions'].append({'index':j,'joint':name,'state_range':[float(s[:,j].min()),float(s[:,j].max())],
             'action_range':[float(a[:,j].min()),float(a[:,j].max())],
             'unit':'degrees' if j<5 else 'normalized 0-100',
             'deployment_representation':'absolute target; model predicts same-anchor relative arm' if j<5 else 'absolute gripper',
             'physical_mapping_sign_offset':'REQUIRES_HARDWARE_VERIFICATION',
             'calibration_convention':'(encoder-(range_min+range_max)/2)*360/4095' if j<5 else '(encoder-range_min)/(range_max-range_min)*100'})
    (args.output/'action_state_contract_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
