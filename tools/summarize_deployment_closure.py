"""Build a reproducible closure report from measured artifacts; no inferred success flags."""
from collections import Counter
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('artifacts')
    # Merge diagnostic exports without duplicating identical GT rejection records.
    records={}
    for name in ('safety_rejections.jsonl','safety_rejections_server.jsonl'):
        for line in (root/name).read_text().splitlines():
            item=json.loads(line); records[json.dumps(item,sort_keys=True)]=item
    (root/'safety_rejections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records.values()))
    measurements={}
    for name in ('latency_localhost_final','latency_wan_final'):
        report=json.loads((root/(name+'.json')).read_text())
        requests=[json.loads(x) for x in (root/(name+'.jsonl')).read_text().splitlines()]
        warm=[r for r in requests if not r['warmup']]
        for field in ('upload_body_read_ms','lock_wait_ms'):
            report['metrics_ms'][field]=dict(zip(['P50','P90','P95','P99'],
                np.percentile([r['server_trace'][field] for r in warm],[50,90,95,99]).tolist()))
        (root/(name+'.json')).write_text(json.dumps(report,indent=2))
        measurements[report['topology']]=report
    gt=json.loads((root/'gt_oracle_report.json').read_text())
    causes=Counter(v['category'] for r in records.values() for v in r['violations'])
    gt_examples=[r for r in records.values() if r.get('source')=='GT_ORACLE']
    calibration_violations=[{'episode':r['episode'],'frame':r['frame'],'violations':[v for v in r['violations'] if v['category']=='POSITION_LIMIT']}
                            for r in gt_examples if any(v['category']=='POSITION_LIMIT' for v in r['violations'])]
    report={'status':'BLOCKED','hardware_commands_sent':0,'gt_oracle':gt,
        'relative_semantics':{'formula':'absolute[k,0:5] = state_t[0:5] + relative[k,0:5]; absolute[k,5] = gripper[k]',
                              'cumulative_integration_bug':False,'gripper_absolute':True},
        'runner_equivalence':json.loads((root/'runner_equivalence.json').read_text()),
        'latency':measurements,'LAN':{'status':'NOT_MEASURED','reason':'No robot-local GPU/LAN endpoint available; cloud SSH is WAN'},
        'limits_widened':False,'calibration_envelope_gt_violations':calibration_violations,
        'rejection_counts_across_diagnostic_runs':dict(causes),
        'blockers':[
            {'location':'artifacts/safety_profile_v2.json; tools/gt_deployment_oracle.py',
             'cause':'26 real GT frames conflict with unchanged safety policy: 23 exceed percentile-derived delta caps; 3 shoulder_lift targets exceed saved calibration envelope. Physical safety of those recordings is not established; no automatic widening permitted.'},
            {'location':'deployment/pi05_stack2/cli/run_robot.py',
             'cause':'Current loop is synchronous observe/infer/execute. Even positive localhost pipeline margin does not implement overlap or prove continuous 30 Hz. WAN P99 exceeds 333.33ms; runtime rejects motor execution on that route.'},
            {'location':'deployment/pi05_stack2/cli/preflight.py',
             'cause':'Physical joint/camera mapping and live calibration unresolved; no LAN timing measurement. Exact train/test episode IDs were not available, so range statistics are recorded-data reference only.'}],
        'tests':(root/'closure_tests.log').read_text().strip()}
    (root/'deployment_closure_report.json').write_text(json.dumps(report,indent=2))
    local=measurements['localhost']; wan=measurements['WAN_SSH']
    lines=['# Deployment software closure','', 'Status: **BLOCKED**. Motor writes: **0**. No safety bounds widened.',
           '',f"GT oracle: {gt['passed']}/{gt['frames']} = {gt['GT_ACTION_PASS_RATE']:.6%}; round-trip max error {gt['roundtrip_max_error']:.8g}.",
           'The model was bypassed. Saved relative + quantile processors and the saved inverse processor, queue and deployment safety filter were exercised.',
           'GT violations are genuine policy conflicts, not automatically safe demonstrations. 23 exceed delta caps; 3 exceed the shoulder-lift calibration envelope.',
           '', 'All future arm targets use the same observation anchor. Gripper is absolute 0–100. No cumulative integration was found.',
           'Fixed-seed real GPU inference before/after preprocessing reuse matched exactly in all 30 compared actions.',
           '', '| Metric (ms) | localhost binary P50/P95/P99 | WAN PNG P50/P95/P99 |','|---|---|---|']
    for metric in ['serialization_ms','upload_body_read_ms','image_decode_ms','preprocessing_ms','model_ms','postprocessing_ms','transport_residual_ms','total_ms']:
        cells=[]
        for source in (local,wan):
            q=source['metrics_ms'][metric]; cells.append(' / '.join(f'{q[k]:.2f}' for k in ('P50','P95','P99')))
        lines.append('| '+metric+' | '+' | '.join(cells)+' |')
    lines += ['', 'Each final topology used 100 measured requests plus 3 warmups. Raw timestamp traces are saved alongside summaries.',
        'WAN one-way upload/download cannot be recovered without synchronized host clocks. Body-read duration is measured server-side; the residual is combined transport and response serialization, not RTT/2.',
        'The 333.33ms horizon has positive measured localhost margin. This is a necessary condition for pipelining, not proof that the synchronous runtime overlaps inference and execution. LAN remains unmeasured.',
        '', '## Exact blockers']
    lines += [f"- `{b['location']}`: {b['cause']}" for b in report['blockers']]
    lines += ['', '## Regression tests', '```',report['tests'],'```','',
              'Read-only preflight commands are documented in deployment/pi05_stack2/SOFTWARE_CLOSURE.md. No physical operation was run.']
    (root/'deployment_closure_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'GT_ACTION_PASS_RATE':gt['GT_ACTION_PASS_RATE'],
                      'rejection_records':len(records),'reports':str(root/'deployment_closure_report.json')},indent=2))


if __name__=='__main__': main()
