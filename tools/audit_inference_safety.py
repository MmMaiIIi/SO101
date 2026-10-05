"""Report policy/recorded targets against the frozen profile; never alter outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from deployment.pi05_stack2.action_adapter import JointBounds, filter_action


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inference",type=Path,default=Path("artifacts/cuda_inference"))
    ap.add_argument("--profile",type=Path,default=Path("artifacts/safety_profile.json"))
    ap.add_argument("--output",type=Path,default=Path("artifacts/action_safety_audit.json"))
    args=ap.parse_args()
    profile=json.loads(args.profile.read_text())
    bounds=JointBounds(tuple(profile["lower"]),tuple(profile["upper"]),tuple(profile["max_delta_per_step"]))
    rows=[]
    for path in sorted(args.inference.glob("*.npz")):
        data=np.load(path)
        row={"sample":path.name}
        for key in ("predicted","ground_truth"):
            previous=data["state"]
            failures=[]
            for index,target in enumerate(data[key]):
                try: filter_action(target,previous,bounds)
                except ValueError as exc: failures.append({"step":index,"reason":str(exc)})
                previous=target
            row[key+"_gate_pass"]=not failures
            row[key+"_failures"]=failures
        rows.append(row)
    report={"samples":len(rows),"predicted_gate_pass":sum(r["predicted_gate_pass"] for r in rows),
            "recorded_gate_pass":sum(r["ground_truth_gate_pass"] for r in rows),"results":rows,
            "interpretation":"Diagnostic only. Targets and safety limits are unchanged; unsafe targets remain rejected."}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print({key:report[key] for key in ("samples","predicted_gate_pass","recorded_gate_pass")})


if __name__=="__main__": main()
