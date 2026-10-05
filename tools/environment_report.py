import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=Path("artifacts/environment_report.txt"))
    args=ap.parse_args()
    import torch
    packages={}
    for name in ["torch","torchvision","lerobot","transformers","numpy","huggingface_hub","safetensors","opencv-python-headless","datasets","pyarrow","av"]:
        try: packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError: packages[name]="not installed"
    report={"os":platform.platform(),"python":sys.version,"packages":packages,
            "cuda_available":torch.cuda.is_available(),"torch_cuda":torch.version.cuda,
            "lerobot_commit":"ff71cae1ae2d09fd035553c35da65888ed6c8304",
            "SO101_base_commit":"f76f8cdc43a4311ae73a78cf74b9c80ab7701ee9",
            "simulation":"skipped at operator request"}
    if torch.cuda.is_available():
        report["gpu"]=torch.cuda.get_device_name()
        report["gpu_bytes"]=torch.cuda.get_device_properties(0).total_memory
        report["nvidia_smi"]=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version,name,memory.total","--format=csv,noheader"],text=True).strip()
    path=Path("/sys/fs/cgroup/memory.max")
    if path.exists(): report["container_memory_limit"]=path.read_text().strip()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
