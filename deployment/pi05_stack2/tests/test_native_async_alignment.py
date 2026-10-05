"""Execute the actual pinned native timing/merge functions without constructing a robot client."""
import ast
from pathlib import Path
from queue import Queue
from threading import Lock
from types import SimpleNamespace
from typing import Callable
import numpy as np
import pytest


def test_against_pinned_native_timing_and_latest_wins():
    lerobot=pytest.importorskip('lerobot')
    import torch
    from dataclasses import dataclass
    from deployment.pi05_stack2.async_runtime import AlignedQueue,Candidate
    from deployment.pi05_stack2.config import MODEL_SHA256
    from deployment.pi05_stack2.protocol import CONTRACT_VERSION
    root=Path(lerobot.__file__).parent/'async_inference'
    helpers=ast.parse((root/'helpers.py').read_text())
    classes=[n for n in helpers.body if isinstance(n,ast.ClassDef) and n.name in ('TimedData','TimedAction')]
    env={'dataclass':dataclass,'PolicyAction':torch.Tensor}
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(root/'helpers.py'),'exec'),env)
    TimedAction=env['TimedAction']
    def native(filename,name):
        tree=ast.parse((root/filename).read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        env={'torch':torch,'TimedAction':TimedAction,'Queue':Queue,'Callable':Callable}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(root/filename),'exec'),env)
        return env[name]
    stamp=native('policy_server.py','_time_action_chunk')
    merge=native('robot_client.py','_aggregate_action_queues')
    a=np.repeat(np.arange(50,dtype=np.float32)[:,None],6,axis=1)
    incoming=stamp(SimpleNamespace(config=SimpleNamespace(environment_dt=1/30)),10.,list(torch.tensor(a)),6)
    assert incoming[7].get_timestamp()==pytest.approx(10+7/30)
    assert incoming[7].get_timestep()==13
    client=SimpleNamespace(action_queue=Queue(),action_queue_lock=Lock(),latest_action_lock=Lock(),latest_action=7)
    merge(client,incoming)
    native_first=client.action_queue.get_nowait()
    q=AlignedQueue(); q.accept(Candidate(6,10.,0,MODEL_SHA256,CONTRACT_VERSION,10.2,10.,10.2,a,{}),8)
    target,c,index,first=q.pop(8)
    assert index==2 and native_first.get_timestep()==8
    np.testing.assert_array_equal(target,native_first.get_action().numpy())
