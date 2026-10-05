"""Versioned LAN/SSH-tunnel transport. Absolute actions only, never motor writes."""
from __future__ import annotations

import base64
import io
import time
import uuid
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from http.client import HTTPConnection, HTTPSConnection
import json
import struct

import numpy as np
from PIL import Image

from .config import CAMERA_ORDER, MODEL_SHA256
from .observation_adapter import make_observation

CONTRACT_VERSION = "so101-pi05-stack2-v1"


def encode_binary_request(observation, sequence, session_id, instruction, action_mode='sync10'):
    """Exact uint8 RGB payload for local/LAN use; no lossy encoding or base64."""
    make_observation(observation['observation.state'],{k:observation[k] for k in CAMERA_ORDER},instruction)
    header={'contract_version':CONTRACT_VERSION,'checkpoint_sha256':MODEL_SHA256,
            'session_id':session_id,'observation_sequence':sequence,'instruction':instruction,
            'observation_timestamp_ns':time.monotonic_ns(),
            'state':np.asarray(observation['observation.state']).tolist(),'action_mode':action_mode}
    encoded=json.dumps(header,allow_nan=False).encode()
    return struct.pack('!I',len(encoded))+encoded+b''.join(observation[k].tobytes(order='C') for k in CAMERA_ORDER)


def decode_binary_request(payload):
    if len(payload)<4: raise ValueError('Truncated binary header')
    n=struct.unpack('!I',payload[:4])[0]; frame_bytes=480*640*3
    if n>65536 or len(payload)!=4+n+2*frame_bytes: raise ValueError('Wrong binary frame size')
    request=json.loads(payload[4:4+n])
    if request.get('contract_version')!=CONTRACT_VERSION or request.get('checkpoint_sha256')!=MODEL_SHA256:
        raise ValueError('Checkpoint/contract mismatch')
    if not isinstance(request.get('session_id'),str) or not request['session_id']:
        raise ValueError('Missing session_id')
    if type(request.get('observation_sequence')) is not int or request['observation_sequence']<0:
        raise ValueError('Invalid observation sequence')
    frames={k:np.frombuffer(payload,dtype=np.uint8,count=frame_bytes,offset=4+n+i*frame_bytes).reshape(480,640,3).copy()
            for i,k in enumerate(CAMERA_ORDER)}
    obs=make_observation(np.asarray(request['state'],np.float32),frames,request['instruction'])
    return request,obs|{'task':request['instruction']}


def encode_request(observation: dict, sequence: int, session_id: str, instruction: str) -> dict:
    cameras = {}
    for key in CAMERA_ORDER:
        stream = io.BytesIO()
        Image.fromarray(observation[key]).save(stream, format="PNG", compress_level=1)
        cameras[key] = base64.b64encode(stream.getvalue()).decode("ascii")
    return {"contract_version": CONTRACT_VERSION, "checkpoint_sha256": MODEL_SHA256,
            "session_id": session_id, "observation_sequence": sequence,
            "observation_timestamp_ns": time.monotonic_ns(), "instruction": instruction,
            "state": np.asarray(observation["observation.state"]).tolist(), "cameras": cameras}


def decode_request(request: dict) -> dict:
    if request.get("contract_version") != CONTRACT_VERSION or request.get("checkpoint_sha256") != MODEL_SHA256:
        raise ValueError("Checkpoint/contract mismatch")
    if not isinstance(request.get("session_id"), str) or not request["session_id"]:
        raise ValueError("Missing session_id")
    if not isinstance(request.get("observation_sequence"), int) or request["observation_sequence"] < 0:
        raise ValueError("Invalid observation sequence")
    images = {}
    for key in CAMERA_ORDER:
        raw = base64.b64decode(request["cameras"][key], validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.mode != "RGB":
                raise ValueError("Camera must be RGB")
            images[key] = np.array(im)
    obs = make_observation(np.asarray(request["state"], dtype=np.float32), images, request["instruction"])
    return obs | {"task": request["instruction"]}


class RemotePolicy:
    def __init__(self, server: str, timeout_s: float = 2.0, transport: str = 'png'):
        self.server = server.rstrip("/")
        self.timeout_s = timeout_s
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        if transport not in ('png','binary'): raise ValueError('Unknown transport')
        self.transport=transport
        parsed = urlsplit(self.server)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Expected an explicit HTTP(S) inference endpoint")
        cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
        self.connection = cls(parsed.hostname, parsed.port, timeout=timeout_s)
        self.base_path = parsed.path.rstrip("/")

    def infer_chunk(self, observation: dict, instruction: str, *, full_horizon: bool = False):
        sequence = self.sequence
        self.sequence += 1
        start = time.perf_counter()
        if self.transport=='binary':
            payload=encode_binary_request(observation,sequence,self.session_id,instruction,'async50' if full_horizon else 'sync10')
            content_type='application/x-so101-observation'
        else:
            body = encode_request(observation, sequence, self.session_id, instruction)
            body['action_mode']='async50' if full_horizon else 'sync10'
            payload = json.dumps(body,allow_nan=False).encode()
            content_type='application/json'
        t1 = time.perf_counter()
        try:
            self.connection.request("POST",self.base_path+"/infer",body=payload,headers={"Content-Type":content_type})
            response = self.connection.getresponse()
            received = response.read()
            t7 = time.perf_counter()
            if response.status != 200: raise RuntimeError(f"Inference HTTP {response.status}: {received[:400]!r}")
            result = json.loads(received)
        except Exception:
            self.connection.close()  # Never transparently replay a timed-out request.
            raise
        elapsed = t7 - start
        if elapsed > self.timeout_s:
            raise TimeoutError("Policy response arrived after timeout")
        if result.get("observation_sequence") != sequence or result.get("session_id") != self.session_id:
            raise ValueError("Stale or mismatched action response")
        if result.get("contract_version") != CONTRACT_VERSION or result.get("checkpoint_sha256") != MODEL_SHA256:
            raise ValueError("Response checkpoint/contract mismatch")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (50 if full_horizon else 10, 6) or not np.isfinite(actions).all():
            raise ValueError("Wrong finite action chunk shape for requested inference mode")
        if result.get("action_sequence") != sequence or result.get("representation") != "absolute":
            raise ValueError("Wrong action sequence/representation")
        t8 = time.perf_counter()
        result["client_trace"] = {"T0_capture_complete":start,"T0_definition":"prerecorded observation available to caller; live capture timestamps separate",
             "T1_serialization_complete":t1,"T7_client_receive":t7,"T8_action_available":t8,
             "serialization_ms":(t1-start)*1000,"client_decode_ms":(t8-t7)*1000,
             "total_ms":(t8-start)*1000,"payload_bytes":len(payload),"response_bytes":len(received)}
        if t8-start > self.timeout_s: raise TimeoutError("Policy processing exceeded complete request deadline")
        return actions, t8-start, result

    def close(self):
        self.connection.close()
