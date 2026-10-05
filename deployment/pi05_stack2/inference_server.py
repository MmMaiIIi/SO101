from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import socket
import json
from pathlib import Path
import time

from .config import MODEL_SHA256, REVISION
from .logging_utils import RunLogger
from .model_loader import load_policy
from .policy_runner import PolicyRunner
from .protocol import CONTRACT_VERSION, decode_request, decode_binary_request


def main():
    ap = argparse.ArgumentParser(description="Local π0.5 server; expose via SSH tunnel, no motor access")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    policy, pre, post, report = load_policy(args.checkpoint, args.device)
    runner = PolicyRunner(policy, pre, post, args.device)
    log = RunLogger()
    log.log("server_start", model=report, contract_version=CONTRACT_VERSION, hardware_access=False)
    sequences = {}
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.connection.settimeout(10)
        def log_message(self, *_):
            pass

        def send_json(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self.send_json(404, {"error": "unknown endpoint"})
            self.send_json(200, {"ready": True, "checkpoint_sha256": MODEL_SHA256,
                                 "contract_version": CONTRACT_VERSION, "device": args.device})

        def do_POST(self):
            if self.path != "/infer":
                return self.send_json(404, {"error": "unknown endpoint"})
            try:
                t2 = time.monotonic()
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 8 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                payload=self.rfile.read(length)
                decode_started = time.monotonic()
                if self.headers.get('Content-Type')=='application/x-so101-observation':
                    request,obs=decode_binary_request(payload)
                else:
                    request=json.loads(payload)
                    obs=decode_request(request)
                decode_done = time.monotonic()
                session, seq = request["session_id"], request["observation_sequence"]
                mode=request.get('action_mode','sync10')
                if mode not in ('sync10','async50'): raise ValueError('Unknown action mode')
                with inference_lock:
                    if seq <= sequences.get(session, -1):
                        raise ValueError("Stale observation sequence")
                    sequences[session] = seq
                    actions, latency = runner.infer_chunk(obs,full_horizon=mode=='async50')
                    model_trace = dict(runner.last_trace)
                result = {"contract_version": CONTRACT_VERSION, "checkpoint_sha256": MODEL_SHA256,
                          "checkpoint_revision": REVISION, "session_id": session,
                          "observation_sequence": seq, "action_sequence": seq,
                          "observation_timestamp_ns": request["observation_timestamp_ns"],
                          "action_timestamp_ns": time.monotonic_ns(), "latency_s": latency,
                          "representation": "absolute", "actions": actions.tolist()}
                result["model_trace"] = model_trace
                result["server_trace"] = {"T2_server_receive":t2,"T6_server_send":time.monotonic(),
                     "upload_body_read_ms":(decode_started-t2)*1000,"image_decode_ms":(decode_done-decode_started)*1000,
                     "lock_wait_ms":max(0,(time.monotonic()-decode_done-latency)*1000)}
                log.log("query", **result)
                self.send_json(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                log.log("inference_error", error_type=type(exc).__name__, error=str(exc))
                self.send_json(500, {"error": "Inference failed; queue cleared"})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"READY {args.host}:{args.port} checkpoint={MODEL_SHA256} hardware_access=false", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        log.close()


if __name__ == "__main__":
    main()
