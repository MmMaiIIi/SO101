"""PyAV decode integrity and content-based freshness proxies, never sensor timestamps."""
import hashlib
from itertools import zip_longest
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw
from qc_signal import longest_run


def fingerprints(rgb):
    """Exact decoded RGB SHA256, quantized 32x24 grayscale fingerprint, and 64-bit dHash."""
    image = Image.fromarray(rgb).convert("L")
    compact = np.asarray(image.resize((32, 24), Image.Resampling.BILINEAR)) // 8
    small = np.asarray(image.resize((9, 8), Image.Resampling.BILINEAR))
    dhash = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
    return hashlib.sha256(rgb.tobytes()).hexdigest(), hashlib.sha256(compact.tobytes()).hexdigest(), dhash


def duplicate_stats(mae, exact, compact_same, dhash_dist, motion):
    """Adaptive low/high MAE clusters; moving+unchanged is only a stale-frame candidate."""
    mae = np.asarray(mae, float)
    exact = np.asarray(exact, bool)
    compact_same = np.asarray(compact_same, bool)
    positive = mae[mae > 0]
    threshold = 0.0
    if len(positive):
        logs = np.log(positive)
        low, high = np.percentile(logs, [25, 75])
        for _ in range(20):
            mask = logs <= (low + high) / 2
            if mask.all() or not mask.any():
                break
            low, high = float(logs[mask].mean()), float(logs[~mask].mean())
        threshold = float(np.exp((low + high) / 2))
    near = (mae <= threshold) & (np.asarray(dhash_dist) <= 1)
    near[0] = False
    motion = np.asarray(motion, float)[:len(mae)]
    positive_motion = motion[motion > 1e-8]
    moving_threshold = float(np.median(positive_motion)) if len(positive_motion) else 1e-8
    moving = motion > moving_threshold
    same_proxy = exact | (compact_same & near)
    suspected = moving & same_proxy
    edges = max(1, len(mae) - 1)
    return {"exact_duplicate_ratio": float(exact.sum()/edges),
            "near_duplicate_ratio": float(near.sum()/edges), "near_MAE_threshold": threshold,
            "longest_exact_duplicate_run_frames": longest_run(exact) + 1 if exact.any() else 1,
            "longest_near_static_run_frames": longest_run(near) + 1 if near.any() else 1,
            "compact_same_ratio": float(compact_same.sum()/edges),
            "moving_threshold": moving_threshold, "moving_edges": int(moving.sum()),
            "moving_but_same_frame_ratio": float(suspected.sum()/max(1,moving.sum())),
            "moving_but_exact_frame_ratio": float((moving & exact).sum()/max(1,moving.sum())),
            "longest_moving_same_run_edges": longest_run(suspected),
            "suspect_frame_indices": np.flatnonzero(suspected).tolist(),
            "mean_pixel_MAE": float(mae[1:].mean()) if len(mae)>1 else 0,
            "sensor_timestamp_available": False,
            "interpretation": "Lossy decode duplicates are proxies; static scenes and compression can obscure or mimic capture repeats."}


def scan_file(path):
    """Fully decode one source MP4, including PTS order and corrupt-frame flags."""
    result = {"path": str(path), "decode_ok": False, "decoded_frames": 0, "corrupt_frames": 0}
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            result.update(codec=stream.codec_context.name, width=stream.width, height=stream.height,
                          nominal_fps=float(stream.average_rate) if stream.average_rate else None,
                          declared_frames=stream.frames, pixel_format=stream.codec_context.format.name,
                          pts_monotonic=True)
            previous = None
            for frame in container.decode(stream):
                result["decoded_frames"] += 1
                result["corrupt_frames"] += int(frame.is_corrupt)
                if frame.pts is None:
                    result["pts_monotonic"] = False
                else:
                    t = float(frame.pts * frame.time_base)
                    if previous is not None and t <= previous:
                        result["pts_monotonic"] = False
                    previous = t
            result["decode_ok"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def clip_frames(path, start, end):
    """Yield decoded RGB frames inside official half-open episode video timestamps."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if start > 0:
            container.seek(int(start / float(stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Source frame is missing PTS")
            t = float(frame.pts * frame.time_base)
            if t < start - 1e-6:
                continue
            if t >= end - 1e-6:
                break
            yield frame.to_ndarray(format="rgb24"), t


def analyze_clip(path, start, end, expected, motion, shape):
    """Decode a segment, measure pixel/hash changes, and sample five boundary/key images."""
    selected = {int(round(q*(expected-1))): q for q in (0, .25, .5, .75, 1)}
    metrics = {"path": str(path), "from_timestamp": start, "to_timestamp": end,
               "decoded_frames": 0, "decode_ok": False, "shape_ok": True, "dtype_ok": True}
    mae, exact, compact, distances, times = [], [], [], [], []
    samples = {}
    prev_rgb = prev_hash = None
    try:
        for i, (rgb, timestamp) in enumerate(clip_frames(path, start, end)):
            hashes = fingerprints(rgb)
            metrics["shape_ok"] &= rgb.shape == tuple(shape)
            metrics["dtype_ok"] &= rgb.dtype == np.uint8
            times.append(timestamp)
            mae.append(float(np.mean(np.abs(rgb.astype(np.int16)-prev_rgb.astype(np.int16)))) if i else 0)
            exact.append(hashes[0] == prev_hash[0] if i else False)
            compact.append(hashes[1] == prev_hash[1] if i else False)
            distances.append((hashes[2]^prev_hash[2]).bit_count() if i else 0)
            if i in selected:
                samples[selected[i]] = Image.fromarray(rgb).resize((320,240))
            prev_rgb, prev_hash = rgb, hashes
        metrics["decoded_frames"] = len(mae)
        metrics["decode_ok"] = True
        metrics["pts_monotonic"] = bool(np.all(np.diff(times)>0))
        if len(mae) and len(motion)>=len(mae):
            metrics.update(duplicate_stats(mae,exact,compact,distances,motion))
    except Exception as exc:
        metrics["error"] = repr(exc)
    metrics["frame_count_matches"] = metrics["decoded_frames"] == expected
    series = {"pixel_MAE": mae, "exact_same": exact, "compact_same": compact,
              "dhash_hamming": distances, "video_pts": times}
    return metrics, series, samples


def contact_sheet(path, samples, episode, task):
    """Two camera rows at start/25/50/75/end; explicitly label missing keyframes."""
    canvas = Image.new("RGB", (1600, 590), "#eef2f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((12,12), f"Episode {episode} | {task}", fill="black")
    for row, camera in enumerate(("front","wrist")):
        y = 45+row*270
        for col, q in enumerate((0,.25,.5,.75,1)):
            x = col*320
            if q in samples.get(camera,{}):
                canvas.paste(samples[camera][q],(x,y+20))
            draw.text((x+8,y),f"{camera} | {int(q*100)}%",fill="black")
    canvas.save(path)


def combined_video(path, cameras, episode, task, state, fps):
    """Generate a compact review-only two-camera MP4 with task/time/measured state."""
    front, wrist = (clip_frames(cameras[k]["path"], cameras[k]["from_timestamp"], cameras[k]["to_timestamp"]) for k in ("front","wrist"))
    count = 0
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=int(fps))
        stream.width, stream.height, stream.pix_fmt = 640, 320, "yuv420p"
        stream.options = {"crf":"25","preset":"veryfast"}
        for i, pair in enumerate(zip_longest(front,wrist)):
            if pair[0] is None or pair[1] is None:
                raise ValueError("front/wrist review clip frame counts differ")
            canvas = Image.new("RGB",(640,320),"#111827")
            for col, item in enumerate(pair):
                canvas.paste(Image.fromarray(item[0]).resize((320,240)),(col*320,0))
            draw = ImageDraw.Draw(canvas)
            draw.text((8,245),f"front                         wrist | ep={episode} t={i/fps:.2f}s",fill="white")
            draw.text((8,267),task,fill="white")
            if i<len(state):
                draw.text((8,290),"measured state: "+", ".join(f"{v:.1f}" for v in state[i]),fill="white")
            frame = av.VideoFrame.from_image(canvas)
            frame.pts = i
            for packet in stream.encode(frame):
                output.mux(packet)
            count += 1
        for packet in stream.encode():
            output.mux(packet)
    return count
