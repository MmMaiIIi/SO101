"""Minimal local HTML reports and CSV sidecars with manual labels preserved."""
import csv
import html
import json
from pathlib import Path


def write_csv(path, rows):
    """Write stable CSV column order from the first row, including empty tables."""
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def manual_csv(path, episodes):
    """Preserve existing KEEP/REVIEW/DROP labels on deterministic QC reruns."""
    existing={}
    if Path(path).exists():
        with Path(path).open(newline="",encoding="utf-8") as f:
            existing={int(r['episode']):r for r in csv.DictReader(f)}
    rows=[]
    for ep in episodes:
        old=existing.get(ep['episode'],{})
        status=old.get('manual_status','')
        if status not in ('','KEEP','REVIEW','DROP'):
            raise ValueError(f"Invalid manual status {status}")
        rows.append(dict(episode=ep['episode'],task=ep['task'],auto_status=ep['auto_status'],
                         manual_status=status,reason=old.get('reason',''),notes=old.get('notes','')))
    write_csv(path,rows)


def generate(report, output):
    """Write both report.html and review.html; no success detector or automatic deletion."""
    esc=lambda x:html.escape(str(x))
    body=["<!doctype html><meta charset='utf-8'><title>SO101 dataset QC</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;background:#f8fafc;color:#172033}section{background:white;padding:20px;margin:22px 0;border:1px solid #ddd;border-radius:8px}img{max-width:100%}video{width:640px;max-width:100%}td,th{padding:7px;border:1px solid #ddd}table{border-collapse:collapse}code{overflow-wrap:anywhere}.WARN{color:#a65a00}.FAIL{color:#bd1717}.PASS{color:#16703c}</style>",
          f"<h1>Dataset QC</h1><p><code>{esc(report['dataset_root'])}</code></p>",
          f"<p>Auto counts: {esc(report.get('status_counts'))}. Integrity errors: {esc(report.get('global_errors',[]))}.</p>",
          "<p>Automatic QC covers integrity and diagnostics. Stacking success and RESET leakage need visual review. Video PTS are encoded timeline, not raw sensor timestamps. Lossy-image repetition metrics are proxies.</p>",
          f"<p>Task coverage: {esc(report.get('task_coverage'))}</p>",
          "<p><a href='episode_summary.csv'>Episode CSV</a> | <a href='manual_review.csv'>Manual labels</a> | <a href='qc_report.json'>Full metric JSON</a></p>"]
    for ep in report.get('episodes',[]):
        idx=ep['episode']; prefix=f"episode_{idx:03d}"
        body.extend([f"<section><h2>Episode {idx} — <span class='{esc(ep['auto_status'])}'>{esc(ep['auto_status'])}</span></h2>",
                     f"<p><strong>{esc(ep['task'])}</strong> | {ep['frames']} frames | {ep['duration_s']:.2f} s | task_index {esc(ep['task_index'])}</p>",
                     f"<p>Flags: {esc(ep['warnings'])}. Hard errors: {esc(ep['errors'])}. Recommendation: {esc(ep['recommendation'])}.</p>",
                     f"<video controls preload='metadata' src='review/{prefix}_combined.mp4'></video>",
                     f"<p><img loading='lazy' src='contact_sheets/{prefix}.png' alt='start 25% 50% 75% end'></p>",
                     "<table><tr><th>Camera</th><th>Decoded</th><th>Exact repeats</th><th>Near-static</th><th>Longest exact frames</th><th>Moving + same proxy</th></tr>"])
        for camera,stats in ep.get('cameras',{}).items():
            body.append(f"<tr><td>{esc(camera)}</td><td>{stats.get('decoded_frames')}</td><td>{stats.get('exact_duplicate_ratio',0):.2%}</td><td>{stats.get('near_duplicate_ratio',0):.2%}</td><td>{stats.get('longest_exact_duplicate_run_frames')}</td><td>{stats.get('moving_but_same_frame_ratio',0):.2%}</td></tr>")
        body.append("</table><details><summary>Action / state lag per joint</summary><table><tr><th>joint</th><th>best RMSE lag frames / ms</th><th>corr</th><th>RMSE</th><th>max-corr lag</th></tr>")
        for joint,stats in ep.get('joints',{}).items():
            lag=stats.get('lag',{})
            body.append(f"<tr><td>{esc(joint)}</td><td>{lag.get('best_lag_frames')} / {lag.get('best_lag_ms')}</td><td>{esc(lag.get('corr'))}</td><td>{esc(lag.get('RMSE'))}</td><td>{lag.get('best_corr_lag_frames')}</td></tr>")
        body.extend(["</table></details>",f"<p>Host camera telemetry (when collected): {esc(ep.get('host_camera_telemetry'))}</p>",f"<p>Boundary heuristics: {esc(ep.get('boundary'))}</p>",
                     f"<img loading='lazy' src='plots/{prefix}_signals.svg'>",f"<img loading='lazy' src='plots/{prefix}_lag.svg'>",
                     f"<img loading='lazy' src='plots/{prefix}_camera.svg'>",f"<img loading='lazy' src='plots/{prefix}_boundary.svg'>",
                     "<details><summary>Source video paths and official intervals</summary><pre>"+esc(json.dumps(ep.get('cameras',{}),indent=2))+"</pre></details></section>"])
    document='\n'.join(body)
    for name in ('report.html','review.html'):
        (Path(output)/name).write_text(document,encoding='utf-8')
