"""Dependency-free, standalone HTML report with inline SVG charts."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path


COLORS = ["#718096", "#ce7445", "#bc9f39", "#087f8c", "#6554c0", "#c64e93"]


def label(result: dict) -> str:
    policy = "RL swarm" if result["policy"] == "rl_swarm" else result["policy"].capitalize()
    return policy + " · " + ("Junction planning" if result["config"].get("replan") else "Departure planning")


def color(result: dict) -> str:
    policies = ["static", "reactive", "predictive", "anticipatory", "cooperative", "rl_swarm"]
    return COLORS[policies.index(result["policy"])]


def chart(results: list[dict], field: str, title: str) -> str:
    width, height, left, top = 860, 210, 48, 15
    max_t = max((r["series"][-1]["tick"] for r in results), default=1) or 1
    max_y = max((row[field] for r in results for row in r["series"]), default=1) or 1
    pieces = [f'<h3>{html.escape(title)}</h3>',
              f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">']
    for fraction in (0, .5, 1):
        y = top + (1 - fraction) * 155
        pieces.append(f'<path d="M{left},{y} H840" stroke="#e3e8ef"/>')
        pieces.append(f'<text x="38" y="{y+4}" text-anchor="end">{max_y*fraction:.0f}</text>')
    for i, result in enumerate(results):
        points = ' '.join(f'{left+row["tick"]/max_t*792:.2f},{top+155-row[field]/max_y*155:.2f}'
                          for row in result["series"])
        dash = ' stroke-dasharray="7 4"' if result["config"].get("replan") else ''
        pieces.append(f'<polyline fill="none" stroke="{color(result)}" stroke-width="2.5"{dash} points="{points}"><title>{html.escape(label(result))}</title></polyline>')
    pieces.extend([f'<text x="48" y="195">0</text><text x="840" y="195" text-anchor="end">{max_t} ticks</text>', '</svg>'])
    return ''.join(pieces)


def write_report(results: list[dict], output: Path, scenario: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps({"scenario": scenario, "results": results}, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["policy", "planning_mode", *results[0]["metrics"]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"policy": result["policy"], "planning_mode": "junction" if result["config"].get("replan") else "departure", **result["metrics"]})
    rows, routes, legend = [], [], []
    for i, result in enumerate(results):
        run_label = html.escape(label(result))
        m = result["metrics"]
        rows.append(f'<tr><th>{run_label}</th><td>{m["average_travel_time"]:.2f}</td>'
                    f'<td>{m["p95_travel_time"]}</td><td>{m["total_delay"]}</td>'
                    f'<td>{m["peak_utilization"]:.2f}×</td>'
                    f'<td>{m["peak_severely_congested_edges"]}</td>'
                    f'<td>{m["window_throughput"]:.3f}</td><td>{m.get("replanning_decisions", 0)}</td><td>{m.get("route_changes", 0)}</td></tr>')
        legend.append(f'<span><i style="background:{color(result)}"></i>{run_label}</span>')
        route_items = ''.join(f'<li><code>{html.escape(path)}</code><strong>{count}</strong></li>'
                              for path, count in result["route_counts"].items())
        routes.append(f'<article><h3>{run_label}</h3><ul>{route_items}</ul></article>')
    scenario_text = html.escape(json.dumps(scenario, indent=2))
    config = results[0]["config"]
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anticipatory traffic routing — experiment report</title><style>
:root{{color-scheme:light;font-family:system-ui,sans-serif;color:#203345;background:#f2f5f8}}
body{{max-width:1120px;margin:0 auto;padding:40px 24px}}h1{{font-size:clamp(28px,5vw,44px);letter-spacing:-1.5px;margin:10px 0}}h2{{font-size:23px}}h3{{font-size:16px}}
.eyebrow{{color:#087f8c;letter-spacing:2px;font-size:12px;font-weight:800}}p{{line-height:1.65}}.lead{{max-width:780px;color:#526576}}section{{background:white;padding:24px;border:1px solid #e0e6ed;border-radius:14px;margin:24px 0}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}}th,td{{text-align:right;padding:14px 12px;border-bottom:1px solid #e3e8ef}}th:first-child{{text-align:left}}thead th{{font-size:12px;color:#526576}}tbody tr:nth-child(4){{background:#ecf8f7}}
.legend{{display:flex;flex-wrap:wrap;gap:18px;font-size:13px}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}svg{{width:100%;height:auto}}svg text{{font:12px system-ui;fill:#66788a}}
.routes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}}article{{background:#f5f7fa;border-radius:10px;padding:8px 18px}}ul{{padding:0;list-style:none}}li{{display:flex;justify-content:space-between;gap:16px;padding:7px 0}}pre{{overflow:auto;font-size:12px}}.note{{font-size:13px;color:#526576}}a{{color:#087f8c}}
</style></head><body><header><div class="eyebrow">TRAFFIC ROUTING / RESEARCH PROTOTYPE</div><h1>Plan for the traffic you create.</h1>
<p class="lead">{len(results)} policy / planning-mode runs, one shared demand trace. Compare observed congestion, predicted background demand, and the downstream impact of committed routes.</p></header>
<section><h2>Policy comparison</h2><p class="note">{html.escape(str(scenario.get("label", "Synthetic network")))} · {html.escape(str(scenario.get("units", "Abstract simulation ticks")))}</p><p class="note">{results[0]['metrics']['vehicles']} vehicles · seed {config['seed']} · forecast horizon {config['forecast_horizon']} ticks · measurement window {config['measurement_window']} ticks</p>
<div class="scroll"><table><thead><tr><th>Policy</th><th>Mean trip<br>(ticks)</th><th>P95 trip<br>(ticks)</th><th>Total delay<br>(vehicle-ticks)</th><th>Peak load<br>ratio</th><th>Peak severe<br>edges</th><th>Throughput<br>(vehicles/tick)</th><th>Replanning<br>decisions</th><th>Route<br>changes</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">Trip and delay metrics include all vehicles through completion. Throughput counts arrivals by the fixed measurement-window boundary. Load ratio = edge occupancy ÷ (entry capacity × free-flow time); it is an occupancy proxy, not physical road density. Severe edges have load ratio ≥ {config['severe_utilization']}.</p></section>
<section><h2>Network over time</h2><p class="note">Color identifies policy. Solid lines: departure planning. Dashed lines: junction planning. Identical curves can overlap.</p><div class="legend">{''.join(legend)}</div>{chart(results, 'queued', 'Vehicles waiting to enter roads')}{chart(results, 'completed', 'Cumulative completed trips')}</section>
<section><h2>Initial assigned routes</h2><p class="note">Controlled vehicles only. Background vehicles follow fixed free-flow shortest paths.</p><div class="routes">{''.join(routes)}</div></section>
<section><h2>Interpretation and limits</h2><p>This is a deterministic point-queue experiment. Roads have fixed transit times, finite admission rates, and unlimited queue storage. The default forecast knows background departures exactly within its horizon; future controlled requests are hidden. Each policy uses the same demand and tie ordering. Cooperative routing minimizes the predicted weighted total travel time, delay, and overload over candidate paths; it is a greedy model-based controller.</p>
<p class="note">Results from a synthetic scenario do not establish performance on real roads. There is no spillback, signal timing, or real-time traffic feed. Junction planning revises remaining routes at branching junctions. RL uses a departure-trained checkpoint without retraining; junction results measure transfer. Candidate paths are limited to the configured number of cheapest simple free-flow paths.</p>
<details><summary>Scenario and reproducibility</summary><pre>{scenario_text}</pre></details><p><a href="metrics.csv">Metrics CSV</a> · <a href="results.json">Complete results JSON</a></p></section></body></html>'''
    (output / "report.html").write_text(document, encoding="utf-8")
