#!/usr/bin/env python3
"""Generate arrow-connected flow diagrams (SVG) for the architecture docs.

Each diagram is a left-to-right flow of rounded-rect nodes joined by line +
arrowhead connectors. Node text is rendered with <foreignObject> HTML so the
browser handles fonts (incl. CJK), wrapping, and weight. Arrows are real SVG
shapes. Output is light-themed to match the docs shell (color-scheme: light).

Run:  python3 scripts/gen_arch_flow_diagrams.py
Writes SVGs to public/diagrams/.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

# ── palette (matches app/styles/tokens.css light theme) ──────────────────
# Light values are the defaults; dark values apply via prefers-color-scheme
# inside each SVG so the diagram adapts when the browser is in dark mode.
LIGHT = {
    "bg": "#f7f5ef",
    "card": "#ffffff",
    "ink": "#111111",
    "muted": "#65645f",
    "line": "#d8d4c9",
    "accent": "#9a8568",
    "band": "#efe9df",  # boundary-band tint, between bg and line
}
DARK = {
    "bg": "#141311",
    "card": "#22201c",
    "ink": "#f3efe6",
    "muted": "#b3ada3",
    "line": "#3a3733",
    "accent": "#c4ad92",
    "band": "#2a2722",
}

# ── layout ────────────────────────────────────────────────────────────────
NODE_W = 212
NODE_H = 104
GAP = 48      # arrow region width
PAD = 16

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "diagrams"


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def node_html(role: str | None, title: str, detail: str | None) -> str:
    role_chip = (
        f'<span class="role">{esc(role)}</span>' if role else ""
    )
    detail_html = (
        f'<p class="detail">{esc(detail)}</p>' if detail else ""
    )
    return (
        f'<div class="node">'
        f'<div class="head">{role_chip}<span class="idx"></span></div>'
        f'<div class="title">{esc(title)}</div>'
        f'{detail_html}'
        f'</div>'
    )


def render(steps: list[dict], out_path: Path, aria: str) -> None:
    n = len(steps)
    width = 2 * PAD + n * NODE_W + (n - 1) * GAP
    height = 2 * PAD + NODE_H

    nodes_svg: list[str] = []
    arrows_svg: list[str] = []
    for i, st in enumerate(steps):
        x = PAD + i * (NODE_W + GAP)
        y = PAD
        # box
        nodes_svg.append(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="10" ry="10" '
            f'style="fill:var(--c-card);stroke:var(--c-line);stroke-width:1"/>'
        )
        # html text
        fo = node_html(st.get("role"), st["title"], st.get("detail"))
        nodes_svg.append(
            f'<foreignObject x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}">{fo}</foreignObject>'
        )
        # arrow to next
        if i < n - 1:
            ax1 = x + NODE_W
            ax2 = x + NODE_W + GAP
            ay = y + NODE_H / 2
            arrows_svg.append(
                f'<line x1="{ax1}" y1="{ay}" x2="{ax2 - 7}" y2="{ay}" '
                f'style="stroke:var(--c-accent);stroke-width:1.5"/>'
            )
            arrows_svg.append(
                f'<polygon points="{ax2 - 7},{ay - 5} {ax2 - 7},{ay + 5} {ax2},{ay}" '
                f'style="fill:var(--c-accent)"/>'
            )

    style = f"""  <style>
    :root {{
      --c-bg: {LIGHT['bg']}; --c-card: {LIGHT['card']}; --c-ink: {LIGHT['ink']};
      --c-muted: {LIGHT['muted']}; --c-line: {LIGHT['line']}; --c-accent: {LIGHT['accent']};
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --c-bg: {DARK['bg']}; --c-card: {DARK['card']}; --c-ink: {DARK['ink']};
        --c-muted: {DARK['muted']}; --c-line: {DARK['line']}; --c-accent: {DARK['accent']};
      }}
    }}
    .node {{ box-sizing: border-box; width: 100%; height: 100%; padding: 10px 12px; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; display: flex; flex-direction: column; gap: 3px; }}
    .head {{ display: flex; align-items: center; gap: 6px; min-height: 16px; }}
    .role {{ font-size: 10px; line-height: 1; color: var(--c-muted); border: 1px solid var(--c-line); border-radius: 3px; padding: 2px 5px; white-space: nowrap; }}
    .title {{ font-size: 13px; font-weight: 650; color: var(--c-ink); line-height: 1.25; }}
    .detail {{ margin: 2px 0 0; font-size: 11px; line-height: 1.45; color: var(--c-muted); }}
  </style>"""

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}">
  <rect x="0" y="0" width="{width}" height="{height}" style="fill:var(--c-bg)"/>
  {''.join(arrows_svg)}
  {''.join(nodes_svg)}
{style}
</svg>
'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def render_boundary(data: dict, out_path: Path, aria: str) -> None:
    """Two-column boundary diagram: model execution | artifact boundary | benchmark.

    Left and right columns each hold a vertical flow of node cards. A tinted
    central band marks the artifact boundary; one bold arrow crosses it to
    show that only GenerationResult / artifact refs cross, never checkpoints
    or evaluators. Node text reuses node_html so the grammar matches the
    linear flow diagrams.
    """
    PAD = 20
    COL_W = 372
    BAND_W = 130
    HEADER_H = 44
    NODE_H = 54
    NODE_VGAP = 14
    CAPTION_H = 40

    left = data["left_nodes"]
    right = data["right_nodes"]
    n_max = max(len(left), len(right))
    nodes_h = n_max * NODE_H + (n_max - 1) * NODE_VGAP
    inner_top = PAD + HEADER_H + 12
    caption_top = inner_top + nodes_h + 18
    col_bottom = caption_top + CAPTION_H
    col_h = col_bottom - PAD
    height = col_bottom + PAD
    width = 2 * PAD + 2 * COL_W + BAND_W

    lx = PAD
    band_x = lx + COL_W
    rx = band_x + BAND_W
    mid_y = inner_top + nodes_h / 2
    node_w = COL_W - 32
    node_x_left = lx + 16
    node_x_right = rx + 16

    def node_span(nodes: list[dict]) -> float:
        cnt = len(nodes)
        return cnt * NODE_H + (cnt - 1) * NODE_VGAP

    def col_nodes(nodes: list[dict], x0: float) -> list[str]:
        start_y = inner_top + (nodes_h - node_span(nodes)) / 2
        out: list[str] = []
        for i, st in enumerate(nodes):
            y = start_y + i * (NODE_H + NODE_VGAP)
            out.append(
                f'<rect x="{x0}" y="{y}" width="{node_w}" height="{NODE_H}" rx="8" ry="8" '
                f'style="fill:var(--c-card);stroke:var(--c-line);stroke-width:1"/>'
            )
            out.append(
                f'<foreignObject x="{x0}" y="{y}" width="{node_w}" height="{NODE_H}">'
                f'{node_html(st.get("role"), st["title"], st.get("detail"))}</foreignObject>'
            )
            if i < len(nodes) - 1:
                ax = x0 + node_w / 2
                ay1 = y + NODE_H
                ay2 = y + NODE_H + NODE_VGAP
                out.append(
                    f'<line x1="{ax}" y1="{ay1}" x2="{ax}" y2="{ay2 - 5}" '
                    f'style="stroke:var(--c-accent);stroke-width:1.5"/>'
                )
                out.append(
                    f'<polygon points="{ax - 5},{ay2 - 5} {ax + 5},{ay2 - 5} {ax},{ay2}" '
                    f'style="fill:var(--c-accent)"/>'
                )
        return out

    parts: list[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" style="fill:var(--c-bg)"/>'
    ]

    # column containers (border only, nodes sit inside)
    for cx in (lx, rx):
        parts.append(
            f'<rect x="{cx}" y="{PAD}" width="{COL_W}" height="{col_h}" rx="12" ry="12" '
            f'style="fill:none;stroke:var(--c-line);stroke-width:1"/>'
        )
    # column headers
    for cx, title in ((lx, data["left_title"]), (rx, data["right_title"])):
        parts.append(
            f'<foreignObject x="{cx}" y="{PAD}" width="{COL_W}" height="{HEADER_H}">'
            f'<div class="bcol-head">{esc(title)}</div></foreignObject>'
        )

    # boundary band (tinted + dashed to read as "the interface")
    parts.append(
        f'<rect x="{band_x}" y="{PAD}" width="{BAND_W}" height="{col_h}" rx="12" ry="12" '
        f'style="fill:var(--c-band);stroke:var(--c-line);stroke-width:1;stroke-dasharray:4 3"/>'
    )
    parts.append(
        f'<foreignObject x="{band_x}" y="{PAD + 10}" width="{BAND_W}" height="44">'
        f'<div class="band-label">{esc(data["boundary_label"])}</div></foreignObject>'
    )

    # inner node flows
    parts.extend(col_nodes(left, node_x_left))
    parts.extend(col_nodes(right, node_x_right))

    # bold cross-boundary arrow at mid-height, label above and below
    parts.append(
        f'<foreignObject x="{band_x}" y="{mid_y - 30}" width="{BAND_W}" height="22">'
        f'<div class="cross-main">{esc(data["cross_main"])}</div></foreignObject>'
    )
    parts.append(
        f'<line x1="{band_x}" y1="{mid_y}" x2="{rx - 9}" y2="{mid_y}" '
        f'style="stroke:var(--c-accent);stroke-width:2.5"/>'
    )
    parts.append(
        f'<polygon points="{rx - 9},{mid_y - 6} {rx - 9},{mid_y + 6} {rx},{mid_y}" '
        f'style="fill:var(--c-accent)"/>'
    )
    parts.append(
        f'<foreignObject x="{band_x}" y="{mid_y + 6}" width="{BAND_W}" height="20">'
        f'<div class="cross-sub">{esc(data["cross_sub"])}</div></foreignObject>'
    )

    # captions under each column
    for cx, cap in ((lx, data["left_caption"]), (rx, data["right_caption"])):
        parts.append(
            f'<foreignObject x="{cx}" y="{caption_top}" width="{COL_W}" height="{CAPTION_H}">'
            f'<div class="bcol-cap">{esc(cap)}</div></foreignObject>'
        )

    style = f"""  <style>
    :root {{
      --c-bg: {LIGHT['bg']}; --c-card: {LIGHT['card']}; --c-ink: {LIGHT['ink']};
      --c-muted: {LIGHT['muted']}; --c-line: {LIGHT['line']}; --c-accent: {LIGHT['accent']};
      --c-band: {LIGHT['band']};
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --c-bg: {DARK['bg']}; --c-card: {DARK['card']}; --c-ink: {DARK['ink']};
        --c-muted: {DARK['muted']}; --c-line: {DARK['line']}; --c-accent: {DARK['accent']};
        --c-band: {DARK['band']};
      }}
    }}
    .node {{ box-sizing: border-box; width: 100%; height: 100%; padding: 7px 10px; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; display: flex; flex-direction: column; gap: 2px; }}
    .head {{ display: flex; align-items: center; gap: 6px; min-height: 14px; }}
    .role {{ font-size: 9px; line-height: 1; color: var(--c-muted); border: 1px solid var(--c-line); border-radius: 3px; padding: 1px 4px; white-space: nowrap; }}
    .title {{ font-size: 12px; font-weight: 650; color: var(--c-ink); line-height: 1.2; }}
    .detail {{ margin: 1px 0 0; font-size: 10px; line-height: 1.4; color: var(--c-muted); }}
    .bcol-head {{ box-sizing: border-box; width: 100%; height: 100%; display: flex; align-items: center; padding: 0 16px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; font-weight: 700; letter-spacing: 0.02em; color: var(--c-ink); }}
    .bcol-cap {{ box-sizing: border-box; width: 100%; height: 100%; padding: 4px 16px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 10px; line-height: 1.4; color: var(--c-muted); font-style: italic; }}
    .band-label {{ box-sizing: border-box; width: 100%; text-align: center; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 10px; line-height: 1.25; font-weight: 700; letter-spacing: 0.04em; color: var(--c-muted); text-transform: uppercase; }}
    .cross-main {{ box-sizing: border-box; width: 100%; text-align: center; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 11px; line-height: 1.2; font-weight: 700; color: var(--c-accent); }}
    .cross-sub {{ box-sizing: border-box; width: 100%; text-align: center; font-family: ui-mono, ui-sans-serif, system-ui, sans-serif; font-size: 9px; line-height: 1.2; color: var(--c-muted); }}
  </style>"""

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}">
  {''.join(parts)}
{style}
</svg>
'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


# ── diagram data ──────────────────────────────────────────────────────────

EVAL = {
    "aria": "Evaluation run flow: request to scorecard",
    "steps": [
        {"title": "Request", "role": "Entry", "detail": "EvaluateRunRequest / ModelBenchmarkRunRequest"},
        {"title": "Facade", "role": "Orchestration", "detail": "run_evaluate: validate + pick delegate"},
        {"title": "Delegate runner", "role": "Execution", "detail": "ContractRunner / ExistingResults / ModelBenchmark"},
        {"title": "Generate", "role": "Model", "detail": "model runner → GenerationResult"},
        {"title": "Metrics", "role": "Metrics", "detail": "per-sample + aggregate"},
        {"title": "Scorecard + report", "role": "Report", "detail": "scorecard.json, report.md"},
    ],
}
RUNTIME = {
    "aria": "Model runtime assembly chain",
    "steps": [
        {"title": "Catalog binding", "role": "Control plane", "detail": "pipeline_target / runtime_profile"},
        {"title": "PipelineABC", "role": "Runtime", "detail": "load / process / stream"},
        {"title": "Operator", "role": "Input", "detail": "validate, media, camera/action, interaction"},
        {"title": "Synthesis", "role": "Inference", "detail": "native runtime or hosted API"},
        {"title": "Representation", "role": "Geometry", "detail": "depth / point cloud / 3DGS"},
        {"title": "Memory", "role": "State", "detail": "streaming / multi-turn"},
        {"title": "GenerationResult", "role": "Artifact", "detail": "normalized outputs + metadata"},
    ],
}
SURF = {
    "aria": "Surfaces to orchestration to execution flow",
    "steps": [
        {"title": "Surface", "role": "Entry", "detail": "CLI / TUI / MCP / Studio"},
        {"title": "Intent", "role": "Intent", "detail": "ModelBenchmark / ScoreArtifacts / GenerateAndScore / ScoreResults / Reproduce"},
        {"title": "prepare_evaluation", "role": "Orchestration", "detail": "validate + preflight diagnostics"},
        {"title": "PreparedEvaluation", "role": "Plan", "detail": "claim policy + readiness"},
        {"title": "execute_prepared", "role": "Compile", "detail": "→ EvaluateRunRequest / ModelBenchmarkRunRequest"},
        {"title": "Delegate runner", "role": "Execution", "detail": "see Workflow"},
    ],
}
DIFF = {
    "aria": "Native diffusion assembly flow",
    "steps": [
        {"title": "Catalog / profile", "role": "Control plane", "detail": "model ID + runtime policy"},
        {"title": "NativeDiffusionPipeline", "role": "Boundary", "detail": "public entrypoint (pipeline.py:16)"},
        {"title": "Recipe", "role": "Recipe", "detail": "immutable component + checkpoint specs"},
        {"title": "Assembler", "role": "Assembly", "detail": "resolve + validate (assembly.py:31)"},
        {"title": "Runner", "role": "Execution", "detail": "condition / init / schedule / denoise / decode"},
        {"title": "Artifact", "role": "Output", "detail": "normalized WorldFoundry artifact"},
    ],
}

EVAL_ZH = {
    "aria": "评测 run 流程：request 到 scorecard",
    "steps": [
        {"title": "Request", "role": "入口", "detail": "EvaluateRunRequest / ModelBenchmarkRunRequest"},
        {"title": "Facade", "role": "编排", "detail": "run_evaluate：校验 + 选 delegate"},
        {"title": "Delegate runner", "role": "执行", "detail": "ContractRunner / ExistingResults / ModelBenchmark"},
        {"title": "Generate", "role": "模型", "detail": "model runner → GenerationResult"},
        {"title": "Metrics", "role": "指标", "detail": "逐样本 + 聚合"},
        {"title": "Scorecard + report", "role": "报告", "detail": "scorecard.json、report.md"},
    ],
}
RUNTIME_ZH = {
    "aria": "模型 runtime 组装链",
    "steps": [
        {"title": "Catalog 绑定", "role": "控制平面", "detail": "pipeline_target / runtime_profile"},
        {"title": "PipelineABC", "role": "运行时", "detail": "load / process / stream"},
        {"title": "Operator", "role": "输入", "detail": "校验、媒体、camera/action、interaction"},
        {"title": "Synthesis", "role": "推理", "detail": "native runtime 或 hosted API"},
        {"title": "Representation", "role": "几何", "detail": "深度 / 点云 / 3DGS"},
        {"title": "Memory", "role": "状态", "detail": "流式 / 多轮"},
        {"title": "GenerationResult", "role": "产物", "detail": "归一化输出 + metadata"},
    ],
}
SURF_ZH = {
    "aria": "使用界面到编排到执行的流程",
    "steps": [
        {"title": "界面", "role": "入口", "detail": "CLI / TUI / MCP / Studio"},
        {"title": "Intent", "role": "意图", "detail": "ModelBenchmark / ScoreArtifacts / GenerateAndScore / ScoreResults / Reproduce"},
        {"title": "prepare_evaluation", "role": "编排", "detail": "校验 + preflight 诊断"},
        {"title": "PreparedEvaluation", "role": "计划", "detail": "claim 策略 + 就绪状态"},
        {"title": "execute_prepared", "role": "编译", "detail": "→ EvaluateRunRequest / ModelBenchmarkRunRequest"},
        {"title": "Delegate runner", "role": "执行", "detail": "见工作流"},
    ],
}
DIFF_ZH = {
    "aria": "原生扩散装配流程",
    "steps": [
        {"title": "Catalog / profile", "role": "控制平面", "detail": "model ID + runtime policy"},
        {"title": "NativeDiffusionPipeline", "role": "边界", "detail": "公开入口（pipeline.py:16）"},
        {"title": "Recipe", "role": "Recipe", "detail": "不可变 component + checkpoint spec"},
        {"title": "Assembler", "role": "装配", "detail": "解析 + 校验（assembly.py:31）"},
        {"title": "Runner", "role": "执行", "detail": "condition / init / schedule / denoise / decode"},
        {"title": "Artifact", "role": "产物", "detail": "归一化 WorldFoundry artifact"},
    ],
}

DIAGRAMS = [
    ("flow-evaluation-core", EVAL, EVAL["aria"]),
    ("flow-model-runtime", RUNTIME, RUNTIME["aria"]),
    ("flow-surfaces-orchestration", SURF, SURF["aria"]),
    ("flow-native-diffusion", DIFF, DIFF["aria"]),
    ("flow-evaluation-core.zh", EVAL_ZH, EVAL_ZH["aria"]),
    ("flow-model-runtime.zh", RUNTIME_ZH, RUNTIME_ZH["aria"]),
    ("flow-surfaces-orchestration.zh", SURF_ZH, SURF_ZH["aria"]),
    ("flow-native-diffusion.zh", DIFF_ZH, DIFF_ZH["aria"]),
]


BOUNDARY = {
    "aria": "Model execution and benchmark evaluation separated by the artifact boundary",
    "left_title": "Model execution",
    "right_title": "Benchmark / reporting",
    "boundary_label": "Artifact boundary",
    "cross_main": "GenerationResult",
    "cross_sub": "artifacts.jsonl",
    "left_caption": "model side never imports a benchmark evaluator",
    "right_caption": "benchmark side never loads a model checkpoint",
    "left_nodes": [
        {"title": "GenerationRequest", "role": "Input", "detail": "sample id, media, text/action, params"},
        {"title": "Pipeline + Operator", "role": "Runtime", "detail": "checkpoint load + native call"},
        {"title": "Native inference", "role": "Inference", "detail": "local runtime or hosted API"},
        {"title": "GenerationResult", "role": "Artifact", "detail": "status, timing, outputs, metadata"},
    ],
    "right_nodes": [
        {"title": "Metrics / benchmark runner", "role": "Evaluation", "detail": "reads artifacts, not checkpoints"},
        {"title": "per_sample + summary", "role": "Metrics", "detail": "per-sample rows + aggregates"},
        {"title": "scorecard.json + report.md", "role": "Report", "detail": "coverage, blockers, eligibility"},
    ],
}
BOUNDARY_ZH = {
    "aria": "模型执行与 benchmark 评测以 artifact 边界分离",
    "left_title": "模型执行",
    "right_title": "评测 / 报告",
    "boundary_label": "Artifact 边界",
    "cross_main": "GenerationResult",
    "cross_sub": "artifacts.jsonl",
    "left_caption": "模型侧不 import 任何 benchmark evaluator",
    "right_caption": "benchmark 侧不加载任何模型 checkpoint",
    "left_nodes": [
        {"title": "GenerationRequest", "role": "输入", "detail": "样本 id、媒体、text/action、参数"},
        {"title": "Pipeline + Operator", "role": "运行时", "detail": "checkpoint 加载 + 原生调用"},
        {"title": "Native inference", "role": "推理", "detail": "本地 runtime 或托管 API"},
        {"title": "GenerationResult", "role": "产物", "detail": "状态、耗时、输出、metadata"},
    ],
    "right_nodes": [
        {"title": "Metrics / benchmark runner", "role": "评测", "detail": "读 artifact，不读 checkpoint"},
        {"title": "per_sample + summary", "role": "指标", "detail": "逐样本行 + 聚合"},
        {"title": "scorecard.json + report.md", "role": "报告", "detail": "覆盖、blocker、合格性"},
    ],
}

BOUNDARY_DIAGRAMS = [
    ("boundary", BOUNDARY, BOUNDARY["aria"]),
    ("boundary.zh", BOUNDARY_ZH, BOUNDARY_ZH["aria"]),
]


def main() -> int:
    for name, data, aria in DIAGRAMS:
        out = OUT / f"{name}.svg"
        render(data["steps"], out, aria)
        print(f"wrote {out.relative_to(ROOT)}  ({len(data['steps'])} steps)")
    for name, data, aria in BOUNDARY_DIAGRAMS:
        out = OUT / f"{name}.svg"
        render_boundary(data, out, aria)
        print(f"wrote {out.relative_to(ROOT)}  (boundary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
