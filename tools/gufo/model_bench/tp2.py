"""Gufo on one host against Gufo on two hosts over RDMA (TP2).

TP2 runs of the published tables store their artifacts beside the one-host
ones, under the target `gufo-tp2` (Q4) or `gufo-tp2-q8`. These tables compare
them with a one-host run of the same build (`gufo-onehost`), or with the
published one-host artifacts where that run is missing; the Q8 table has no
one-host counterpart, because the Q8 model does not fit one host.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .artifacts import artifact_path, load_artifact
from .config import BenchConfig, TableSpec
from .render import TODO, _depth_label, _fmt, _fmt_stat, _number, _serving_rate, gain, model_label

TP2 = "gufo-tp2"
TP2_Q8 = "gufo-tp2-q8"
ONE_HOST = "gufo-onehost"
TABLES = ("single-ar-tp2", "single-mtp-tp2", "multi-ar-tp2", "multi-mtp-tp2", "tp2-q8")
Q8_LABEL = "Flash-Next Q8 TP2"


def _one_host(config: BenchConfig, table: TableSpec, mode: str | None = None) -> dict[str, Any] | None:
    """The same build on one host, else the published one-host artifact."""
    same_build = load_artifact(artifact_path(config, table, ONE_HOST, mode))
    return same_build or load_artifact(artifact_path(config, table, "gufo", mode))


def _stat(report: dict[str, Any] | None, depth: int, key: str) -> str | None:
    return _fmt_stat((report or {}).get("rows", {}).get(str(depth)), key)


def _rate(report: dict[str, Any] | None, users: int) -> str | None:
    value = _serving_rate(report, users)
    return None if value is None else _fmt(value)


def _best(values: list[str | None]) -> str | None:
    return max((v for v in values if _number(v) is not None), key=_number, default=None)


def _pair(one: str | None, two: str | None) -> list[str]:
    one, two = one or TODO, two or TODO
    return [one, two, gain(_number(two), _number(one), "higher")]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---:"] * len(header)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def render_table(config: BenchConfig, table_id: str) -> str:
    if table_id == "single-ar-tp2":
        base = config.table("single-ar")
        one, two = _one_host(config, base), load_artifact(artifact_path(config, base, TP2))
        header = [f"{model_label(config, base)}<br>Depth (tokens)",
                  "Gufo pp (tok/s)", "Gufo TP2 pp (tok/s)", "Gain",
                  "Gufo tg (tok/s)", "Gufo TP2 tg (tok/s)", "Gain"]
        rows = [[_depth_label(d), *_pair(_stat(one, d, "pp"), _stat(two, d, "pp")),
                 *_pair(_stat(one, d, "tg"), _stat(two, d, "tg"))] for d in base.spec["depths"]]
        return _table(header, rows)
    if table_id == "single-mtp-tp2":
        base = config.table("single-mtp")
        workloads = base.workload_tables()
        reports = [(_one_host(config, w), load_artifact(artifact_path(config, w, TP2)))
                   for w in workloads]
        header = [f"{model_label(config, base)}<br>Depth (tokens)",
                  "Gufo pp (tok/s)", "Gufo TP2 pp (tok/s)", "Gain pp"]
        for workload in workloads:
            label = workload.spec["label"]
            header += [f"Gufo tg {label} (tok/s)", f"Gufo TP2 tg {label} (tok/s)", f"Gain {label}"]
        rows = []
        for depth in base.spec["depths"]:
            row = [_depth_label(depth),
                   *_pair(_best([_stat(one, depth, "pp") for one, _ in reports]),
                          _best([_stat(two, depth, "pp") for _, two in reports]))]
            for one, two in reports:
                row += _pair(_stat(one, depth, "tg"), _stat(two, depth, "tg"))
            rows.append(row)
        return _table(header, rows)
    if table_id == "multi-ar-tp2":
        base = config.table("multi-ar")
        one, two = _one_host(config, base, "ar"), load_artifact(artifact_path(config, base, TP2, "ar"))
        header = [f"{model_label(config, base)}<br>Users",
                  "Gufo AR (tok/s)", "Gufo TP2 AR (tok/s)", "Gain"]
        rows = [[str(c), *_pair(_rate(one, c), _rate(two, c))] for c in base.spec["concurrency"]]
        return _table(header, rows)
    if table_id == "multi-mtp-tp2":
        base = config.table("multi-mtp")
        mode = config.speculative["mode"]
        workloads = base.workload_tables()
        reports = [(_one_host(config, w, mode), load_artifact(artifact_path(config, w, TP2, mode)))
                   for w in workloads]
        header = [f"{model_label(config, base)}<br>Users"]
        for workload in workloads:
            label = workload.spec["label"]
            header += [f"Gufo {label} (tok/s)", f"Gufo TP2 {label} (tok/s)", "Gain"]
        rows = []
        for users in base.spec["concurrency"]:
            row = [str(users)]
            for one, two in reports:
                row += _pair(_rate(one, users), _rate(two, users))
            rows.append(row)
        return _table(header, rows)
    if table_id == "tp2-q8":
        ar = config.table("single-ar")
        workloads = config.table("single-mtp").workload_tables()
        ar_report = load_artifact(artifact_path(config, ar, TP2_Q8))
        mtp_reports = [load_artifact(artifact_path(config, w, TP2_Q8)) for w in workloads]
        spec = config.speculative["label"]
        header = [f"{Q8_LABEL}<br>Depth (tokens)", "pp (tok/s)", "tg AR (tok/s)"]
        header += [f"tg {spec} {w.spec['label']} (tok/s)" for w in workloads]
        rows = []
        for depth in ar.spec["depths"]:
            cells = [_stat(ar_report, depth, "pp"), _stat(ar_report, depth, "tg")]
            cells += [_stat(report, depth, "tg") for report in mtp_reports]
            rows.append([_depth_label(depth), *(cell or TODO for cell in cells)])
        return _table(header, rows)
    raise SystemExit(f"no TP2 renderer for table {table_id}")


def chart_q8(config: BenchConfig, rows: dict[str, dict[str, str]], path: Path) -> bool:
    """Prefill on the left axis and generation on the right, over depth."""
    from .charts import COLORS, SURFACE, TEXT, TEXT_SECONDARY, _depth_ticks, _has_data, _plt, _series

    labels = [_depth_label(d) for d in config.table("single-ar").spec["depths"]]
    spec = config.speculative["label"]
    workloads = config.table("single-mtp").workload_tables()
    pp = _series(rows, labels, "pp")
    generation = [("tg AR", _series(rows, labels, "tg AR"), COLORS["gufo"])]
    shades = [COLORS["spec"], COLORS["ref_spec"]]
    for workload, color in zip(workloads, shades):
        name = f"tg {spec} {workload.spec['label']}"
        generation.append((name, _series(rows, labels, name), color))
    if not _has_data(pp, *(values for _, values, _ in generation)):
        return False
    plt = _plt()
    fig, left = plt.subplots(figsize=(8, 3.4))
    right = left.twinx()
    right.spines["right"].set_visible(True)
    right.grid(False)
    x = range(len(labels))

    def draw(ax: Any, name: str, values: list[float], color: str, style: str) -> None:
        points = [(i, v) for i, v in zip(x, values) if not math.isnan(v)]
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=2,
                linestyle=style, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1, label=name)

    draw(left, "pp", pp, COLORS["tp2"], "--")
    for name, values, color in generation:
        draw(right, name, values, color, "-")
    for ax, values, unit in ((left, [pp], "prefill tok/s"),
                             (right, [v for _, v, _ in generation], "generation tok/s")):
        top = max((v for s in values for v in s if not math.isnan(v)), default=1.0)
        ax.set_ylim(0, top * 1.2)
        ax.set_ylabel(unit, color=TEXT_SECONDARY)
    left.set_xticks(list(x), _depth_ticks(labels))
    left.set_xlabel("context depth (tokens)")
    left.grid(axis="x", visible=False)
    handles = [h for ax in (left, right) for h in ax.get_legend_handles_labels()[0]]
    names = [n for ax in (left, right) for n in ax.get_legend_handles_labels()[1]]
    left.legend(handles, names, loc="lower left", ncol=2)
    fig.suptitle(f"{Q8_LABEL} · single user", x=0.01, ha="left", fontsize=10, color=TEXT)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    return True
