#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from lerobot.async_inference.system_resources import summarize_samples

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PLOT_GROUPS = [
    ("cpu_memory", ["cpu_percent", "memory_percent", "process_cpu_percent"]),
    ("gpu", ["gpu_util_percent", "gpu_memory_util_percent", "tegra_gr3d_percent"]),
    ("memory_mb", ["memory_used_mb", "gpu_memory_used_mb", "process_rss_mb"]),
    ("power_w", ["gpu_power_w", "tegra_vdd_gpu_w", "tegra_vdd_cpu_soc_mss_w", "tegra_vin_w"]),
    ("temperature_c", ["gpu_temp_c", "tegra_gpu_temp_c"]),
]

METRIC_LABELS = {
    "cpu_percent": "整机 CPU 平均占用率 (%)",
    "memory_percent": "整机内存占用率 (%)",
    "memory_used_mb": "整机已用内存 (MB)",
    "process_cpu_percent": "policy_server 进程 CPU 占用率 (%)",
    "process_rss_mb": "policy_server 进程常驻内存 RSS (MB)",
    "gpu_util_percent": "NVIDIA GPU 计算单元占用率 (%)",
    "gpu_memory_util_percent": "NVIDIA GPU 显存控制器占用率 (%)",
    "gpu_memory_used_mb": "NVIDIA GPU 显存占用 (MB)",
    "gpu_power_w": "nvidia-smi GPU 功耗 (W)",
    "tegra_vdd_gpu_w": "Tegra VDD_GPU 供电功耗 (W)",
    "tegra_vdd_cpu_soc_mss_w": "Tegra CPU/SOC/MSS 供电功耗 (W)",
    "tegra_vin_w": "Tegra 整机输入功耗 VIN (W)",
    "gpu_temp_c": "nvidia-smi GPU 温度 (°C)",
    "tegra_gpu_temp_c": "Tegra GPU 温度 (°C)",
}

GROUP_LABELS = {
    "cpu_memory": "CPU 与占用率时间序列",
    "gpu": "GPU 占用率时间序列",
    "memory_mb": "内存占用时间序列",
    "power_w": "功耗时间序列",
    "temperature_c": "温度时间序列",
}

BAR_GROUP_LABELS = {
    "cpu_percent": "CPU 占用率",
    "memory_mb": "内存占用",
    "gpu_percent": "GPU 占用率",
    "gpu_power_w": "GPU 功耗",
    "system_power_w": "系统功耗",
    "temperature_c": "温度",
}

STAT_LABELS = {
    "average": "平均值",
    "peak": "峰值",
}

PLOT_DESCRIPTIONS = {
    "average_cpu_percent_bar_comparison": "平均 CPU 占用率对比。`整机 CPU` 反映所有系统进程总负载；`policy_server 进程 CPU` 只统计推理服务进程自身的 CPU 时间。",
    "average_memory_mb_bar_comparison": "平均内存占用对比。`整机已用内存` 是系统级内存使用量；`policy_server RSS` 是推理服务进程实际常驻内存。",
    "average_gpu_percent_bar_comparison": "平均 GPU 占用率对比。`GPU 计算单元占用率` 来自 nvidia-smi，反映推理期间 GPU compute 活跃程度。",
    "average_gpu_power_w_bar_comparison": "平均 GPU 功耗对比。`nvidia-smi GPU 功耗` 与 `Tegra VDD_GPU` 来源不同，可共同判断 GPU 供电负载。",
    "average_system_power_w_bar_comparison": "平均系统功耗对比。`CPU/SOC/MSS` 代表核心计算域供电功耗，`VIN` 代表整机输入功耗。",
    "average_temperature_c_bar_comparison": "平均温度对比。用于观察 pi05 推理对 GPU 温度的持续影响。",
    "peak_cpu_percent_bar_comparison": "CPU 峰值对比。峰值更能反映模型加载、首次推理或瞬时调度造成的短时压力。",
    "peak_memory_mb_bar_comparison": "内存峰值对比。用于判断模型加载和推理过程中最高内存需求。",
    "peak_gpu_percent_bar_comparison": "GPU 占用率峰值对比。用于观察推理瞬间 GPU 是否被打满。",
    "peak_gpu_power_w_bar_comparison": "GPU 功耗峰值对比。用于观察 GPU 在推理高峰时的最大供电需求。",
    "peak_system_power_w_bar_comparison": "系统功耗峰值对比。用于观察整机输入功耗和 CPU/SOC 域功耗的最大瞬时负载。",
    "peak_temperature_c_bar_comparison": "温度峰值对比。用于判断运行期间最高温度是否接近散热或降频风险区间。",
    "cpu_memory_comparison": "CPU 与百分比时间序列。每个子图展示空载基线和 pi05 推理运行随时间变化的曲线。",
    "gpu_comparison": "GPU 占用率时间序列。可以看到推理请求到达时 GPU utilization 的尖峰和间歇。",
    "memory_mb_comparison": "内存时间序列。用于观察模型加载后内存爬升、稳定段以及退出前后的变化。",
    "power_w_comparison": "功耗时间序列。用于观察 GPU、CPU/SOC 和整机 VIN 功耗随推理负载的变化。",
    "temperature_c_comparison": "温度时间序列。用于观察持续运行过程中 GPU 温度的上升和稳定趋势。",
}

PLOT_TITLES = {
    "average_cpu_percent_bar_comparison": "平均值 CPU 占用率对比",
    "average_memory_mb_bar_comparison": "平均值内存占用对比",
    "average_gpu_percent_bar_comparison": "平均值 GPU 占用率对比",
    "average_gpu_power_w_bar_comparison": "平均值 GPU 功耗对比",
    "average_system_power_w_bar_comparison": "平均值系统功耗对比",
    "average_temperature_c_bar_comparison": "平均值温度对比",
    "peak_cpu_percent_bar_comparison": "峰值 CPU 占用率对比",
    "peak_memory_mb_bar_comparison": "峰值内存占用对比",
    "peak_gpu_percent_bar_comparison": "峰值 GPU 占用率对比",
    "peak_gpu_power_w_bar_comparison": "峰值 GPU 功耗对比",
    "peak_system_power_w_bar_comparison": "峰值系统功耗对比",
    "peak_temperature_c_bar_comparison": "峰值温度对比",
    "cpu_memory_comparison": "CPU 与占用率时间序列对比",
    "gpu_comparison": "GPU 占用率时间序列对比",
    "memory_mb_comparison": "内存占用时间序列对比",
    "power_w_comparison": "功耗时间序列对比",
    "temperature_c_comparison": "温度时间序列对比",
}

REPORT_METRICS = [
    "cpu_percent",
    "memory_percent",
    "memory_used_mb",
    "process_cpu_percent",
    "process_rss_mb",
    "gpu_util_percent",
    "gpu_memory_util_percent",
    "gpu_memory_used_mb",
    "gpu_power_w",
    "tegra_vdd_gpu_w",
    "tegra_vdd_cpu_soc_mss_w",
    "tegra_vin_w",
    "gpu_temp_c",
    "tegra_gpu_temp_c",
]


BAR_GROUPS = [
    ("cpu_percent", ["cpu_percent", "process_cpu_percent"]),
    ("memory_mb", ["memory_used_mb", "process_rss_mb"]),
    ("gpu_percent", ["gpu_util_percent", "gpu_memory_util_percent"]),
    ("gpu_power_w", ["gpu_power_w", "tegra_vdd_gpu_w"]),
    ("system_power_w", ["tegra_vdd_cpu_soc_mss_w", "tegra_vin_w"]),
    ("temperature_c", ["gpu_temp_c", "tegra_gpu_temp_c"]),
]


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if sample.get("kind") == "system_resource":
                    sample["_source_file"] = str(path)
                    samples.append(sample)
    samples.sort(key=lambda item: item.get("time", 0))
    return samples


def numeric_series(samples: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    points = [
        (float(sample["elapsed_s"]), float(sample[key]))
        for sample in samples
        if isinstance(sample.get("elapsed_s"), int | float) and isinstance(sample.get(key), int | float)
    ]
    if not points:
        return [], []
    x0 = points[0][0]
    return [x - x0 for x, _ in points], [y for _, y in points]


def metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key)


def plot_description(path: Path) -> tuple[str, str]:
    stem = path.stem
    for suffix, description in PLOT_DESCRIPTIONS.items():
        if stem.endswith(suffix):
            return PLOT_TITLES.get(suffix, suffix), description
    return stem, "该图展示空载基线与 pi05 推理运行之间的资源使用差异。"


def plot_groups(samples: list[dict[str, Any]], output_dir: Path, stem: str) -> list[Path]:
    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_name, keys in PLOT_GROUPS:
        available = [key for key in keys if any(isinstance(sample.get(key), int | float) for sample in samples)]
        if not available:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        for key in available:
            x, y = numeric_series(samples, key)
            if x:
                ax.plot(x, y, label=metric_label(key))
        ax.set_title(GROUP_LABELS.get(group_name, group_name))
        ax.set_xlabel("运行时间 (s)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        path = output_dir / f"{stem}_{group_name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output_paths.append(path)
    return output_paths


def plot_comparison_groups(
    baseline_samples: list[dict[str, Any]],
    run_samples: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> list[Path]:
    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_name, keys in PLOT_GROUPS:
        available = [
            key
            for key in keys
            if any(isinstance(sample.get(key), int | float) for sample in baseline_samples + run_samples)
        ]
        if not available:
            continue

        fig, axes = plt.subplots(len(available), 1, figsize=(12, max(3.5, 2.7 * len(available))), sharex=False)
        if len(available) == 1:
            axes = [axes]

        for ax, key in zip(axes, available, strict=True):
            bx, by = numeric_series(baseline_samples, key)
            rx, ry = numeric_series(run_samples, key)
            if bx:
                ax.plot(bx, by, label="空载基线", color="#4c78a8", linewidth=1.8)
            if rx:
                ax.plot(rx, ry, label="pi05 推理运行", color="#f58518", linewidth=1.8)
            ax.set_title(metric_label(key))
            ax.set_xlabel("运行时间 (s)")
            ax.grid(True, alpha=0.25)
            ax.legend()

        fig.suptitle(f"{GROUP_LABELS.get(group_name, group_name)}对比", y=0.995)
        fig.tight_layout()

        path = output_dir / f"{stem}_{group_name}_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output_paths.append(path)
    return output_paths


def plot_comparison_summary_bars(
    baseline_samples: list[dict[str, Any]],
    run_samples: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> list[Path]:
    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics = summarize_samples(baseline_samples).get("metrics", {})
    run_metrics = summarize_samples(run_samples).get("metrics", {})

    for stat_name, field_name in [("average", "avg"), ("peak", "peak")]:
        for group_name, group_metrics in BAR_GROUPS:
            metrics = [
                key
                for key in group_metrics
                if key in baseline_metrics
                and key in run_metrics
                and isinstance(baseline_metrics[key].get(field_name), int | float)
                and isinstance(run_metrics[key].get(field_name), int | float)
            ]
            if not metrics:
                continue

            baseline_values = [baseline_metrics[key][field_name] for key in metrics]
            run_values = [run_metrics[key][field_name] for key in metrics]
            x_positions = list(range(len(metrics)))
            width = 0.38

            fig, ax = plt.subplots(figsize=(max(7, len(metrics) * 2.4), 4.5))
            ax.bar(
                [x - width / 2 for x in x_positions],
                baseline_values,
                width,
                label="空载基线",
                color="#4c78a8",
            )
            ax.bar(
                [x + width / 2 for x in x_positions],
                run_values,
                width,
                label="pi05 推理运行",
                color="#f58518",
            )
            ax.set_title(
                f"{STAT_LABELS.get(stat_name, stat_name)} {BAR_GROUP_LABELS.get(group_name, group_name)}对比"
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels([metric_label(metric) for metric in metrics], rotation=15, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()

            path = output_dir / f"{stem}_{stat_name}_{group_name}_bar_comparison.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            output_paths.append(path)

    return output_paths


def write_report(samples: list[dict[str, Any]], plot_paths: list[Path], report_path: Path) -> None:
    summary = summarize_samples(samples)
    metrics = summary.get("metrics", {})
    lines = [
        "# 系统资源报告",
        "",
        f"- 样本数: {summary.get('count', 0)}",
        f"- 持续时间: {summary.get('duration_s', 0):.2f}s",
        "",
        "## 指标统计",
        "",
        "| 指标 | 原始字段 | 平均值 | 峰值 | 最小值 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for key in REPORT_METRICS:
        metric = metrics.get(key)
        if not metric:
            continue
        lines.append(
            f"| {metric_label(key)} | `{key}` | {metric['avg']:.3f} | {metric['peak']:.3f} | {metric['min']:.3f} |"
        )

    if plot_paths:
        lines.extend(["", "## 图表说明", ""])
        for path in plot_paths:
            title, description = plot_description(path)
            lines.extend([f"### {title}", "", description, ""])
            lines.append(f"![{path.stem}]({path.name})")
            lines.append("")

    report_path.write_text("\n".join(lines) + "\n")


def write_comparison_report(
    baseline_samples: list[dict[str, Any]],
    run_samples: list[dict[str, Any]],
    plot_paths: list[Path],
    report_path: Path,
    baseline_label: str,
    run_label: str,
) -> None:
    baseline_summary = summarize_samples(baseline_samples)
    run_summary = summarize_samples(run_samples)
    baseline_metrics = baseline_summary.get("metrics", {})
    run_metrics = run_summary.get("metrics", {})

    lines = [
        "# 系统资源对比报告",
        "",
        f"- 空载基线: {baseline_label}",
        f"- 推理运行: {run_label}",
        f"- 空载基线样本数: {baseline_summary.get('count', 0)}",
        f"- 推理运行样本数: {run_summary.get('count', 0)}",
        f"- 空载基线持续时间: {baseline_summary.get('duration_s', 0):.2f}s",
        f"- 推理运行持续时间: {run_summary.get('duration_s', 0):.2f}s",
        "",
        "## 指标统计",
        "",
        "| 指标 | 原始字段 | 基线平均值 | 推理平均值 | 平均值增量 | 基线峰值 | 推理峰值 | 峰值增量 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key in REPORT_METRICS:
        baseline_metric = baseline_metrics.get(key)
        run_metric = run_metrics.get(key)
        if not baseline_metric and not run_metric:
            continue
        baseline_avg = baseline_metric["avg"] if baseline_metric else None
        run_avg = run_metric["avg"] if run_metric else None
        baseline_peak = baseline_metric["peak"] if baseline_metric else None
        run_peak = run_metric["peak"] if run_metric else None

        def fmt(value: float | None) -> str:
            return f"{value:.3f}" if value is not None else "n/a"

        avg_delta = run_avg - baseline_avg if run_avg is not None and baseline_avg is not None else None
        peak_delta = run_peak - baseline_peak if run_peak is not None and baseline_peak is not None else None
        lines.append(
            f"| {metric_label(key)} | `{key}` | {fmt(baseline_avg)} | {fmt(run_avg)} | {fmt(avg_delta)} | "
            f"{fmt(baseline_peak)} | {fmt(run_peak)} | {fmt(peak_delta)} |"
        )

    if plot_paths:
        lines.extend(["", "## 图表说明", ""])
        for path in plot_paths:
            title, description = plot_description(path)
            lines.extend([f"### {title}", "", description, ""])
            lines.append(f"![{path.stem}]({path.name})")
            lines.append("")

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot system resource JSONL logs and write a Markdown report.")
    parser.add_argument("inputs", nargs="+", help="System resource JSONL file(s)")
    parser.add_argument("--baseline", nargs="+", help="Optional baseline JSONL file(s) for comparison")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--run-label", default="run")
    parser.add_argument("--output-dir", default="/home/hrx/Projects/lerobot/logs/system_resources_report")
    parser.add_argument("--stem", default="system_resources")
    parser.add_argument("--report-name", default="system_resources_report.md")
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    samples = read_jsonl(input_paths)
    if not samples:
        raise SystemExit("No system resource samples found.")

    output_dir = Path(args.output_dir)
    if args.baseline:
        baseline_paths = [Path(path) for path in args.baseline]
        baseline_samples = read_jsonl(baseline_paths)
        if not baseline_samples:
            raise SystemExit("No baseline system resource samples found.")
        plot_paths = [
            *plot_comparison_summary_bars(baseline_samples, samples, output_dir, args.stem),
            *plot_comparison_groups(baseline_samples, samples, output_dir, args.stem),
        ]
        write_comparison_report(
            baseline_samples,
            samples,
            plot_paths,
            output_dir / args.report_name,
            args.baseline_label,
            args.run_label,
        )
    else:
        plot_paths = plot_groups(samples, output_dir, args.stem)
        write_report(samples, plot_paths, output_dir / args.report_name)
    print(output_dir / args.report_name)


if __name__ == "__main__":
    main()
