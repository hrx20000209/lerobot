#!/usr/bin/env python

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from queue import Queue
from typing import Any


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "[N/A]":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat") as f:
            fields = f.readline().split()[1:]
    except OSError:
        return None
    values = [int(field) for field in fields]
    idle = values[3] + values[4]
    total = sum(values)
    return total, idle


def _cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return 100.0 * (1.0 - idle_delta / total_delta)


def _read_meminfo() -> dict[str, float | None]:
    values: dict[str, float] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    values[parts[0].rstrip(":")] = float(parts[1])
    except OSError:
        return {
            "memory_total_mb": None,
            "memory_available_mb": None,
            "memory_used_mb": None,
            "memory_percent": None,
        }

    total_mb = values.get("MemTotal", 0.0) / 1024.0
    available_mb = values.get("MemAvailable", 0.0) / 1024.0
    used_mb = total_mb - available_mb if total_mb else None
    percent = 100.0 * used_mb / total_mb if total_mb and used_mb is not None else None
    return {
        "memory_total_mb": total_mb,
        "memory_available_mb": available_mb,
        "memory_used_mb": used_mb,
        "memory_percent": percent,
    }


def _read_process_stats(
    previous: tuple[float, float] | None, elapsed_s: float | None, pid: int | None = None
) -> tuple[dict[str, float | None], tuple[float, float] | None]:
    pid = pid or os.getpid()
    stat_path = f"/proc/{pid}/stat"
    status_path = f"/proc/{pid}/status"
    result: dict[str, float | None] = {
        "process_cpu_percent": None,
        "process_rss_mb": None,
    }
    current_cpu: tuple[float, float] | None = None

    try:
        with open(stat_path) as f:
            stat = f.read().split()
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        utime = float(stat[13]) / ticks_per_second
        stime = float(stat[14]) / ticks_per_second
        current_cpu = (utime, stime)
        if previous is not None and elapsed_s and elapsed_s > 0:
            result["process_cpu_percent"] = 100.0 * (
                (current_cpu[0] + current_cpu[1]) - (previous[0] + previous[1])
            ) / elapsed_s
    except (OSError, IndexError, KeyError, ValueError):
        current_cpu = None

    try:
        with open(status_path) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    result["process_rss_mb"] = float(line.split()[1]) / 1024.0
                    break
    except (OSError, IndexError, ValueError):
        pass

    return result, current_cpu


def _parse_tegrastats(line: str | None) -> dict[str, float | str | None]:
    if not line:
        return {}

    result: dict[str, float | str | None] = {"tegrastats_line": line.strip()}

    ram = re.search(r"\bRAM\s+(\d+)/(\d+)MB", line)
    if ram:
        used = float(ram.group(1))
        total = float(ram.group(2))
        result["tegra_ram_used_mb"] = used
        result["tegra_ram_total_mb"] = total
        result["tegra_ram_percent"] = 100.0 * used / total if total else None

    cpu = re.search(r"\bCPU\s+\[([^\]]+)\]", line)
    if cpu:
        usages = [float(match) for match in re.findall(r"(\d+(?:\.\d+)?)%@", cpu.group(1))]
        if usages:
            result["tegra_cpu_percent"] = sum(usages) / len(usages)
            result["tegra_cpu_peak_core_percent"] = max(usages)

    gr3d = re.search(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
    if gr3d:
        result["tegra_gr3d_percent"] = float(gr3d.group(1))

    gpu_temp = re.search(r"\bgpu@(-?\d+(?:\.\d+)?)C", line)
    if gpu_temp:
        result["tegra_gpu_temp_c"] = float(gpu_temp.group(1))

    for rail in ("VDD_GPU", "VDD_CPU_SOC_MSS", "VIN_SYS_5V0", "VIN"):
        match = re.search(rf"\b{rail}\s+(\d+(?:\.\d+)?)mW(?:/(\d+(?:\.\d+)?)mW)?", line)
        if match:
            key = rail.lower()
            result[f"tegra_{key}_w"] = float(match.group(1)) / 1000.0
            if match.group(2):
                result[f"tegra_{key}_avg_w"] = float(match.group(2)) / 1000.0

    return result


def _sample_nvidia_smi() -> dict[str, float | str | None]:
    if shutil.which("nvidia-smi") is None:
        return {}

    query = (
        "timestamp,index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,power.draw,power.limit,temperature.gpu"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    if completed.returncode != 0 or not completed.stdout.strip():
        return {}

    rows = list(csv.reader(completed.stdout.splitlines()))
    if not rows:
        return {}

    row = [item.strip() for item in rows[0]]
    if len(row) < 10:
        return {}

    result: dict[str, float | str | None] = {
        "gpu_timestamp": row[0],
        "gpu_index": _safe_float(row[1]),
        "gpu_name": row[2],
        "gpu_util_percent": _safe_float(row[3]),
        "gpu_memory_util_percent": _safe_float(row[4]),
        "gpu_memory_used_mb": _safe_float(row[5]),
        "gpu_memory_total_mb": _safe_float(row[6]),
        "gpu_power_w": _safe_float(row[7]),
        "gpu_power_limit_w": _safe_float(row[8]),
        "gpu_temp_c": _safe_float(row[9]),
    }
    return result


def summarize_samples(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples = list(samples)
    summary: dict[str, Any] = {"count": len(samples), "metrics": {}}
    if not samples:
        return summary

    start = samples[0].get("time")
    end = samples[-1].get("time")
    summary["start_time"] = start
    summary["end_time"] = end
    if isinstance(start, int | float) and isinstance(end, int | float):
        summary["duration_s"] = end - start

    numeric_keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if isinstance(value, int | float) and key not in {"time", "elapsed_s"}
        }
    )
    for key in numeric_keys:
        values = [float(sample[key]) for sample in samples if isinstance(sample.get(key), int | float)]
        if not values:
            continue
        summary["metrics"][key] = {
            "avg": sum(values) / len(values),
            "peak": max(values),
            "min": min(values),
        }

    return summary


class SystemResourceRecorder:
    def __init__(
        self,
        name: str,
        log_dir: str | Path,
        interval_s: float = 1.0,
        enabled: bool = False,
        sample_nvidia_smi: bool = False,
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.interval_s = max(0.1, float(interval_s))
        self.enabled = enabled
        self.sample_nvidia_smi = sample_nvidia_smi
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{name}_system_resources_{self.run_id}.jsonl"
        self.summary_path = self.log_dir / f"{name}_system_resources_summary_{self.run_id}.json"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tegra_thread: threading.Thread | None = None
        self._tegra_proc: subprocess.Popen[str] | None = None
        self._tegra_lines: Queue[str] = Queue(maxsize=1)
        self._samples: list[dict[str, Any]] = []

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._start_tegrastats()
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-system-resources", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s + 1.0))
        self._stop_tegrastats()
        summary = summarize_samples(self._samples)
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    def _start_tegrastats(self) -> None:
        if shutil.which("tegrastats") is None:
            return
        try:
            self._tegra_proc = subprocess.Popen(
                ["tegrastats", "--interval", str(int(self.interval_s * 1000))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self._tegra_proc = None
            return

        self._tegra_thread = threading.Thread(target=self._read_tegrastats, daemon=True)
        self._tegra_thread.start()

    def _stop_tegrastats(self) -> None:
        if self._tegra_proc is not None and self._tegra_proc.poll() is None:
            self._tegra_proc.terminate()
            try:
                self._tegra_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._tegra_proc.kill()
        if self._tegra_thread is not None:
            self._tegra_thread.join(timeout=1)

    def _read_tegrastats(self) -> None:
        if self._tegra_proc is None or self._tegra_proc.stdout is None:
            return
        for line in self._tegra_proc.stdout:
            if self._stop_event.is_set():
                break
            if self._tegra_lines.full():
                try:
                    self._tegra_lines.get_nowait()
                except Exception:
                    pass
            self._tegra_lines.put(line)

    def _latest_tegrastats_line(self) -> str | None:
        line = None
        while not self._tegra_lines.empty():
            line = self._tegra_lines.get_nowait()
        return line

    def _run(self) -> None:
        previous_cpu = _read_cpu_times()
        previous_process_cpu: tuple[float, float] | None = None
        previous_time = time.time()
        start_time = previous_time
        _, previous_process_cpu = _read_process_stats(None, None)

        with self.log_path.open("a", buffering=1) as f:
            if self._stop_event.wait(self.interval_s):
                return
            while not self._stop_event.is_set():
                sample_time = time.time()
                elapsed = sample_time - previous_time
                current_cpu = _read_cpu_times()
                process_stats, current_process_cpu = _read_process_stats(previous_process_cpu, elapsed)

                sample: dict[str, Any] = {
                    "time": sample_time,
                    "elapsed_s": sample_time - start_time,
                    "kind": "system_resource",
                    "recorder": self.name,
                    "cpu_percent": _cpu_percent(previous_cpu, current_cpu),
                    **_read_meminfo(),
                    **process_stats,
                    **_parse_tegrastats(self._latest_tegrastats_line()),
                    **(_sample_nvidia_smi() if self.sample_nvidia_smi else {}),
                }
                f.write(json.dumps(sample, sort_keys=True) + "\n")
                self._samples.append(sample)

                previous_cpu = current_cpu
                previous_process_cpu = current_process_cpu
                previous_time = sample_time
                self._stop_event.wait(self.interval_s)


def collect_for_duration(
    name: str,
    log_dir: str,
    duration_s: float,
    interval_s: float,
    sample_nvidia_smi: bool = False,
) -> Path:
    recorder = SystemResourceRecorder(
        name=name,
        log_dir=log_dir,
        interval_s=interval_s,
        enabled=True,
        sample_nvidia_smi=sample_nvidia_smi,
    )
    recorder.start()
    try:
        time.sleep(duration_s)
    finally:
        recorder.stop()
    return recorder.log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record system resource usage to JSONL.")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--log-dir", default="logs/system_resources")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument(
        "--sample-nvidia-smi",
        action="store_true",
        help="Also call nvidia-smi every sample. Avoid this inside active gRPC processes.",
    )
    args = parser.parse_args()
    log_path = collect_for_duration(
        args.name,
        args.log_dir,
        args.duration_s,
        args.interval_s,
        sample_nvidia_smi=args.sample_nvidia_smi,
    )
    print(log_path)


if __name__ == "__main__":
    main()
