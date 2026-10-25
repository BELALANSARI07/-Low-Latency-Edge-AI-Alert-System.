"""
Empirical Telemetry Collection & Edge Hardware Diagnostics Engine.
Measures real frame-to-frame intervals, multi-stage inference latencies (P50, P95, P99),
and host hardware utilization without fabricated benchmarks.
"""

import time
import os
from collections import deque
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


class TelemetryCollector:
    """
    Collects real-time edge telemetry and hardware load metrics.
    All reported benchmarks are strictly measured from runtime operations.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.frame_times: deque = deque(maxlen=window_size)
        self.total_latencies: deque = deque(maxlen=window_size)
        self.preprocess_latencies: deque = deque(maxlen=window_size)
        self.inference_latencies: deque = deque(maxlen=window_size)
        self.postprocess_latencies: deque = deque(maxlen=window_size)

        self.total_frames = 0
        self.alerts_count = 0
        self.start_time = time.time()

    def record_frame(
        self,
        total_latency_ms: float,
        preprocess_ms: Optional[float] = None,
        inference_ms: Optional[float] = None,
        postprocess_ms: Optional[float] = None,
    ):
        """Records a processed frame timestamp and multi-stage execution latencies."""
        now = time.perf_counter()
        self.frame_times.append(now)
        self.total_latencies.append(float(total_latency_ms))

        if preprocess_ms is not None:
            self.preprocess_latencies.append(float(preprocess_ms))
        if inference_ms is not None:
            self.inference_latencies.append(float(inference_ms))
        if postprocess_ms is not None:
            self.postprocess_latencies.append(float(postprocess_ms))

        self.total_frames += 1

    def record_alert(self):
        """Increments alerts triggered counter."""
        self.alerts_count += 1

    def get_current_fps(self) -> float:
        """
        Calculates instantaneous rolling frames per second over the active window.
        Returns 0.0 if insufficient frames have been processed.
        """
        if len(self.frame_times) < 2:
            return 0.0
        delta = self.frame_times[-1] - self.frame_times[0]
        if delta <= 0:
            return 0.0
        return round((len(self.frame_times) - 1) / delta, 1)

    def _compute_percentiles(self, d: deque) -> Dict[str, Optional[float]]:
        """Computes mean, P50, P95, and P99 percentiles for a metric deque."""
        if not d:
            return {
                "avg_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "min_ms": None,
                "max_ms": None,
            }
        arr = np.array(list(d))
        return {
            "avg_ms": round(float(np.mean(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
        }

    def _get_gpu_telemetry(self) -> Dict[str, Any]:
        """
        Queries actual GPU/VRAM telemetry if hardware sensors exist,
        otherwise reports null without inventing mock values.
        """
        # Jetson tegrastats / sysfs check
        jetson_cur_freq = "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq"
        if os.path.exists(jetson_cur_freq):
            try:
                with open(jetson_cur_freq, "r") as f:
                    freq = int(f.read().strip())
                return {
                    "available": True,
                    "gpu_freq_hz": freq,
                    "vram_used_mb": None,
                    "power_draw_w": None
                }
            except Exception:
                pass

        # Standard system without hardware GPU instrumentation
        return {
            "available": False,
            "vram_used": None,
            "power_draw_w": None
        }

    def get_hardware_telemetry(self) -> Dict[str, Any]:
        """Queries CPU, memory, and measured latency statistics."""
        if psutil is not None:
            cpu_pct = round(psutil.cpu_percent(interval=None), 1)
            mem = psutil.virtual_memory()
            ram_used_mb = round(mem.used / (1024 * 1024), 1)
            ram_pct = round(mem.percent, 1)
        else:
            cpu_pct = None
            ram_used_mb = None
            ram_pct = None

        latency_stats = self._compute_percentiles(self.total_latencies)
        gpu_stats = self._get_gpu_telemetry()

        return {
            "fps": self.get_current_fps(),
            "cpu_percent": cpu_pct,
            "ram_used_mb": ram_used_mb,
            "ram_percent": ram_pct,
            "vram_used": gpu_stats.get("vram_used"),
            "power_draw_w": gpu_stats.get("power_draw_w"),
            "gpu_telemetry_available": gpu_stats["available"],
            "latency_avg_ms": latency_stats["avg_ms"],
            "latency_p50_ms": latency_stats["p50_ms"],
            "latency_p95_ms": latency_stats["p95_ms"],
            "latency_p99_ms": latency_stats["p99_ms"],
            "latency_min_ms": latency_stats["min_ms"],
            "latency_max_ms": latency_stats["max_ms"],
            "stage_latencies": {
                "preprocess": self._compute_percentiles(self.preprocess_latencies),
                "inference": self._compute_percentiles(self.inference_latencies),
                "postprocess": self._compute_percentiles(self.postprocess_latencies),
            },
            "total_frames_processed": self.total_frames,
            "total_alerts_count": self.alerts_count,
            "uptime_seconds": int(time.time() - self.start_time),
        }
