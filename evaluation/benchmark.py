from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch


class Benchmarker:
    """Profiles inference time and memory for AQUILA-NET models."""

    def __init__(self, n_runs: int = 3) -> None:
        self.n_runs = n_runs

    def benchmark_inference_time(
        self,
        model: torch.nn.Module,
        sample_input,
    ) -> dict:
        """Measure forward-pass latency over *n_runs* repetitions.

        A warm-up run is performed before timing to avoid cold-start bias.

        Args:
            model: PyTorch module to profile.
            sample_input: Input passed to ``model.forward``; typically a
                          torch_geometric Data object.

        Returns:
            Dict with keys 'mean_ms', 'std_ms', 'min_ms', 'max_ms'.
        """
        model.eval()
        device = next(model.parameters(), torch.tensor(0.0)).device

        if hasattr(sample_input, "to"):
            sample_input = sample_input.to(device)

        # warm-up
        with torch.no_grad():
            model(sample_input)

        times: list[float] = []
        with torch.no_grad():
            for _ in range(self.n_runs):
                start = time.perf_counter()
                model(sample_input)
                end = time.perf_counter()
                times.append((end - start) * 1000.0)

        times_arr = np.array(times)
        return {
            "mean_ms": float(times_arr.mean()),
            "std_ms": float(times_arr.std()),
            "min_ms": float(times_arr.min()),
            "max_ms": float(times_arr.max()),
        }

    def benchmark_memory_usage(
        self,
        model: torch.nn.Module,
        sample_input,
    ) -> dict:
        """Estimate model parameter count and memory footprint.

        Memory is estimated as parameter count × 4 bytes (float32), which
        gives a conservative lower bound (activations are excluded).

        Args:
            model: PyTorch module to profile.
            sample_input: Unused; kept for API consistency.

        Returns:
            Dict with keys 'n_parameters', 'param_memory_mb',
            'trainable_parameters'.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        param_memory_mb = total_params * 4 / (1024 ** 2)

        return {
            "n_parameters": total_params,
            "trainable_parameters": trainable_params,
            "param_memory_mb": round(param_memory_mb, 3),
        }

    def run_full_benchmark(
        self,
        model: torch.nn.Module,
        data_loader,
    ) -> pd.DataFrame:
        """Run inference-time and memory benchmarks over the first batch.

        Args:
            model: Model to benchmark.
            data_loader: Iterable; the first batch is used as sample input.

        Returns:
            Single-row DataFrame with columns: mean_ms, std_ms, min_ms,
            max_ms, n_parameters, trainable_parameters, param_memory_mb.
        """
        sample_input = next(iter(data_loader))

        time_results = self.benchmark_inference_time(model, sample_input)
        mem_results = self.benchmark_memory_usage(model, sample_input)

        combined = {**time_results, **mem_results}
        return pd.DataFrame([combined])
