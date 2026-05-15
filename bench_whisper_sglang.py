#!/usr/bin/env python3
"""
Whisper Model Benchmarking Script - Offline SGLang
==================================================

Pure offline inference benchmarking using SGLang's Engine class.
No server, no network latency - just pure model processing time.

Usage:

    SGLANG_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper_sglang.py \
        --model openai/whisper-medium \
        --data-path ./dataset_100_30sec \
        --batch-sizes 4,8,16 \
        --output-dir ./results \
        --cores "0-31|32-63" \
        --tp 2 \
        --num-iterations 100 \
        --warmup-iterations 3 \
        --target-duration 30.0 \
        --target-sample-rate 16000

    SGLANG_CPU_OMP_THREADS_BIND='0-63|64-127' python bench_whisper_sglang.py \
        --model openai/whisper-medium \
        --data-path ./dataset_100_30sec \
        --batch-sizes 8,16,32 \
        --output-dir ./results_sglang_tp2_128 \
        --cores "0-63|64-127" \
        --tp 2 \
        --num-iterations 100 \
        --warmup-iterations 3 \
        --target-duration 30.0 \
        --target-sample-rate 16000

    python bench_whisper_sglang.py \
        --model openai/whisper-medium \
        --data-path ./dataset_100_30sec \
        --batch-sizes 4,8,16 \
        --output-dir ./results_sglang_tp4_128 \
        --cores "0-31|32-63|64-95|96-127" \
        --tp 4 \
        --num-iterations 100 \
        --warmup-iterations 3 \
        --target-duration 30.0 \
        --target-sample-rate 16000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import statistics
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas openpyxl")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed. Run: pip install librosa")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed. Run: pip install soundfile")
    sys.exit(1)

# SGLang imports - will be checked at runtime
SGLANG_AVAILABLE = False
try:
    import sglang as sgl
    from sglang import Engine
    SGLANG_AVAILABLE = True
except ImportError:
    pass


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""
    model: str
    data_path: str
    batch_sizes: List[int]
    output_dir: str
    cores: str
    tp: int
    num_iterations: int
    warmup_iterations: int
    target_duration: float
    target_sample_rate: int
    seed: int
    device: str
    dtype: str
    trust_remote_code: bool
    profile: bool = False
    profile_dir: str = "sglang_profile"


@dataclass
class IterationResult:
    """Result of a single inference iteration."""
    batch_size: int
    iteration_id: int
    processing_time_ms: float
    num_tokens: int
    num_audios: int
    success: bool
    error: Optional[str] = None


@dataclass
class BatchMetrics:
    """Aggregated metrics for a batch size."""
    batch_size: int
    iterations_run: int
    mean_processing_time_ms: float
    median_processing_time_ms: float
    p95_time_ms: float
    std_dev_ms: float
    avg_tokens_per_batch: float
    tokens_per_second: float
    time_per_token_ms: float
    audios_per_second: float
    total_audios_processed: int


class WhisperOfflineBenchmark:
    """Offline benchmarking class for Whisper with SGLang."""

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm', '.opus'}

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.audio_file_cache: List[str] = []  # list of audio file paths
        self.results: List[IterationResult] = []
        self.engine: Optional[Engine] = None

        # Set random seed for reproducibility
        np.random.seed(config.seed)

    def setup_cpu_environment(self):
        """Setup CPU-specific environment variables for SGLang."""
        if self.config.device == "cpu" and self.config.cores:
            os.environ["SGLANG_CPU_OMP_THREADS_BIND"] = self.config.cores
            print("CPU environment configured:")
            print(f"  SGLANG_CPU_OMP_THREADS_BIND: {self.config.cores}")

    def init_model(self):
        """Initialize the SGLang Engine for offline inference."""
        if not SGLANG_AVAILABLE:
            print("ERROR: sglang not installed. Run: pip install sglang")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("INITIALIZING MODEL")
        print("=" * 60)
        print(f"Model: {self.config.model}")
        print(f"Device: {self.config.device}")
        print(f"Tensor Parallel: {self.config.tp}")
        print(f"Dtype: {self.config.dtype}")

        self.setup_cpu_environment()

        try:
            self.engine = Engine(
                model_path=self.config.model,
                tp_size=self.config.tp,
                dtype=self.config.dtype,
                device=self.config.device,
                trust_remote_code=self.config.trust_remote_code,
                disable_overlap_schedule=True,
                log_level="warning",
                chunked_prefill_size=-1,
            )
            print("Model loaded successfully!")

        except Exception as e:
            print(f"ERROR initializing model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def discover_audio_files(self) -> List[str]:
        """Find all supported audio files in the data path."""
        data_path = Path(self.config.data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            audio_files.extend(data_path.glob(f"**/*{ext}"))
            audio_files.extend(data_path.glob(f"**/*{ext.upper()}"))

        return sorted([str(f) for f in audio_files])

    def preprocess_audio_files(self) -> int:
        """Discover and validate audio files, caching their paths."""
        print("\n" + "=" * 60)
        print("PRE-PROCESSING AUDIO FILES")
        print("=" * 60)

        audio_files = self.discover_audio_files()
        print(f"Found {len(audio_files)} audio files")
        print(f"Target duration: {self.config.target_duration} seconds")
        print(f"Target sample rate: {self.config.target_sample_rate} Hz")

        valid_count = 0
        skipped_count = 0

        for i, audio_path in enumerate(audio_files):
            try:
                # Validate file is loadable and check duration
                info = sf.info(audio_path)
                duration = info.duration
                if duration > 0:
                    self.audio_file_cache.append(audio_path)
                    valid_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                skipped_count += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files...")

        print(f"\nValid audio files: {valid_count}")
        print(f"Skipped files: {skipped_count}")

        if valid_count == 0:
            raise ValueError("No valid audio files found!")

        return valid_count

    def run_inference_batch(
        self,
        audio_paths: List[str]
    ) -> Tuple[float, int, bool, Optional[str]]:
        """
        Run offline inference on a batch of audio files.
        Returns: (processing_time_ms, num_tokens, success, error_message)
        """
        try:
            batch_size = len(audio_paths)

            # SGLang Engine.generate accepts batch prompts and batch audio_data
            prompts = [""] * batch_size
            sampling_params = {
                "temperature": 0.0,
                "max_new_tokens": 448,
            }

            # Measure pure processing time
            start_time = time.perf_counter()
            outputs = self.engine.generate(
                prompt=prompts,
                audio_data=audio_paths,
                sampling_params=sampling_params,
            )
            end_time = time.perf_counter()

            processing_time_ms = (end_time - start_time) * 1000

            # Count total tokens generated
            total_tokens = sum(
                o["meta_info"]["completion_tokens"] for o in outputs
            )

            return processing_time_ms, total_tokens, True, None

        except Exception as e:
            import traceback
            traceback.print_exc()
            return 0.0, 0, False, str(e)

    def run_warmup(self) -> bool:
        """Run warmup iterations to prime JIT, caches, etc."""
        print("\n" + "=" * 60)
        print("WARM-UP PHASE")
        print("=" * 60)
        print(f"Running {self.config.warmup_iterations} warmup iterations...")

        for i in range(self.config.warmup_iterations):
            idx = np.random.randint(0, len(self.audio_file_cache))
            audio_path = self.audio_file_cache[idx]

            proc_time, tokens, success, error = self.run_inference_batch([audio_path])

            if success:
                print(f"  Warmup {i + 1}/{self.config.warmup_iterations}: {proc_time:.2f}ms, {tokens} tokens")
            else:
                print(f"  Warmup {i + 1}/{self.config.warmup_iterations}: FAILED - {error}")
                if i == 0:
                    print("\nWARNING: First warmup failed!")
                    return False

        print("Warmup complete. Discarding warmup results.")
        return True

    def run_batch_benchmark(self, batch_size: int) -> BatchMetrics:
        """Run benchmark for a specific batch size."""
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING BATCH SIZE: {batch_size}")
        print(f"{'=' * 60}")

        processing_times: List[float] = []
        tokens_list: List[int] = []
        success_count = 0
        fail_count = 0
        total_audios = 0

        num_iterations = self.config.num_iterations

        for iter_idx in range(num_iterations):
            # Select batch_size random audios
            indices = np.random.choice(
                len(self.audio_file_cache),
                size=min(batch_size, len(self.audio_file_cache)),
                replace=True
            )
            batch_paths = [self.audio_file_cache[i] for i in indices]

            proc_time, tokens, success, error = self.run_inference_batch(batch_paths)

            result = IterationResult(
                batch_size=batch_size,
                iteration_id=iter_idx,
                processing_time_ms=proc_time,
                num_tokens=tokens,
                num_audios=len(batch_paths),
                success=success,
                error=error
            )
            self.results.append(result)

            if success:
                processing_times.append(proc_time)
                tokens_list.append(tokens)
                success_count += 1
                total_audios += len(batch_paths)
            else:
                fail_count += 1

            # Progress update every 10 iterations
            if (iter_idx + 1) % 10 == 0:
                print(f"  Completed {iter_idx + 1}/{num_iterations} iterations...")

        # Calculate metrics
        if len(processing_times) == 0:
            print(f"  WARNING: All {num_iterations} iterations failed!")
            return BatchMetrics(
                batch_size=batch_size,
                iterations_run=num_iterations,
                mean_processing_time_ms=0.0,
                median_processing_time_ms=0.0,
                p95_time_ms=0.0,
                std_dev_ms=0.0,
                avg_tokens_per_batch=0.0,
                tokens_per_second=0.0,
                time_per_token_ms=0.0,
                audios_per_second=0.0,
                total_audios_processed=0
            )

        mean_time = statistics.mean(processing_times)
        median_time = statistics.median(processing_times)
        p95_time = np.percentile(processing_times, 95)
        std_dev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0

        total_tokens = sum(tokens_list)
        avg_tokens = total_tokens / len(tokens_list) if tokens_list else 0
        total_time_sec = sum(processing_times) / 1000.0
        tokens_per_sec = total_tokens / total_time_sec if total_time_sec > 0 else 0
        time_per_token = (total_time_sec * 1000) / total_tokens if total_tokens > 0 else 0
        audios_per_sec = total_audios / total_time_sec if total_time_sec > 0 else 0

        metrics = BatchMetrics(
            batch_size=batch_size,
            iterations_run=num_iterations,
            mean_processing_time_ms=round(mean_time, 3),
            median_processing_time_ms=round(median_time, 3),
            p95_time_ms=round(p95_time, 3),
            std_dev_ms=round(std_dev, 3),
            avg_tokens_per_batch=round(avg_tokens, 2),
            tokens_per_second=round(tokens_per_sec, 2),
            time_per_token_ms=round(time_per_token, 3),
            audios_per_second=round(audios_per_sec, 4),
            total_audios_processed=total_audios
        )

        print(f"\n  Batch Size {batch_size} Results:")
        print(f"    Successful: {success_count}, Failed: {fail_count}")
        print(f"    Mean Processing Time: {metrics.mean_processing_time_ms:.2f} ms")
        print(f"    Median Processing Time: {metrics.median_processing_time_ms:.2f} ms")
        print(f"    P95 Processing Time: {metrics.p95_time_ms:.2f} ms")
        print(f"    Audios/Second: {metrics.audios_per_second:.4f}")
        print(f"    Tokens/Second: {metrics.tokens_per_second:.2f}")

        return metrics

    def run_full_benchmark(self) -> List[BatchMetrics]:
        """Run the complete benchmark across all batch sizes."""
        print("\n" + "=" * 60)
        print("WHISPER OFFLINE BENCHMARK - SGLang")
        print("=" * 60)
        print(f"Model: {self.config.model}")
        print(f"Device: {self.config.device}")
        print(f"Batch Sizes: {self.config.batch_sizes}")
        print(f"Iterations per batch: {self.config.num_iterations}")
        print(f"Cores: {self.config.cores or 'auto'}")
        print(f"Tensor Parallel: {self.config.tp}")

        # Initialize model
        self.init_model()

        # Preprocess audio
        num_valid = self.preprocess_audio_files()
        print(f"Loaded {num_valid} valid audio files into cache")

        # Warmup
        if not self.run_warmup():
            print("\nERROR: Warmup failed. Aborting benchmark.")
            sys.exit(1)

        # Run benchmarks for each batch size
        all_metrics: List[BatchMetrics] = []

        for batch_size in sorted(self.config.batch_sizes):
            metrics = self.run_batch_benchmark(batch_size)
            all_metrics.append(metrics)

        return all_metrics

    def run_profiling(self) -> List[BatchMetrics]:
        """Run benchmark with torch profiler via SGLang's built-in profiling."""
        print("\n" + "=" * 60)
        print("WHISPER OFFLINE BENCHMARK - SGLang (PROFILING)")
        print("=" * 60)
        print(f"Model: {self.config.model}")
        print(f"Device: {self.config.device}")
        print(f"Batch Sizes: {self.config.batch_sizes}")
        print(f"Iterations per batch: {self.config.num_iterations}")
        print(f"Cores: {self.config.cores or 'auto'}")
        print(f"Tensor Parallel: {self.config.tp}")
        print(f"Profile Dir: {self.config.profile_dir}")

        # Set profiler env var
        os.environ["SGLANG_TORCH_PROFILER_DIR"] = self.config.profile_dir
        os.makedirs(self.config.profile_dir, exist_ok=True)

        # Initialize model
        self.init_model()

        # Preprocess audio
        num_valid = self.preprocess_audio_files()
        print(f"Loaded {num_valid} valid audio files into cache")

        # Warmup
        if not self.run_warmup():
            print("\nERROR: Warmup failed. Aborting benchmark.")
            sys.exit(1)

        # Start profiler
        print("\nProfiling enabled - starting profiler...")
        known_files = set(os.listdir(self.config.profile_dir))
        self.engine.start_profile()

        # Run benchmarks for each batch size
        all_metrics: List[BatchMetrics] = []

        for batch_size in sorted(self.config.batch_sizes):
            metrics = self.run_batch_benchmark(batch_size)
            all_metrics.append(metrics)

        # Stop profiler and wait for trace file
        self.engine.stop_profile()
        print("\nProfiler stopped. Waiting for trace file...")
        self._monitor_trace_file(known_files, self.config.profile_dir)

        return all_metrics

    @staticmethod
    def _monitor_trace_file(known_files, directory, interval=1):
        """Monitor directory for new trace files from profiler."""
        print(f"Monitoring {directory} for new trace files...")
        while True:
            flag = False
            time.sleep(interval)
            current_files = set(os.listdir(directory))
            new_files = current_files - known_files
            for new_file in new_files:
                new_file_path = os.path.join(directory, new_file)
                print(f"New file detected: {new_file}")
                previous_size = 0
                while True:
                    try:
                        current_size = os.path.getsize(new_file_path)
                    except FileNotFoundError:
                        print(f"File {new_file} is no longer accessible.")
                        break
                    if current_size > previous_size:
                        previous_size = current_size
                    else:
                        flag = True
                        break
                    time.sleep(interval)
            if flag:
                break

    def save_results(self, metrics: List[BatchMetrics]) -> Tuple[str, str, str]:
        """Save benchmark results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with model name and timestamp
        model_name = self.config.model.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"whisper_bench_sglang_{model_name}_{timestamp}"

        # Metadata
        metadata = {
            "model": self.config.model,
            "device": self.config.device,
            "cores": self.config.cores,
            "tensor_parallel": self.config.tp,
            "dtype": self.config.dtype,
            "num_iterations": self.config.num_iterations,
            "warmup_iterations": self.config.warmup_iterations,
            "target_duration_sec": self.config.target_duration,
            "target_sample_rate": self.config.target_sample_rate,
            "batch_sizes": self.config.batch_sizes,
            "timestamp": timestamp,
            "seed": self.config.seed,
            "benchmark_type": "offline",
            "backend": "sglang"
        }

        # Save JSON summary
        json_path = output_dir / f"{base_filename}_summary.json"
        summary = {
            "metadata": metadata,
            "results": [asdict(m) for m in metrics]
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved JSON summary: {json_path}")

        # Save CSV details (per-iteration)
        csv_path = output_dir / f"{base_filename}_details.csv"
        details_df = pd.DataFrame([asdict(r) for r in self.results])
        details_df.to_csv(csv_path, index=False)
        print(f"Saved CSV details: {csv_path}")

        # Save Excel with metrics and metadata (if openpyxl available)
        excel_path = output_dir / f"{base_filename}_report.xlsx"

        try:
            # Create metrics DataFrame
            metrics_df = pd.DataFrame([asdict(m) for m in metrics])
            metrics_df.columns = [
                "BatchSize", "IterationsRun", "MeanProcessingTime(ms)",
                "MedianProcessingTime(ms)", "P95Time(ms)", "StdDev(ms)",
                "AvgTokens/Batch", "Tokens/Second", "Time/Token(ms)",
                "Audios/Second", "TotalAudiosProcessed"
            ]

            # Create metadata DataFrame
            metadata_df = pd.DataFrame([
                {"Parameter": k, "Value": str(v)}
                for k, v in metadata.items()
            ])

            # Write to Excel with multiple sheets
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                metrics_df.to_excel(writer, sheet_name='Experiment Results', index=False)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                details_df.to_excel(writer, sheet_name='Iteration Details', index=False)

            print(f"Saved Excel report: {excel_path}")
        except ImportError:
            print("WARNING: openpyxl not installed, skipping Excel report. Run: pip install openpyxl")
            excel_path = "N/A"

        return str(json_path), str(csv_path), str(excel_path)

    def print_summary_table(self, metrics: List[BatchMetrics]):
        """Print a formatted summary table."""
        print("\n" + "=" * 110)
        print("EXPERIMENT RESULTS SUMMARY (Pure Processing Time - No Network Latency)")
        print("=" * 110)

        # Header
        headers = [
            "BatchSize", "Iters", "Mean(ms)", "Median(ms)",
            "P95(ms)", "StdDev", "Tok/Batch", "Tok/Sec",
            "ms/Token", "Audio/Sec"
        ]
        header_line = " | ".join(f"{h:>10}" for h in headers)
        print(header_line)
        print("-" * len(header_line))

        # Data rows
        for m in metrics:
            row = [
                f"{m.batch_size:>10}",
                f"{m.iterations_run:>10}",
                f"{m.mean_processing_time_ms:>10.2f}",
                f"{m.median_processing_time_ms:>10.2f}",
                f"{m.p95_time_ms:>10.2f}",
                f"{m.std_dev_ms:>10.2f}",
                f"{m.avg_tokens_per_batch:>10.1f}",
                f"{m.tokens_per_second:>10.1f}",
                f"{m.time_per_token_ms:>10.3f}",
                f"{m.audios_per_second:>10.4f}"
            ]
            print(" | ".join(row))

        print("=" * 110)

    def shutdown(self):
        """Shutdown the engine."""
        if self.engine is not None:
            self.engine.shutdown()


def parse_batch_sizes(batch_str: str) -> List[int]:
    """Parse comma-separated batch sizes."""
    sizes = []
    for part in batch_str.split(","):
        part = part.strip()
        if part:
            sizes.append(int(part))
    return sorted(set(sizes))


def main():
    parser = argparse.ArgumentParser(
        description="Whisper Model Offline Benchmarking Script - SGLang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python bench_whisper_sglang.py --model openai/whisper-large-v3 --data-path ./audios

  # Full configuration for CPU
  python bench_whisper_sglang.py \\
      --model openai/whisper-large-v3 \\
      --data-path ./audio_samples \\
      --batch-sizes 1,2,4,8,16,32,64 \\
      --output-dir ./results \\
      --cores "0-63" \\
      --tp 1 \\
      --device cpu \\
      --num-iterations 100 \\
      --warmup-iterations 3

Process:
  1. Pre-processing: Discover and validate audio files
  2. Model Init: Load model into SGLang offline engine
  3. Warm-up: Prime JIT, caches with warmup iterations (discarded)
  4. Timed Evaluation: Run batched inference, measure pure processing time
  5. Aggregation: Calculate mean, median, p95, std dev
        """
    )

    # Required arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name/ID (e.g., openai/whisper-large-v3)"
    )
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        required=True,
        help="Path to directory containing audio files"
    )

    # Batch configuration
    parser.add_argument(
        "--batch-sizes", "-b",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated batch sizes (default: 1,2,4,8,16,32,64)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./results_whisper_sglang",
        help="Output directory for results (default: ./results_whisper_sglang)"
    )

    # CPU/SGLang configuration
    parser.add_argument(
        "--cores",
        type=str,
        default="",
        help="CPU core binding (e.g., '0-31|32-63')"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "xpu"],
        help="Device to run on (default: cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type (default: bfloat16, options: auto, float16, bfloat16, float32)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code from HuggingFace (default: True)"
    )

    # Benchmark configuration
    parser.add_argument(
        "--num-iterations", "-n",
        type=int,
        default=100,
        help="Number of iterations per batch size (default: 100)"
    )
    parser.add_argument(
        "--warmup-iterations", "-w",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)"
    )

    # Audio processing configuration
    parser.add_argument(
        "--target-duration",
        type=float,
        default=30.0,
        help="Target audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Profiling
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Enable torch profiling via SGLang's built-in profiler"
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="sglang_profile",
        help="Directory to save profiler output (default: sglang_profile)"
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    # Create config
    config = BenchmarkConfig(
        model=args.model,
        data_path=args.data_path,
        batch_sizes=batch_sizes,
        output_dir=args.output_dir,
        cores=args.cores,
        tp=args.tp,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
        target_duration=args.target_duration,
        target_sample_rate=args.target_sample_rate,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        profile=args.profile,
        profile_dir=args.profile_dir
    )

    # Run benchmark
    benchmark = WhisperOfflineBenchmark(config)

    try:
        if config.profile:
            metrics = benchmark.run_profiling()
        else:
            metrics = benchmark.run_full_benchmark()

        benchmark.print_summary_table(metrics)
        json_path, csv_path, excel_path = benchmark.save_results(metrics)

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Results saved to:")
        print(f"  - JSON Summary: {json_path}")
        print(f"  - CSV Details:  {csv_path}")
        print(f"  - Excel Report: {excel_path}")
        if config.profile:
            print(f"  - Profile Dir:  {config.profile_dir}")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        benchmark.shutdown()


if __name__ == "__main__":
    main()
