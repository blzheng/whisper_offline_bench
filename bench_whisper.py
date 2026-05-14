#!/usr/bin/env python3
"""
Whisper Model Benchmarking Script - Offline vLLM
================================================

Pure offline inference benchmarking using vLLM's LLM class.
No server, no network latency - just pure model processing time.

Usage:

    OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper.py \
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

    OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper.py \
        --model RedHatAI/whisper-medium-quantized.w8a8 \
        --data-path ./dataset_100_30sec \
        --batch-sizes 4,8,16 \
        --output-dir ./results \
        --cores "0-31|32-63" \
        --tp 2 \
        --num-iterations 100 \
        --warmup-iterations 3 \
        --target-duration 30.0 \
        --target-sample-rate 16000

    OMP_NUM_THREADS=64 VLLM_CPU_OMP_THREADS_BIND='0-63|64-127' python bench_whisper.py \
        --model openai/whisper-medium \
        --data-path ./dataset_100_30sec \
        --batch-sizes 8,16,32 \
        --output-dir ./results_18_03_tp2_128 \
        --cores "0-63|64-127" \
        --tp 2 \
        --num-iterations 100 \
        --warmup-iterations 3 \
        --target-duration 30.0 \
        --target-sample-rate 16000

    OMP_NUM_THREADS=32 python bench_whisper.py \
    --model openai/whisper-medium \
    --data-path ./dataset_100_30sec \
    --batch-sizes 4,8,16 \
    --output-dir ./results_18_03_tp4_128 \
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
from vllm.config import ProfilerConfig

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

# vLLM imports - will be checked at runtime
VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    VLLM_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH_PROFILER_AVAILABLE = True
except ImportError:
    TORCH_PROFILER_AVAILABLE = False
    print("WARNING: torch not installed. Profiling will be disabled.")



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


class AudioProcessor:
    """Handles audio loading, resampling, and preprocessing."""
    
    def __init__(self, target_sample_rate: int, target_duration: float):
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_samples = int(target_sample_rate * target_duration)
    
    def load_and_preprocess(self, audio_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load audio file and resample to target rate.
        Returns (audio, sample_rate) tuple matching temp.py pattern.
        """
        try:
            # Load audio with librosa (matching temp.py exactly)
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None


# Whisper prompt tokens
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


class WhisperOfflineBenchmark:
    """Offline benchmarking class for Whisper with vLLM."""
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm', '.opus'}
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.processor = AudioProcessor(
            config.target_sample_rate, 
            config.target_duration
        )
        self.audio_cache: List[Tuple[str, np.ndarray, int]] = []  # (path, audio_array, sample_rate)
        self.results: List[IterationResult] = []
        self.llm: Optional[LLM] = None

        if self.config.profile:
            self.profile_config =  ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=self.config.output_dir,
                torch_profiler_with_stack=True, 
                torch_profiler_with_flops=True, 
                torch_profiler_record_shapes=True, 
            )
    
        # Set random seed for reproducibility
        np.random.seed(config.seed)

    def _get_omp_num_threads(self) -> int:
        """Determine the number of OMP threads based on CPU cores."""
        if self.config.cores:
            # Count total cores specified in the --cores string
            total_cores = 0
            core_range = self.config.cores.split("|")
            if "-" in core_range:
                start, end = core_range.split("-")
                total_cores = str(int(end) - int(start) + 1)
            else:
                total_cores = "32"
            return total_cores
        else:
            # If no cores specified, use all available cores
            return os.cpu_count() or 1
    
    def setup_cpu_environment(self):
        """Setup CPU-specific environment variables for vLLM."""
        # NOTE: For Whisper (encoder-decoder), we skip CPU-specific optimizations
        # that are tuned for text-only LLMs to avoid matrix dimension issues
        if self.config.device == "cpu" and self.config.cores:
            # os.environ["OMP_NUM_THREADS"] = self._get_omp_num_threads()
            os.environ["VLLM_CPU_OMP_THREADS_BIND"] = self.config.cores
            # os.environ['LD_PRELOAD'] = f"/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:{os.environ['VIRTUAL_ENV']}/lib/libiomp5.so"
            print("CPU environment configured:")
            print(f"  VLLM_CPU_OMP_THREADS_BIND: {self.config.cores}")
            print(f"  OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    
    def init_model(self):
        """Initialize the vLLM model for offline inference."""
        if not VLLM_AVAILABLE:
            print("ERROR: vLLM not installed. Run: pip install vllm")
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
            # Initialize LLM for Whisper
            # Note: Whisper in vLLM uses the encoder-decoder architecture
            self.llm = LLM(
                model=self.config.model,
                tensor_parallel_size=self.config.tp,
                dtype=self.config.dtype,
                # max_num_batched_tokens=16384,
                # enforce_eager=True,  # Ensure we get pure processing time without async optimizations
                trust_remote_code=self.config.trust_remote_code,
                profiler_config=self.profile_config if self.config.profile else None,
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
        """Load and preprocess all audio files, caching valid ones."""
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
            result = self.processor.load_and_preprocess(audio_path)
            
            if result is not None:
                audio, sr = result  # Unpack (audio, sr) tuple
                self.audio_cache.append((audio_path, audio, sr))
                valid_count += 1
            else:
                skipped_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files...")
        
        print(f"\nValid files (>= {self.config.target_duration}s): {valid_count}")
        print(f"Skipped files (too short): {skipped_count}")
        
        if valid_count == 0:
            raise ValueError("No valid audio files found!")
        
        return valid_count
    
    def run_inference_batch(
        self, 
        audio_batch: List[Tuple[np.ndarray, int]]
    ) -> Tuple[float, int, bool, Optional[str]]:
        """
        Run offline inference on a batch of audio.
        Returns: (processing_time_ms, num_tokens, success, error_message)
        """
        try:
            # Prepare inputs for Whisper (matching temp.py pattern)
            inputs = []
            for audio, sr in audio_batch:
                # Create multi-modal input dict for Whisper
                # Audio must be passed as tuple (audio_array, sample_rate)
                inputs.append({
                    "prompt": WHISPER_PROMPT,
                    "multi_modal_data": {
                        "audio": (audio, sr)
                    }
                })
            
            # Sampling params for transcription
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for benchmarking
                max_tokens=448,
            )
            
            # Measure pure processing time
            start_time = time.perf_counter()
            outputs = self.llm.generate(inputs, sampling_params)
            end_time = time.perf_counter()
            
            processing_time_ms = (end_time - start_time) * 1000
            
            # Count total tokens generated
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            
            return processing_time_ms, total_tokens, True, None
            
        except Exception as e:
            return 0.0, 0, False, str(e)
    
    def run_warmup(self) -> bool:
        """Run warmup iterations to prime JIT, caches, etc."""
        print("\n" + "=" * 60)
        print("WARM-UP PHASE")
        print("=" * 60)
        print(f"Running {self.config.warmup_iterations} warmup iterations...")
        
        for i in range(self.config.warmup_iterations):
            # Pick random audio for warmup
            idx = np.random.randint(0, len(self.audio_cache))
            _, audio, sr = self.audio_cache[idx]
            
            proc_time, tokens, success, error = self.run_inference_batch([(audio, sr)])
            
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
                len(self.audio_cache), 
                size=min(batch_size, len(self.audio_cache)), 
                replace=True
            )
            # Extract (audio, sr) tuples for each selected audio
            batch_audios = [(self.audio_cache[i][1], self.audio_cache[i][2]) for i in indices]
            
            proc_time, tokens, success, error = self.run_inference_batch(batch_audios)
            
            result = IterationResult(
                batch_size=batch_size,
                iteration_id=iter_idx,
                processing_time_ms=proc_time,
                num_tokens=tokens,
                num_audios=len(batch_audios),
                success=success,
                error=error
            )
            self.results.append(result)
            
            if success:
                processing_times.append(proc_time)
                tokens_list.append(tokens)
                success_count += 1
                total_audios += len(batch_audios)
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
        print("WHISPER OFFLINE BENCHMARK - vLLM")
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
        """Run the complete benchmark across all batch sizes."""
        print("\n" + "=" * 60)
        print("WHISPER OFFLINE BENCHMARK - vLLM")
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
        

        if self.config.profile:
            print("\nProfiling enabled - starting profiler...")
            self.llm.start_profile()

        for batch_size in sorted(self.config.batch_sizes):
            metrics = self.run_batch_benchmark(batch_size)
            all_metrics.append(metrics)
        
        if self.config.profile:
            self.llm.stop_profile()
            print("\nProfiler stopped. Profile data saved to vLLM profile directory.")
        
        return all_metrics
    
    
    def save_results(self, metrics: List[BatchMetrics]) -> Tuple[str, str, str]:
        """Save benchmark results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with model name and timestamp
        model_name = self.config.model.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"whisper_bench_{model_name}_{timestamp}"
        
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
            "benchmark_type": "offline"
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
        
        # Save Excel with metrics and metadata
        excel_path = output_dir / f"{base_filename}_report.xlsx"
        
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
            
            # Also include per-iteration details
            details_df.to_excel(writer, sheet_name='Iteration Details', index=False)
        
        print(f"Saved Excel report: {excel_path}")
        
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
        description="Whisper Model Offline Benchmarking Script - vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python bench_whisper.py --model openai/whisper-large-v3 --data-path ./audios
  
  # Full configuration for CPU
  python bench_whisper.py \\
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
  1. Pre-processing: Load, resample, and trim audio to target duration
  2. Model Init: Load model into vLLM offline engine
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
        default="./results_whisper",
        help="Output directory for results (default: ./results_whisper)"
    )
    
    # CPU/vLLM configuration (matching run_vllm.sh flags)
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
        help="Data type (default: auto, options: auto, float16, bfloat16, float32)"
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
        help="Target audio duration in seconds (default: 10.0)"
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

    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Enable detailed profiling (not implemented in this version)"
    )

    parser.add_argument(
        "--profile-dir",
        type=str,
        default="vllm_profile",
        help="Directory to save profiler output (default: vllm_profile)"
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
        profile=args.profile,
        trust_remote_code=args.trust_remote_code
    )
    
    # Run benchmark
    benchmark = WhisperOfflineBenchmark(config)
    
    try:
        if config.profile:

            print("\nProfiling enabled - running benchmark with profiler...")

            metrics = benchmark.run_profiling()
            benchmark.print_summary_table(metrics)
            json_path, csv_path, excel_path = benchmark.save_results(metrics) 
            
            # Print profiler summary
            print("\n" + "=" * 60)
            print("PROFILER SUMMARY (Top 20 operations by CPU time)")
            print("=" * 60)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
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
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
