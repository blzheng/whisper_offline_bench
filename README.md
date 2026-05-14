# Whisper Offline Benchmarking with vLLM

A comprehensive benchmarking script for OpenAI Whisper models using vLLM's offline inference engine. This tool measures pure model processing performance without server or network overhead.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [CPU Installation](#cpu-installation)
  - [XPU Installation](#xpu-installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Options](#command-line-options)
  - [Examples](#examples)
- [Output](#output)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Pure Offline Inference**: No server, no network latency - just model processing time
- **Multi-Batch Testing**: Benchmark multiple batch sizes in a single run
- **Comprehensive Metrics**: Mean, median, P95, standard deviation, tokens/sec, audios/sec
- **Audio Preprocessing**: Automatic resampling and duration normalization
- **Profiling Support**: Built-in PyTorch profiler integration
- **Reproducible**: Seeded random selection for consistent results
- **Multiple Output Formats**: JSON, CSV, and Excel reports

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Virtual environment tool (`uv` or `venv`)
- Intel CPU or XPU hardware

### CPU Installation

For Intel CPU support with vLLM:

```bash
# 1. Create and activate virtual environment
uv venv --python=3.10
source .venv/bin/activate

# 2. Install vLLM for CPU
export VLLM_VERSION=0.18.0
uv pip install \
  https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl \
  --torch-backend cpu

# 3. Install audio processing dependencies
uv pip install librosa soundfile numpy pandas openpyxl

# 4. (Optional) Install tcmalloc for better memory performance
sudo apt-get install libtcmalloc-minimal4

# 5. Set environment variables
export OMP_NUM_THREADS=32
export VLLM_CPU_OMP_THREADS_BIND='0-31|32-63'
export TC_PATH="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
export LD_PRELOAD="$TC_PATH:$VIRTUAL_ENV/lib/libiomp5.so"
```

### XPU Installation

For Intel XPU (GPU) support:

```bash
# 1. Create and activate virtual environment
uv venv --python=3.10
source .venv/bin/activate

# 2. Install prerequisites
uv pip install vllm-xpu-kernels
uv pip install packaging==25.0 cmake==3.27.0 setuptools==80.0

# 3. Clone and build vLLM for XPU
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -v -r requirements/xpu.txt
VLLM_TARGET_DEVICE=xpu uv pip install --no-build-isolation -e . -v

# 4. Install correct Triton version
uv pip uninstall triton triton-xpu
uv pip install triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu

# 5. Install audio dependencies
uv pip install librosa soundfile numpy pandas openpyxl
```

---

## Quick Start

```bash
# 1. Run basic benchmark
OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 4,8,16 \
  --output-dir ./results

# 3. Results will be saved in ./results/ directory
```

---

## Usage

### Command-Line Options

#### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--model` | `-m` | Model name/ID (e.g., `openai/whisper-medium`, `openai/whisper-large-v3`) |
| `--data-path` | `-d` | Path to directory containing audio files |

#### Batch Configuration

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--batch-sizes` | `-b` | `1,2,4,8,16,32,64` | Comma-separated batch sizes to test |

#### Output Configuration

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--output-dir` | `-o` | `./results_whisper` | Output directory for results |

#### CPU/vLLM Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--cores` | `""` | CPU core binding (e.g., `'0-31\|32-63'` for TP=2) |
| `--tp` | `1` | Tensor parallel size |
| `--device` | `cpu` | Device to run on (`cpu`, `cuda`, `xpu`) |
| `--dtype` | `bfloat16` | Data type (`bfloat16`, `float16`, `float32`) |
| `--trust-remote-code` | `True` | Trust remote code from HuggingFace |

#### Benchmark Configuration

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--num-iterations` | `-n` | `100` | Number of iterations per batch size |
| `--warmup-iterations` | `-w` | `3` | Number of warmup iterations (discarded) |

#### Audio Processing

| Argument | Default | Description |
|----------|---------|-------------|
| `--target-duration` | `30.0` | Target audio duration in seconds |
| `--target-sample-rate` | `16000` | Target sample rate in Hz |
| `--seed` | `42` | Random seed for reproducibility |

#### Profiling

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--profile` | `-p` | `False` | Enable PyTorch profiler |
| `--profile-dir` | | `vllm_profile` | Directory to save profiler output |

### Examples

#### Basic CPU Benchmarking (TP=1)

```bash
OMP_NUM_THREADS=64 python bench_whisper.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 1,2,4,8,16 \
  --output-dir ./results_tp1 \
  --tp 1
```

#### CPU with Tensor Parallelism (TP=2)

```bash
OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 4,8,16,32 \
  --output-dir ./results_tp2 \
  --cores "0-31|32-63" \
  --tp 2 \
  --num-iterations 100
```

#### CPU with Tensor Parallelism (TP=4)

```bash
OMP_NUM_THREADS=32 python bench_whisper.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 8,16,32,64 \
  --output-dir ./results_tp4 \
  --cores "0-31|32-63|64-95|96-127" \
  --tp 4 \
  --num-iterations 100
```

#### Quantized Model Benchmarking

```bash
OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper.py \
  --model RedHatAI/whisper-medium-quantized.w8a8 \
  --data-path ./dataset_100_30sec \
  --batch-sizes 8,16,32 \
  --output-dir ./results_quantized \
  --cores "0-31|32-63" \
  --tp 2
```

#### With Profiling Enabled

```bash
OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 8,16 \
  --output-dir ./results_profiled \
  --cores "0-31|32-63" \
  --tp 2 \
  --profile \
  --profile-dir ./vllm_profile \
  --num-iterations 10
```

#### XPU/GPU Benchmarking

```bash
python bench_whisper.py \
  --model openai/whisper-large-v3 \
  --data-path ./dataset_100_30sec \
  --batch-sizes 8,16,32,64 \
  --output-dir ./results_xpu \
  --device xpu \
  --dtype bfloat16 \
  --tp 2
```

---

## Output

The benchmark generates three output files for each run:

### 1. JSON Summary (`*_summary.json`)

Contains metadata and aggregated metrics:

```json
{
  "metadata": {
    "model": "openai/whisper-medium",
    "device": "cpu",
    "tensor_parallel": 2,
    "num_iterations": 100,
    ...
  },
  "results": [
    {
      "batch_size": 8,
      "mean_processing_time_ms": 1420.5,
      "tokens_per_second": 252.3,
      "audios_per_second": 5.63,
      ...
    }
  ]
}
```

### 2. CSV Details (`*_details.csv`)

Per-iteration results:

| batch_size | iteration_id | processing_time_ms | num_tokens | num_audios | success |
|------------|--------------|--------------------|-----------|-----------:|---------|
| 8 | 0 | 1425.3 | 358 | 8 | True |
| 8 | 1 | 1418.7 | 360 | 8 | True |

### 3. Excel Report (`*_report.xlsx`)

Multi-sheet workbook with:
- **Experiment Results**: Aggregated metrics per batch size
- **Metadata**: Configuration parameters
- **Iteration Details**: Per-iteration raw data

### Console Output

The script prints a summary table at the end:

```
================================================================================
EXPERIMENT RESULTS SUMMARY (Pure Processing Time - No Network Latency)
================================================================================
 BatchSize |      Iters |   Mean(ms) | Median(ms) |    P95(ms) |     StdDev | Tok/Batch |   Tok/Sec | ms/Token | Audio/Sec
------------|------------|------------|------------|------------|------------|-----------|-----------|----------|----------
         4 |        100 |     712.45 |     710.23 |     725.30 |      15.20 |      180.0 |     253.2 |    3.950 |    5.6150
         8 |        100 |    1420.50 |    1418.75 |    1445.80 |      28.50 |      360.0 |     253.5 |    3.946 |    5.6321
        16 |        100 |    2840.20 |    2835.60 |    2890.40 |      55.30 |      720.0 |     253.6 |    3.944 |    5.6340
================================================================================
```

---

## Contributing

For issues or improvements, please open an issue or pull request in the repository.

## License

This project follows the vLLM license terms.
