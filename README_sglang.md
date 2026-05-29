# Whisper Offline Benchmarking with SGLang

A comprehensive benchmarking script for OpenAI Whisper models using SGLang's offline inference engine. This tool measures pure model processing performance without server or network overhead.

## Installation

```
# Create conda ENV
conda create -n sglang python=3.10
conda activate sglang
pip install uv

# Install SGLang
git clone -b beilei/whisper https://github.com/blzheng/sglang.git sglang
cd sglang/
cd python/
cp pyproject_cpu.toml  pyproject.toml
pip install -e .
export VER_TORCH=2.9.0
export VER_TORCHVISION=0.24.0
export VER_TRITON=3.5.0
uv pip install torch==${VER_TORCH} torchvision==${VER_TORCHVISION} triton==${VER_TRITON} --force-reinstall
pip install torch==${VER_TORCH} torchvision==${VER_TORCHVISION} triton==${VER_TRITON} --force-reinstall --index-url https://download.pytorch.org/whl/cpu
cd ../sgl-kernel/
cp pyproject_cpu.toml pyproject.toml
pip install uv
pip install scikit-build-core
uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
pip install dist/sgl_kernel_cpu-0.3.21-cp312-cp312-linux_x86_64.whl

export SGLANG_USE_CPU_ENGINE=1
conda install -y gperftools -c conda-forge
pip install intel-openmp==2024.2.0

# Set IOMP and tcmalloc Preload for better performance
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so

pip install pandas openpyxl

```


## Quick Start

```bash
# 1. Run basic benchmark (Results will be saved in ./results/ directory)
SGLANG_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper_sglang.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 4,8,16 \
  --output-dir ./results

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

#### CPU/SGLang Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--cores` | `""` | CPU core binding (e.g., `'0-31\|32-63'` for TP=2) |
| `--tp` | `1` | Tensor parallel size |
| `--device` | `cpu` | Device to run on (`cpu`) |
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
| `--profile-dir` | | `sglang_profile` | Directory to save profiler output |

### Examples

#### Basic CPU Benchmarking (TP=1)

```bash
OMP_NUM_THREADS=64 python bench_whisper_sglang.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 1,2,4,8,16 \
  --output-dir ./results_tp1 \
  --tp 1
```

#### CPU with Tensor Parallelism (TP=2)

```bash
SGLANG_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper_sglang.py \
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
python bench_whisper_sglang.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 8,16,32,64 \
  --output-dir ./results_tp4 \
  --cores "0-31|32-63|64-95|96-127" \
  --tp 4 \
  --num-iterations 100
```

#### With Profiling Enabled

```bash
SGLANG_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper_sglang.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 8,16 \
  --output-dir ./results_profiled \
  --cores "0-31|32-63" \
  --tp 2 \
  --profile \
  --profile-dir ./sglang_profile \
  --num-iterations 10
```

#### With torch.compile Enabled

```bash
SGLANG_CPU_OMP_THREADS_BIND='0-31|32-63' python bench_whisper_sglang.py \
  --model openai/whisper-medium \
  --data-path ./dataset_100_30sec \
  --batch-sizes 8,16 \
  --output-dir ./results_profiled \
  --cores "0-31|32-63" \
  --tp 2 \
  --enable-torch-compile \
  --num-iterations 10
```


