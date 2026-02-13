# microgpt (optimized + CUDA)

![microgpt logo](assets/microgpt.png)

A minimal GPT project with two aligned implementations:

- `microgpt.py`: pure Python, dependency-free reference implementation.
- `microgpt_cuda.cu`: CUDA/C++ implementation for Windows (MSVC + CUDA), optimized for speed while keeping the same model/training logic.

## What this repo focuses on

- Keep the project small and readable.
- Preserve algorithmic parity between Python and CUDA paths.
- Push performance through GPU residency and kernel fusion where it matters.

Core model/training recipe (both paths):

- Character tokenizer with `<BOS>`.
- GPT-style block with RMSNorm, causal multi-head attention, and ReLU^2 MLP.
- Weight tying (`wte` reused as LM head).
- AdamW + cosine LR + global grad clipping.
- Train/val split, periodic validation, top-k sampling inference.

## Repository layout

- `microgpt.py`: full Python algorithm (train + val + inference).
- `microgpt_cuda.cu`: full CUDA/C++ algorithm (train + val + inference).
- `microgpt_optimized.html`: side-by-side Python/CUDA code converter view.
- `CMakeLists.txt`: CUDA build entry.
- `input.txt`: corpus (auto-downloaded if missing on first run).

## Quick start (Python)

```bash
python microgpt.py
```

If `input.txt` is missing, the script downloads the default names dataset automatically.

## Quick start (CUDA / Windows)

Prerequisites:

- NVIDIA GPU + compatible driver
- CUDA Toolkit (your setup: CUDA 13.1)
- Visual Studio 2022 (MSVC, x64 toolchain)
- CMake 3.24+

Build:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

Run:

```bash
.\build\Release\microgpt_cuda.exe --help
.\build\Release\microgpt_cuda.exe
```

Smoke test:

```bash
.\build\Release\microgpt_cuda.exe --steps 5 --samples 3
```

## CUDA CLI options

- `--steps <int>`: training steps (default `500`)
- `--val-every <int>`: validation interval (default `100`)
- `--val-docs <int>`: max validation docs per eval (default `20`)
- `--samples <int>`: generated samples after training (default `20`)
- `--top-k <int>`: top-k for sampling (default `5`)
- `--temperature <float>`: sampling temperature (default `0.6`)
- `--seed <int>`: RNG seed (default `42`)

## Important implementation notes

- CUDA path keeps parameters, gradients, and optimizer states on GPU.
- Training step is fused into one kernel launch (forward + backward + grad clip + AdamW update).
- Current fused implementation is specialized to `n_layer = 1` (same as current Python config).
- `kMaxVocab = 256` in `microgpt_cuda.cu`; if your dataset exceeds this, increase it and rebuild.
- Default `CMAKE_CUDA_ARCHITECTURES` is `86`; set it to your GPU architecture when needed.

## Code converter page

Open `microgpt_optimized.html` in a browser to switch between:

- Python view
- CUDA view
- Bilingual side-by-side comparison

This is useful for checking one-to-one conceptual mapping between the two codebases.

## Credits

Original microgpt idea and baseline by [@karpathy](https://github.com/karpathy):

- https://karpathy.ai/microgpt.html
- https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
