# microgpt (optimized)

![microgpt logo](assets/microgpt.png)

Optimized version of [Karpathy's microgpt](https://karpathy.ai/microgpt.html), the most atomic way to train and inference a GPT in pure, dependency-free Python.

**293 lines, 0 dependencies.** All optimizations preserve the original simplicity.

## What's Changed

| Optimization | Lines | Impact |
|---|---|---|
| Direct `__truediv__` implementation | +8 | ~20-30% fewer computation graph nodes per step |
| Fused `cross_entropy` (log-softmax + NLL) | +5 | Fewer nodes + better numerical stability |
| Iterative `backward()` topological sort | 0 | Eliminates recursion depth limit |
| `sum(losses[1:], losses[0])` | 0 | Removes phantom `Value(0)` node |
| Adam running product | +2 | Numerically stable bias correction at large step counts |
| `with open()` file handle | +1 | Proper resource cleanup |
| **Weight tying** (wte = lm_head) | -1 | Standard GPT-2 practice, fewer params |
| **Cosine LR schedule** | 0 | Smoother decay than linear |
| **Train/val split** (90/10) | +3 | Basic ML hygiene, detect overfitting |
| **Periodic validation** (every 100 steps) | +10 | Pure-float NLL eval on held-out docs |
| **Gradient clipping** (global norm) | +4 | Prevents exploding gradients, stabilizes training |
| **AdamW weight decay** | +1 | Decoupled regularization |
| **Top-k sampling** (k=5) | +4 | Higher quality inference, avoids garbage tokens |
| **Per-step timing** | +3 | Performance observability in ms/step |

**Total: +50 lines** (243 -> 293), no new dependencies.

## Files

- **`microgpt.py`** - Complete optimized Python algorithm (runnable)
- **`microgpt_cuda.cu`** - CUDA/C++ port with full train/val/inference loop
- **`microgpt_optimized.html`** - Syntax-highlighted 3-column view with change annotations
- **`CMakeLists.txt`** - CMake entrypoint for CUDA build

## Quick Start

```bash
python microgpt.py
```

It auto-downloads `input.txt` on first run, trains for 500 steps with periodic validation, then generates samples via top-k sampling.

## CUDA Build

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\microgpt_cuda.exe
```

For quick smoke tests:

```bash
.\build\Release\microgpt_cuda.exe --steps 5 --samples 3
```

## Credits

Original by [@karpathy](https://github.com/karpathy) - [microgpt](https://karpathy.ai/microgpt.html) | [Gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
