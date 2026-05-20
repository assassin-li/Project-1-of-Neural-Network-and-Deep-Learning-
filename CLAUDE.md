# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an educational deep learning framework built from scratch in NumPy. It implements core neural network components (forward/backward passes, optimizers, schedulers) without PyTorch or TensorFlow. The codebase is small (~500 lines of core logic) and entirely contained in `mynn/`.

## Dependencies

There is no package manager file. Ensure the following are installed:

- `numpy`
- `matplotlib`
- `Pillow` (for `draw_tools/draw.py`)
- `tqdm`
- `tkinter` (standard library, used by drawing GUI)

## Common Commands

- **Train a model**: `python test_train.py`
- **Evaluate a saved model**: `python test_model.py`
- **Explore dataset**: Open `dataset_explore.ipynb` in Jupyter
- **Visualize weights**: `python weight_visualization.py`

No formal test suite, linter, or build system exists.

## Architecture

### `mynn/` — Core Framework

| File | Purpose |
|------|---------|
| `op.py` | Base `Layer`, `Linear`, `ReLU`, `MultiCrossEntropyLoss` (partial), `softmax`, `conv2D` (stub) |
| `models.py` | `Model_MLP` (implemented) and `Model_CNN` (stub); `save_model`/`load_model` via `pickle` |
| `optimizer.py` | `Optimizer` base, `SGD` (implemented), `MomentGD` (stub) |
| `lr_scheduler.py` | `StepLR` (implemented), `MultiStepLR` (stub), `ExponentialLR` (stub) |
| `runner.py` | `RunnerM` class: training loop, evaluation, model checkpointing |
| `metric.py` | `accuracy` metric for classification |

### Entry Scripts

- `test_train.py`: Loads MNIST data, builds `Model_MLP`, trains with `RunnerM`, and plots results.
- `test_model.py`: Loads a saved `.pickle` model and evaluates on the MNIST test set.

### Data and Visualization

- `dataset/MNIST/`: Contains gzipped MNIST ubyte files.
- `draw_tools/draw.py`: Tkinter-based paint application (Chinese UI) for drawing digits.
- `draw_tools/plot.py`: Helper to plot training/dev loss and accuracy curves.

## Key Implementation Details

- **Manual backprop**: All layers implement `forward()` and `backward()`. `Linear.backward()` computes grads for `W` and `b` and passes gradients to the previous layer.
- **Weight decay**: Implemented inside `Linear` and applied during the optimizer step when `weight_decay=True`.
- **Loss backward**: `MultiCrossEntropyLoss.backward()` computes gradients and calls `self.model.backward(self.grads)` to propagate through the network.
- **Pickle serialization**: `save_model` stores `[size_list, act_func, {W, b, weight_decay, lambda}, ...]`. `load_model` reconstructs the layer list from this list.
- **Data paths**: Hardcoded Windows-style relative paths (e.g., `r'.\dataset\MNIST\train-images-idx3-ubyte.gz'`). Adjust if running on a different OS.
- **Runner evaluation**: `RunnerM.train()` evaluates on the dev set after every batch, not just every epoch.

## Known Incomplete Stubs

Several files contain student-assigned stubs that may need completion:

- `mynn/op.py`: `MultiCrossEntropyLoss.forward()` (has a syntax error), `conv2D`
- `mynn/optimizer.py`: `MomentGD`
- `mynn/lr_scheduler.py`: `MultiStepLR`, `ExponentialLR`
- `mynn/models.py`: `Model_CNN`
- `hyperparameter_search.py`: Empty stub
