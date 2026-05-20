---
name: nndl-project-1
description: Neural Network and Deep Learning course Project-1 requirements, deadline, and grading criteria.
metadata:
  type: project
---

Course project for "Neural Network and Deep Learning" (Yanwei Fu, Shufeng Nan, Fudan).

**Deadline:** 2026-05-24 23:59 (late penalty: 10% per week).
**Submission:** single PDF report via eLearning + GitHub code link. Do not upload dataset/weights to GitHub; use ModelScope for model files.
**Contact:** deeplearning.fudan@yandex.com

**Three Parts:**
1. **Part A — MLP Baseline (Required):** implement Linear forward/backward, cross-entropy loss with softmax, train on MNIST, report train/val performance, include learning curve.
2. **Part B — CNN Model (Required):** implement conv2D by yourself, build simple CNN, compare fairly with MLP, discuss why CNN is better/worse.
3. **Part C — Two Additional Directions (choose 2 of 5):**
   - Direction 1: Optimization (momentum, LR scheduling, different LR)
   - Direction 2: Regularization (L2, dropout, early stopping)
   - Direction 3: Data Augmentation (small rotation, translation, resizing)
   - Direction 4: Robustness Analysis (rotation, translation, Gaussian noise)
   - Direction 5: Error Analysis and Visualization (confusion matrix, misclassified examples, weight/kernel visualization)

**Grading:**
- Implementation correctness: 35%
- CNN implementation + MLP-vs-CNN comparison: 30%
- Quality of two chosen directions: 20%
- Discussion and report clarity: 15%

**Key Principles:**
- Do NOT use PyTorch/TensorFlow for the required operators.
- Change one major factor at a time for fair comparisons.
- Clear comparison and explanation matter more than trying many disconnected ideas.
- CPU only; no GPU needed.

**Why:** This is a foundational course project focused on understanding core NN components through manual NumPy implementation.
**How to apply:** When helping, prioritize correctness of forward/backward passes and fair experimental design over chasing SOTA accuracy. Keep comparisons controlled.
