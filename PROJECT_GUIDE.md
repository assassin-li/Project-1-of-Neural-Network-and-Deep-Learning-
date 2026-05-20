# 项目结构与数据流指南

本文档详细介绍本项目的代码结构、数据如何在各模块之间流动，以及如何运行项目。

---

## 一、项目结构

```
.
├── README.md                    # 项目说明与任务要求
├── PROJECT_GUIDE.md             # 本文件
├── dataset_explore.ipynb        # Jupyter Notebook：探索 MNIST 数据集
├── test_train.py                # 训练入口脚本
├── test_model.py                # 测试/评估入口脚本
├── weight_visualization.py      # 权重可视化脚本
├── hyperparameter_search.py     # 超参数搜索（待实现）
├── dataset/
│   └── MNIST/                   # MNIST 数据集（gzip 压缩的 ubyte 格式）
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte.gz
│       ├── t10k-images-idx3-ubyte.gz
│       └── t10k-labels-idx1-ubyte.gz
├── draw_tools/
│   ├── draw.py                  # Tkinter 手写画板（中文界面）
│   └── plot.py                  # 训练曲线绘制工具
└── mynn/                        # 核心神经网络框架（纯 NumPy 实现）
    ├── __init__.py              # 包入口，导出各子模块
    ├── op.py                    # 核心算子：Linear、ReLU、Loss、Conv2D 等
    ├── models.py                # 模型定义：MLP、CNN
    ├── optimizer.py             # 优化器：SGD、MomentGD
    ├── lr_scheduler.py          # 学习率调度器
    ├── runner.py                # 训练器：封装训练循环、评估、保存
    └── metric.py                # 评估指标：accuracy
```

---

## 二、数据流详解

本框架采用**手动反向传播**机制。数据从输入到输出的完整流动过程如下：

### 2.1 数据加载阶段

**入口**：`test_train.py`（第 14~40 行）

```
MNIST gzip 文件 → 解压为 numpy 数组 → 归一化到 [0, 1] → 划分训练集/验证集
```

- 图像数据形状：`[N, 784]`（28×28 展平为一维）
- 标签数据形状：`[N,]`（0~9 的整数类别）
- 随机打乱后，前 10000 条作为验证集，其余 50000 条作为训练集

### 2.2 模型前向传播（Forward）

**入口**：`mynn/models.py` → `Model_MLP.forward()`

```
输入 X [batch, 784]
    ↓
Linear(784 → 600)   →  Z = X @ W + b
    ↓
ReLU                →  A = max(0, Z)
    ↓
Linear(600 → 10)    →  logits = A @ W + b
    ↓
输出 logits [batch, 10]
```

每一层在 `forward()` 中会**保存自己的输入**（`self.input`），供反向传播时使用。

### 2.3 损失计算（Loss）

**入口**：`mynn/op.py` → `MultiCrossEntropyLoss.forward()`

```
logits [batch, 10]
    ↓
Softmax             →  probs [batch, 10]（每行和为 1 的概率分布）
    ↓
取正确类别的概率    →  right_prob [batch,]
    ↓
交叉熵损失          →  loss（标量）
```

**注意**：`MultiCrossEntropyLoss` 内部包含 Softmax。若不需要 Softmax，可调用 `cancel_soft_max()` 取消。

### 2.4 反向传播（Backward）

**入口**：`mynn/runner.py` → `train()` 中调用 `self.loss_fn.backward()`

数据流是**从后往前**逐层传递梯度：

#### 第一步：Loss 层反向
```
MultiCrossEntropyLoss.backward()
    ↓
计算损失对 logits 的梯度 → self.grads [batch, 10]
    ↓
调用 self.model.backward(self.grads)
```

#### 第二步：模型层反向
```
Model_MLP.backward(loss_grad)
    ↓
遍历 layers（从最后一层到第一层）：
    Linear.backward(grad)   → 计算 dW, db；返回对输入的梯度
    ReLU.backward(grad)     → 将负数位置的梯度置零；返回梯度
    ↓
最终梯度传回第一层
```

以 `Linear` 为例，反向传播公式：
```
grads['W'] = input.T @ grad_out          # [in_dim, out_dim]
grads['b'] = sum(grad_out, axis=0)       # [1, out_dim]
grad_in  = grad_out @ W.T                # [batch, in_dim]
```

### 2.5 参数更新（Optimizer Step）

**入口**：`mynn/optimizer.py` → `SGD.step()`

```
遍历所有可优化层（optimizable == True）：
    如果开启 weight_decay：
        params[key] *= (1 - lr * lambda)     # L2 正则化（权重衰减）
    params[key] -= lr * grads[key]            # 梯度下降更新
```

### 2.6 学习率调度

**入口**：`mynn/lr_scheduler.py`

`RunnerM` 在每个 batch 训练后调用 `scheduler.step()`。例如 `StepLR`：
```
每 step_size 个 batch → lr *= gamma
```

### 2.7 评估与保存

**入口**：`mynn/runner.py` → `evaluate()` / `save_model()`

- `evaluate()`：前向传播 → 计算 loss 和 accuracy
- `save_model()`：将 `[size_list, act_func, {W, b, weight_decay, lambda}, ...]` 用 `pickle` 保存到 `.pickle` 文件

---

## 三、应该运行什么？

### 3.1 探索数据（推荐第一步）

打开 `dataset_explore.ipynb`，了解 MNIST 数据的二进制格式，并可视化一些样本图片。

### 3.2 训练模型

```bash
python test_train.py
```

**流程说明**：
1. 加载 MNIST 训练数据
2. 划分出 10000 条验证集
3. 构建 `Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])`
4. 使用 `SGD` 优化器 + `MultiStepLR` 学习率调度
5. 训练 5 个 epoch，每 100 个 iteration 打印日志
6. 自动保存验证集上表现最好的模型到 `./best_models/best_model.pickle`
7. 弹出 Matplotlib 窗口，显示训练曲线（loss 和 accuracy）

**自定义训练**：
- 修改 `test_train.py` 中的 `train_images_path` / `train_labels_path` 可使用自己的数据集
- 修改网络结构：`Model_MLP([784, 隐藏层大小, 10], 'ReLU', [...])`
- 修改学习率：`SGD(init_lr=0.06, ...)`
- 修改训练轮数：`runner.train(..., num_epochs=5, ...)`

### 3.3 测试模型

```bash
python test_model.py
```

**流程说明**：
1. 加载已保存的模型（默认路径 `.\saved_models\best_model_1.pickle`）
2. 加载 MNIST 测试集（10000 条）
3. 前向传播计算 logits
4. 输出测试集准确率

**注意**：请修改 `test_model.py` 第 9 行的模型路径，指向你实际保存的 `.pickle` 文件。

### 3.4 可视化权重

```bash
python weight_visualization.py
```

加载已训练好的模型，将第一层 Linear 层的权重可视化，观察网络学习到了什么样的特征模式。

### 3.5 手写数字识别演示

```bash
python draw_tools/draw.py
```

启动 Tkinter 画板，用鼠标手写数字，程序会调用已训练的模型进行实时识别（需要预先保存好模型并配置正确路径）。

---

## 四、模块依赖关系图

```
test_train.py
    ├── mynn.models.Model_MLP
    │       └── mynn.op.Linear, ReLU
    ├── mynn.optimizer.SGD
    │       └── 遍历 model.layers 更新参数
    ├── mynn.lr_scheduler.MultiStepLR
    │       └── 修改 optimizer.init_lr
    ├── mynn.op.MultiCrossEntropyLoss
    │       └── Softmax + 交叉熵
    ├── mynn.runner.RunnerM
    │       └── 整合 model, optimizer, loss_fn, scheduler
    │       └── 调用 train() → forward → loss → backward → step
    └── draw_tools.plot.plot
            └── Matplotlib 绘制训练曲线

test_model.py
    ├── mynn.models.Model_MLP
    │       └── load_model() → pickle 反序列化
    └── mynn.metric.accuracy
            └── np.argmax 计算分类准确率
```

---

## 五、开发注意事项

1. **手动反向传播**：修改 `op.py` 中的算子时，务必同时正确实现 `forward()` 和 `backward()`，否则梯度无法正确回传。
2. **形状检查**：NumPy 矩阵乘法对形状要求严格。若出现维度不匹配错误，请检查各层输入输出形状。
3. **Pickle 路径**：模型保存/加载使用 Windows 风格路径（如 `r'.\best_models'`）。若在 Linux/macOS 上运行，请改为 `'./best_models'`。
4. **Loss 的语法错误**：`mynn/op.py` 中 `MultiCrossEntropyLoss.forward()` 存在未完成的代码（`right_prob = ` 后缺少内容），需要自行补全。
5. **Batch Size**：`RunnerM` 默认 `batch_size=32`。最后一个不完整的 batch 也会参与训练（由于 `int(X.shape[0] / self.batch_size) + 1`）。
