"""
Visualization utilities for MNIST experiments.
Includes: learning curves, confusion matrix, weight visualization, CNN kernel visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_learning_curves(train_loss, dev_loss, train_scores, dev_scores,
                         dev_iters=None, save_path=None, title="Learning Curves"):
    """
    Plot training and validation loss / accuracy curves.
    train_loss/scores: recorded per iteration
    dev_loss/scores: recorded at eval intervals
    dev_iters: iteration indices where dev was evaluated (default: same as train)
    """
    train_iters = np.arange(len(train_loss))
    if dev_iters is None:
        dev_iters = np.arange(len(dev_loss))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(train_iters, train_loss, color='#E3E37D', alpha=0.6, label="Train loss")
    axes[0].plot(dev_iters, dev_loss, color='#968A62', linestyle="-", marker="o", markersize=3, label="Dev loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_title("Loss Curve")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(train_iters, train_scores, color='#E3E37D', alpha=0.6, label="Train accuracy")
    axes[1].plot(dev_iters, dev_scores, color='#968A62', linestyle="-", marker="o", markersize=3, label="Dev accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")
    plt.close()


def plot_confusion_matrix(model, X, y, class_names=None, save_path=None, title="Confusion Matrix"):
    """
    Compute and plot confusion matrix for given data.
    model: trained model with __call__ method
    X: input data [N, ...]
    y: true labels [N]
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # Forward pass
    logits = model(X)
    preds = np.argmax(logits, axis=-1)

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y, preds):
        cm[true_label, pred_label] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted label',
           ylabel='True label',
           title=title)

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.close()

    return cm


def visualize_mlp_weights(model, save_path=None, title="MLP Weight Visualization"):
    """
    Visualize weights of the first Linear layer in an MLP.
    For MNIST [784 -> ...], each weight column can be reshaped to 28x28.
    """
    # Find first Linear layer
    first_linear = None
    for layer in model.layers:
        if hasattr(layer, 'params') and 'W' in layer.params:
            first_linear = layer
            break

    if first_linear is None:
        print("No Linear layer found in model.")
        return

    W = first_linear.params['W']  # shape [in_dim, out_dim]
    in_dim, out_dim = W.shape

    if in_dim != 784:
        print(f"First Linear in_dim is {in_dim}, not 784. Skipping 28x28 reshape.")
        return

    n_show = min(out_dim, 64)  # show at most 64 filters
    grid_size = int(np.ceil(np.sqrt(n_show)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(n_show):
        ax = axes[i]
        w_img = W[:, i].reshape(28, 28)
        vmax = np.abs(w_img).max()
        ax.imshow(w_img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide extra subplots
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved MLP weight visualization to {save_path}")
    plt.close()


def visualize_cnn_kernels(model, save_path=None, title="CNN Convolution Kernels"):
    """
    Visualize convolution kernels from conv2D layers.
    For first conv layer with single-channel input, show each kernel as grayscale.
    For deeper layers, show average across input channels.
    """
    conv_layers = []
    for layer in model.layers:
        if layer.__class__.__name__ == 'conv2D':
            conv_layers.append(layer)

    if not conv_layers:
        print("No conv2D layers found in model.")
        return

    n_layers = len(conv_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    for idx, layer in enumerate(conv_layers):
        W = layer.params['W']  # [out_channels, in_channels, k, k]
        out_c, in_c, k, _ = W.shape

        # Average over input channels for visualization
        W_vis = W.mean(axis=1)  # [out_channels, k, k]

        n_show = min(out_c, 16)
        grid = int(np.ceil(np.sqrt(n_show)))

        # Create a grid image
        grid_img = np.zeros((grid * k, grid * k))
        for i in range(n_show):
            row = i // grid
            col = i % grid
            kernel = W_vis[i]
            vmax = np.abs(kernel).max() + 1e-8
            # Normalize to [0, 1] for display
            grid_img[row * k:(row + 1) * k, col * k:(col + 1) * k] = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)

        ax = axes[idx]
        ax.imshow(grid_img, cmap='viridis')
        ax.set_title(f"Conv Layer {idx+1}\n{out_c}x{in_c}x{k}x{k}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved CNN kernel visualization to {save_path}")
    plt.close()


def visualize_misclassified_examples(model, X, y, class_names=None,
                                      n_show=16, save_path=None,
                                      title="Misclassified Examples"):
    """
    Visualize misclassified examples.
    X: input images [N, ...]
    y: true labels [N]
    n_show: number of examples to show (default 16)
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    logits = model(X)
    preds = np.argmax(logits, axis=-1)

    mis_idx = np.where(preds != y)[0]
    if len(mis_idx) == 0:
        print("No misclassified examples found!")
        return

    # Randomly sample misclassified examples
    n_show = min(n_show, len(mis_idx))
    chosen = np.random.choice(mis_idx, n_show, replace=False)

    grid = int(np.ceil(np.sqrt(n_show)))
    fig, axes = plt.subplots(grid, grid, figsize=(10, 10))
    axes = axes.flatten()

    for i, idx in enumerate(chosen):
        ax = axes[i]
        img = X[idx]

        # Reshape for display
        if img.ndim == 1:
            img = img.reshape(28, 28)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)  # [1, 28, 28] -> [28, 28]
        elif img.ndim == 3:
            img = img.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]

        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {class_names[y[idx]]}\nPred: {class_names[preds[idx]]}",
                     color='red', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved misclassified examples to {save_path}")
    plt.close()


def plot_epoch_learning_curves(train_loss_per_epoch, dev_loss_per_epoch,
                                train_acc_per_epoch, dev_acc_per_epoch,
                                save_path=None, title="Learning Curves (per Epoch)"):
    """
    Plot learning curves averaged per epoch (smoother than per-iteration).
    """
    epochs = np.arange(1, len(train_loss_per_epoch) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss_per_epoch, 'o-', color='#E3E37D', label="Train loss")
    axes[0].plot(epochs, dev_loss_per_epoch, 's--', color='#968A62', label="Dev loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc_per_epoch, 'o-', color='#E3E37D', label="Train accuracy")
    axes[1].plot(epochs, dev_acc_per_epoch, 's--', color='#968A62', label="Dev accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved epoch learning curves to {save_path}")
    plt.close()
