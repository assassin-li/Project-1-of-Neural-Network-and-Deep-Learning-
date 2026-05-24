"""
Compare different learning rates on MLP.
Trains 3 MLP models with small/moderate/large learning rates,
plots their learning curves on the same figure for direct comparison.
"""

import numpy as np
import mynn as nn
from visualization import (plot_learning_curves, plot_confusion_matrix,
                            visualize_mlp_weights, plot_epoch_learning_curves)
import os
import pickle
from struct import unpack
import gzip
import matplotlib.pyplot as plt


def load_mnist_data(normalize_type='minmax', seed=309):
    """Load MNIST and split into train/dev/test."""
    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    np.random.seed(seed)
    idx = np.random.permutation(np.arange(train_imgs.shape[0]))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    if normalize_type == 'minmax':
        train_imgs = train_imgs / 255.0
        valid_imgs = valid_imgs / 255.0
        test_imgs = test_imgs / 255.0
    elif normalize_type == 'standard':
        mean = train_imgs.mean(axis=0, keepdims=True)
        std = train_imgs.std(axis=0, keepdims=True) + 1e-8
        train_imgs = (train_imgs - mean) / std
        valid_imgs = (valid_imgs - mean) / std
        test_imgs = (test_imgs - mean) / std

    return (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs)


def train_model(model, optimizer, loss_fn, train_set, dev_set,
                num_epochs=5, batch_size=32, scheduler=None,
                log_iters=100, eval_interval=50, save_dir=None, exp_name='exp'):
    """Training loop that records per-epoch and per-iteration metrics."""
    X_train, y_train = train_set
    X_dev, y_dev = dev_set

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    history = {
        'train_loss': [], 'train_acc': [],
        'dev_loss': [], 'dev_acc': [],
        'iteration_train_loss': [], 'iteration_train_acc': [],
        'iteration_dev_loss': [], 'iteration_dev_acc': [],
        'iteration_dev_iters': [],
    }

    best_dev_acc = 0.0
    best_model_path = None

    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    for epoch in range(num_epochs):
        idx = np.random.permutation(n_samples)
        X_train_shuffled = X_train[idx]
        y_train_shuffled = y_train[idx]

        epoch_train_losses = []
        epoch_train_accs = []

        for iteration in range(n_batches):
            start = iteration * batch_size
            end = min((iteration + 1) * batch_size, n_samples)
            batch_X = X_train_shuffled[start:end]
            batch_y = y_train_shuffled[start:end]

            if batch_X.shape[0] == 0:
                continue

            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            acc = nn.metric.accuracy(logits, batch_y)

            epoch_train_losses.append(loss)
            epoch_train_accs.append(acc)
            history['iteration_train_loss'].append(loss)
            history['iteration_train_acc'].append(acc)

            loss_fn.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_iter = epoch * n_batches + iteration
            if iteration % eval_interval == 0:
                dev_logits = model(X_dev)
                dev_loss = loss_fn(dev_logits, y_dev)
                dev_acc = nn.metric.accuracy(dev_logits, y_dev)
                history['iteration_dev_loss'].append(dev_loss)
                history['iteration_dev_acc'].append(dev_acc)
                history['iteration_dev_iters'].append(global_iter)

                if iteration % log_iters == 0:
                    print(f"[{exp_name}] Epoch {epoch}, Iter {iteration}: "
                          f"train_loss={loss:.4f}, train_acc={acc:.4f}, "
                          f"dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}")

        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_acc = np.mean(epoch_train_accs)

        dev_logits = model(X_dev)
        dev_loss = loss_fn(dev_logits, y_dev)
        dev_acc = nn.metric.accuracy(dev_logits, y_dev)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)

        print(f"[{exp_name}] === Epoch {epoch} Summary === "
              f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.4f}, "
              f"dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            if save_dir:
                best_model_path = os.path.join(save_dir, f'{exp_name}_best.pickle')
                model.save_model(best_model_path)
                print(f"[{exp_name}] Best dev acc updated: {best_dev_acc:.4f}, model saved.")

    history['best_dev_acc'] = best_dev_acc
    return history, best_model_path


def plot_lr_comparison(histories, save_path=None):
    """
    Plot learning curves of multiple LR settings on the same figure.
    histories: dict {lr_value: history_dict}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'0.001': '#4C8BF5', '0.06': '#E3E37D', '0.5': '#E8453C'}

    for lr_str, hist in histories.items():
        color = colors.get(lr_str, '#333333')
        dev_iters = hist.get('iteration_dev_iters', None)

        # Loss curve
        train_iters = np.arange(len(hist['iteration_train_loss']))
        if dev_iters is None:
            dev_iters = np.arange(len(hist['iteration_dev_loss']))

        axes[0].plot(train_iters, hist['iteration_train_loss'],
                     color=color, alpha=0.4, linewidth=0.8)
        axes[0].plot(dev_iters, hist['iteration_dev_loss'],
                     color=color, linestyle='-', marker='o', markersize=3,
                     label=f"LR={lr_str}")

        # Accuracy curve
        axes[1].plot(train_iters, hist['iteration_train_acc'],
                     color=color, alpha=0.4, linewidth=0.8)
        axes[1].plot(dev_iters, hist['iteration_dev_acc'],
                     color=color, linestyle='-', marker='o', markersize=3,
                     label=f"LR={lr_str}")

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Dev Loss Comparison")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Dev Accuracy Comparison")
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Learning Rate Comparison on MLP", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved LR comparison plot to {save_path}")
    plt.close()


def main():
    save_dir = r'./experiment_results/lr_comparison'
    os.makedirs(save_dir, exist_ok=True)

    train_set, dev_set, test_set = load_mnist_data(normalize_type='minmax', seed=309)

    lr_configs = [
        {'name': 'MLP_lr_small', 'lr': 0.001},
        {'name': 'MLP_lr_moderate', 'lr': 0.06},
        {'name': 'MLP_lr_large', 'lr': 2.0},
    ]

    results = {}
    histories = {}

    for cfg in lr_configs:
        name = cfg['name']
        lr = cfg['lr']
        print(f"\n{'='*60}")
        print(f"Running: {name} (lr={lr})")
        print(f"{'='*60}")

        np.random.seed(309)
        model = nn.models.Model_MLP([784, 128, 10], 'ReLU', [1e-4, 1e-4])
        optimizer = nn.optimizer.SGD(init_lr=lr, model=model)
        scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=800, gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

        history, best_model_path = train_model(
            model, optimizer, loss_fn, train_set, dev_set,
            num_epochs=5, batch_size=32, scheduler=scheduler,
            log_iters=100, eval_interval=50,
            save_dir=save_dir, exp_name=name
        )

        # Evaluate on test set
        test_X, test_y = test_set
        test_logits = model(test_X)
        test_loss = loss_fn(test_logits, test_y)
        test_acc = nn.metric.accuracy(test_logits, test_y)
        history['test_loss'] = test_loss
        history['test_acc'] = test_acc
        print(f"[{name}] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        # Individual visualizations
        viz_dir = os.path.join(save_dir, name)
        os.makedirs(viz_dir, exist_ok=True)

        plot_learning_curves(
            history['iteration_train_loss'], history['iteration_dev_loss'],
            history['iteration_train_acc'], history['iteration_dev_acc'],
            dev_iters=history.get('iteration_dev_iters', None),
            save_path=os.path.join(viz_dir, 'learning_curves_iteration.png'),
            title=f"{name} - Learning Curves"
        )

        plot_epoch_learning_curves(
            history['train_loss'], history['dev_loss'],
            history['train_acc'], history['dev_acc'],
            save_path=os.path.join(viz_dir, 'learning_curves_epoch.png'),
            title=f"{name} - Epoch Curves"
        )

        dev_X, dev_y = dev_set
        plot_confusion_matrix(
            model, dev_X[:5000], dev_y[:5000],
            class_names=[str(i) for i in range(10)],
            save_path=os.path.join(viz_dir, 'confusion_matrix.png'),
            title=f"{name} - Confusion Matrix"
        )

        visualize_mlp_weights(
            model,
            save_path=os.path.join(viz_dir, 'weight_visualization.png'),
            title=f"{name} - First Layer Weights"
        )

        history_path = os.path.join(viz_dir, 'history.pickle')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        results[name] = {
            'best_dev_acc': history['best_dev_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc,
        }
        histories[str(lr)] = history

    # Combined comparison plot
    plot_lr_comparison(histories, save_path=os.path.join(save_dir, 'lr_comparison.png'))

    # Summary table
    print("\n" + "=" * 60)
    print("LEARNING RATE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<20} {'Best Dev Acc':>12} {'Test Loss':>12} {'Test Acc':>12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<20} {res['best_dev_acc']:>12.4f} {res['test_loss']:>12.4f} {res['test_acc']:>12.4f}")
    print("=" * 60)

    with open(os.path.join(save_dir, 'summary.pickle'), 'wb') as f:
        pickle.dump({'results': results, 'histories': histories}, f)


if __name__ == '__main__':
    main()
