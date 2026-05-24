"""
Unified experiment script for MNIST classification.
Supports: MLP baseline, CNN, and extra directions (standardization, momentum).
"""

import numpy as np
import mynn as nn
from visualization import (plot_learning_curves, plot_confusion_matrix,
                            visualize_mlp_weights, visualize_cnn_kernels,
                            plot_epoch_learning_curves)
import os
import pickle
from struct import unpack
import gzip

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_mnist_data(normalize_type='minmax', cnn_format=False, seed=309):
    """
    Load MNIST dataset.
    normalize_type: 'minmax' -> [0, 1], 'standard' -> zero mean, unit variance
    cnn_format: if True, reshape images to [N, 1, 28, 28]
    Returns: (train_set, dev_set, test_set)
    each set is a tuple (X, y)
    """
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

    # Shuffle train set and split into train/dev
    np.random.seed(seed)
    idx = np.random.permutation(np.arange(train_imgs.shape[0]))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    # Normalization
    if normalize_type == 'minmax':
        train_imgs = train_imgs / 255.0
        valid_imgs = valid_imgs / 255.0
        test_imgs = test_imgs / 255.0
    elif normalize_type == 'standard':
        # Compute mean and std on training set only
        mean = train_imgs.mean(axis=0, keepdims=True)
        std = train_imgs.std(axis=0, keepdims=True) + 1e-8
        train_imgs = (train_imgs - mean) / std
        valid_imgs = (valid_imgs - mean) / std
        test_imgs = (test_imgs - mean) / std
    else:
        raise ValueError(f"Unknown normalize_type: {normalize_type}")

    # CNN format
    if cnn_format:
        train_imgs = train_imgs.reshape(-1, 1, 28, 28)
        valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)

    return (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_model(model, optimizer, loss_fn, train_set, dev_set,
                num_epochs=5, batch_size=32, scheduler=None,
                log_iters=100, eval_interval=50, save_dir=None, exp_name='exp'):
    """
    Custom training loop that records per-epoch metrics.
    Returns: history dict with lists of metrics.
    """
    X_train, y_train = train_set
    X_dev, y_dev = dev_set

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    history = {
        'train_loss': [], 'train_acc': [],
        'dev_loss': [], 'dev_acc': [],
        'iteration_train_loss': [], 'iteration_train_acc': [],
        'iteration_dev_loss': [], 'iteration_dev_acc': [],
        'iteration_dev_iters': [],  # record which iteration dev was evaluated
    }

    best_dev_acc = 0.0
    best_model_path = None

    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    for epoch in range(num_epochs):
        # Shuffle each epoch
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

            # Skip empty batches
            if batch_X.shape[0] == 0:
                continue

            # Forward
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            acc = nn.metric.accuracy(logits, batch_y)

            epoch_train_losses.append(loss)
            epoch_train_accs.append(acc)

            history['iteration_train_loss'].append(loss)
            history['iteration_train_acc'].append(acc)

            # Backward + optimize
            loss_fn.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Evaluate on dev periodically
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

        # Epoch-level averages
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_acc = np.mean(epoch_train_accs)

        # Final dev eval for this epoch
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


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------

def run_experiment(config):
    """
    config dict keys:
        - name: experiment name
        - model_type: 'mlp' or 'cnn'
        - normalize: 'minmax' or 'standard'
        - optimizer_type: 'sgd' or 'momentum'
        - lr: learning rate
        - batch_size
        - num_epochs
        - seed
        - lambda_list: weight decay list (optional)
    """
    name = config['name']
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"{'='*60}")

    np.random.seed(config.get('seed', 309))

    # Load data
    cnn_format = config['model_type'] == 'cnn'
    train_set, dev_set, test_set = load_mnist_data(
        normalize_type=config['normalize'],
        cnn_format=cnn_format,
        seed=config.get('seed', 309)
    )

    # Build model
    if config['model_type'] == 'mlp':
        size_list = config.get('size_list', [784, 128, 10])
        lambda_list = config.get('lambda_list', None)
        model = nn.models.Model_MLP(size_list, 'ReLU', lambda_list)
    elif config['model_type'] == 'cnn':
        lambda_list = config.get('lambda_list', None)
        model = nn.models.Model_CNN(lambda_list)
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    # Build optimizer
    lr = config['lr']
    if config['optimizer_type'] == 'sgd':
        optimizer = nn.optimizer.SGD(init_lr=lr, model=model)
    elif config['optimizer_type'] == 'momentum':
        mu = config.get('mu', 0.9)
        optimizer = nn.optimizer.MomentGD(init_lr=lr, model=model, mu=mu)
    else:
        raise ValueError(f"Unknown optimizer_type: {config['optimizer_type']}")

    # Scheduler
    step_size = config.get('scheduler_step_size', 800)
    gamma = config.get('scheduler_gamma', 0.5)
    scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    # Loss
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    # Train
    save_dir = config.get('save_dir', r'./experiment_results')
    history, best_model_path = train_model(
        model, optimizer, loss_fn, train_set, dev_set,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        scheduler=scheduler,
        log_iters=config.get('log_iters', 100),
        save_dir=save_dir,
        exp_name=name
    )

    # Load best model for evaluation and visualization
    if best_model_path and os.path.exists(best_model_path):
        if config['model_type'] == 'mlp':
            model.load_model(best_model_path)
        elif config['model_type'] == 'cnn':
            model.load_model(best_model_path)

    # Evaluate on test set
    test_X, test_y = test_set
    test_logits = model(test_X)
    test_loss = loss_fn(test_logits, test_y)
    test_acc = nn.metric.accuracy(test_logits, test_y)
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    print(f"[{name}] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Visualizations
    viz_dir = os.path.join(save_dir, name)
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Learning curves (per iteration)
    plot_learning_curves(
        history['iteration_train_loss'], history['iteration_dev_loss'],
        history['iteration_train_acc'], history['iteration_dev_acc'],
        dev_iters=history.get('iteration_dev_iters', None),
        save_path=os.path.join(viz_dir, 'learning_curves_iteration.png'),
        title=f"{name} - Learning Curves (per Iteration)"
    )

    # 2. Learning curves (per epoch)
    plot_epoch_learning_curves(
        history['train_loss'], history['dev_loss'],
        history['train_acc'], history['dev_acc'],
        save_path=os.path.join(viz_dir, 'learning_curves_epoch.png'),
        title=f"{name} - Learning Curves (per Epoch)"
    )

    # 3. Confusion matrix (on dev set)
    dev_X, dev_y = dev_set
    # Use a subset for speed if too large
    cm_subset_size = min(5000, dev_X.shape[0])
    plot_confusion_matrix(
        model, dev_X[:cm_subset_size], dev_y[:cm_subset_size],
        class_names=[str(i) for i in range(10)],
        save_path=os.path.join(viz_dir, 'confusion_matrix.png'),
        title=f"{name} - Confusion Matrix (Dev)"
    )

    # 4. Weight / Kernel visualization
    if config['model_type'] == 'mlp':
        visualize_mlp_weights(
            model,
            save_path=os.path.join(viz_dir, 'weight_visualization.png'),
            title=f"{name} - First Layer Weights"
        )
    elif config['model_type'] == 'cnn':
        visualize_cnn_kernels(
            model,
            save_path=os.path.join(viz_dir, 'kernel_visualization.png'),
            title=f"{name} - Convolution Kernels"
        )

    # 5. Misclassified examples (Error Analysis)
    from visualization import visualize_misclassified_examples
    dev_X, dev_y = dev_set
    mis_subset_size = min(1000, dev_X.shape[0])
    visualize_misclassified_examples(
        model, dev_X[:mis_subset_size], dev_y[:mis_subset_size],
        class_names=[str(i) for i in range(10)],
        n_show=16,
        save_path=os.path.join(viz_dir, 'misclassified_examples.png'),
        title=f"{name} - Misclassified Examples"
    )

    # Save history
    history_path = os.path.join(viz_dir, 'history.pickle')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"[{name}] History saved to {history_path}")

    return history


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    experiments = [
        # 1. MLP Baseline (moderate learning rate, comparable params to CNN)
        {
            'name': 'MLP_baseline',
            'model_type': 'mlp',
            'size_list': [784, 128, 10],
            'normalize': 'minmax',
            'optimizer_type': 'sgd',
            'lr': 0.06,
            'batch_size': 32,
            'num_epochs': 5,
            'seed': 309,
            'lambda_list': [1e-4, 1e-4],
            'scheduler_step_size': 800,
            'scheduler_gamma': 0.5,
            'log_iters': 100,
            'save_dir': r'./experiment_results',
        },
        # 2. CNN (controlled variables: same lr, epochs, batch, seed, scheduler)
        {
            'name': 'CNN',
            'model_type': 'cnn',
            'normalize': 'minmax',
            'optimizer_type': 'sgd',
            'lr': 0.06,
            'batch_size': 32,
            'num_epochs': 5,
            'seed': 309,
            'lambda_list': [1e-4, 1e-4, 1e-4, 1e-4],  # 2 conv + 2 linear
            'scheduler_step_size': 800,
            'scheduler_gamma': 0.5,
            'log_iters': 100,
            'save_dir': r'./experiment_results',
        },
    ]

    results = {}
    for config in experiments:
        history = run_experiment(config)
        results[config['name']] = history

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<25} {'Best Dev Acc':>12} {'Test Loss':>12} {'Test Acc':>12}")
    print("-" * 70)
    for name, hist in results.items():
        print(f"{name:<25} {hist['best_dev_acc']:>12.4f} {hist.get('test_loss', 0):>12.4f} {hist.get('test_acc', 0):>12.4f}")
    print("=" * 70)

    # Save results summary
    with open(r'./experiment_results/summary.pickle', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
