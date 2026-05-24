"""
Load a saved model and evaluate on the MNIST test set.
Generates confusion matrix and misclassified examples visualization.

Usage:
    python test_model.py --model_path path/to/model.pickle --model_type mlp
    python test_model.py --model_path path/to/model.pickle --model_type cnn
"""

import argparse
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import os
from visualization import plot_confusion_matrix, visualize_misclassified_examples


def load_test_data(cnn_format=False):
    """Load MNIST test data."""
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs / 255.0

    if cnn_format:
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)

    return test_imgs, test_labs


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on MNIST test set.")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved .pickle model file')
    parser.add_argument('--model_type', type=str, required=True, choices=['mlp', 'cnn'],
                        help='Model type: mlp or cnn')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save visualization outputs')
    args = parser.parse_args()

    # Load model
    if args.model_type == 'mlp':
        model = nn.models.Model_MLP()
    elif args.model_type == 'cnn':
        model = nn.models.Model_CNN()
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.load_model(args.model_path)
    print(f"Loaded model from {args.model_path}")

    # Load test data
    cnn_format = args.model_type == 'cnn'
    test_X, test_y = load_test_data(cnn_format=cnn_format)
    print(f"Test set: {test_X.shape[0]} samples")

    # Evaluate
    logits = model(test_X)
    test_acc = nn.metric.accuracy(logits, test_y)
    print(f"Test accuracy: {test_acc:.4f}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    exp_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Confusion matrix
    cm = plot_confusion_matrix(
        model, test_X, test_y,
        class_names=[str(i) for i in range(10)],
        save_path=os.path.join(args.output_dir, f'{exp_name}_confusion_matrix.png'),
        title=f"{exp_name} - Confusion Matrix (Test)"
    )

    # Misclassified examples
    visualize_misclassified_examples(
        model, test_X, test_y,
        class_names=[str(i) for i in range(10)],
        n_show=16,
        save_path=os.path.join(args.output_dir, f'{exp_name}_misclassified.png'),
        title=f"{exp_name} - Misclassified Examples"
    )

    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
