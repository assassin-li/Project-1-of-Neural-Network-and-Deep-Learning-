"""
Standalone weight / kernel visualization script.
Usage:
    python visualize_weights.py --model_path path/to/model.pickle --model_type mlp --output_path weights.png
    python visualize_weights.py --model_path path/to/model.pickle --model_type cnn --output_path kernels.png
"""

import argparse
import mynn as nn
from visualization import visualize_mlp_weights, visualize_cnn_kernels


def main():
    parser = argparse.ArgumentParser(description="Visualize model weights or CNN kernels.")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved .pickle model file')
    parser.add_argument('--model_type', type=str, required=True, choices=['mlp', 'cnn'],
                        help='Model type: mlp or cnn')
    parser.add_argument('--output_path', type=str, default='visualization.png',
                        help='Output image path')
    parser.add_argument('--title', type=str, default=None,
                        help='Title for the plot')
    args = parser.parse_args()

    if args.model_type == 'mlp':
        model = nn.models.Model_MLP()
        model.load_model(args.model_path)
        title = args.title or "MLP Weight Visualization"
        visualize_mlp_weights(model, save_path=args.output_path, title=title)
    elif args.model_type == 'cnn':
        model = nn.models.Model_CNN()
        model.load_model(args.model_path)
        title = args.title or "CNN Convolution Kernels"
        visualize_cnn_kernels(model, save_path=args.output_path, title=title)

    print(f"Visualization saved to {args.output_path}")


if __name__ == '__main__':
    main()
