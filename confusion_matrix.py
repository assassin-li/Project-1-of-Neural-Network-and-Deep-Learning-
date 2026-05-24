"""
Generate confusion matrix and visualize misclassified examples.
Run this after training a model and saving it to a .pickle file.
"""
import numpy as np
import mynn as nn
from struct import unpack
import gzip
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.random.seed(309)

# --------------------- Load Test Data ---------------------
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / 255.0

# --------------------- Load Model ---------------------
# Change this path to your saved model
model_path = r'.\best_models\best_model.pickle'

# For CNN, reshape input
test_imgs_cnn = test_imgs.reshape(-1, 1, 28, 28)

# Example: load CNN model
model = nn.models.Model_CNN()
model.load_model(model_path)

# Forward pass
logits = model(test_imgs_cnn)
preds = np.argmax(logits, axis=1)

# --------------------- Confusion Matrix ---------------------
cm = confusion_matrix(test_labs, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on MNIST Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Print per-class accuracy
for i in range(10):
    correct = cm[i, i]
    total = cm[i, :].sum()
    print(f"Digit {i}: {correct}/{total} = {correct/total:.4f}")

# --------------------- Misclassified Examples ---------------------
misclassified_idx = np.where(preds != test_labs)[0]
print(f"\nTotal misclassified: {len(misclassified_idx)} / {len(test_labs)}")

# Sample up to 20 misclassified images
n_show = min(20, len(misclassified_idx))
sample_idx = np.random.choice(misclassified_idx, size=n_show, replace=False)

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
axes = axes.flatten()

for i, idx in enumerate(sample_idx):
    img = test_imgs[idx].reshape(28, 28)
    true_label = test_labs[idx]
    pred_label = preds[idx]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"{true_label} -> {pred_label}", color='red' if true_label != pred_label else 'green')
    axes[i].axis('off')

plt.suptitle("Misclassified Examples")
plt.tight_layout()
plt.savefig('misclassified_examples.png', dpi=150)
plt.show()
