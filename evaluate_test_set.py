import torch
from main import SimpleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from metrics_utils import compute_confusion_metrics
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from datetime import datetime
import json

MODEL_PATH = "models/20250910_183320_best.pth"
TEST_DATASET_PATH = "test_dataset"

if torch.backends.mps.is_available(): 
    DEVICE = torch.device("mps") 
elif torch.cuda.is_available(): 
    DEVICE = torch.device("cuda") 
else: 
    DEVICE = torch.device("cpu")


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in ckpt:
        num_classes = ckpt.get("num_classes", 5)
        model = SimpleCNN(num_classes=num_classes)
        state = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] Missing keys: {missing} | Unexpected keys: {unexpected}")
        class_names = ckpt.get("class_names")
    else:
        state = ckpt
        num_classes = 5
        model = SimpleCNN(num_classes=num_classes)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] Missing keys: {missing} | Unexpected keys: {unexpected}")
        class_names = None
    model.to(DEVICE)
    model.eval()
    return model, class_names

def get_test_loader():
    transform = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
    )
    test_ds = datasets.ImageFolder(TEST_DATASET_PATH, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    return test_loader


def evaluate():
    model, class_names_ckpt = load_model()
    test_loader = get_test_loader()
    ds = test_loader.dataset
    class_names_ds = ds.classes
    if class_names_ckpt and len(class_names_ckpt) == len(class_names_ds):
        class_names = class_names_ckpt
    else:
        class_names = class_names_ds

    all_preds = []
    all_labels = []
    all_probs = []
    sample_paths = [p for p,_ in ds.samples]
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(1)
            all_preds.append(preds)
            all_labels.append(y)
            all_probs.append(probs)
    if not all_preds:
        print("No data.")
        return

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    probs_full = torch.cat(all_probs)
    preds_np = preds.numpy()
    labels_np = labels.numpy()

    cm = confusion_matrix(labels_np, preds_np, labels=list(range(len(class_names))))
    metrics = compute_confusion_metrics(cm, class_names)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('evaluation', timestamp)
    pred_dir = os.path.join(base_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    print(f"Saving evaluation artifacts to: {base_dir}")

    print("\nPer-class metrics:")
    header = f"{'Class':<12} {'Prec%':>8} {'Rec%':>8} {'F1%':>8} {'Support':>8}"
    print(header)
    print('-' * len(header))
    for row in metrics['per_class']:
        print(f"{row['label']:<12} {row['precision']*100:8.2f} {row['recall']*100:8.2f} {row['f1']*100:8.2f} {row['support']:8d}")

    print("\nMacro:")
    print(f"Accuracy:        {metrics['accuracy']*100:.2f}%")
    print(f"Precision(macro): {metrics['macro_precision']*100:.2f}%")
    print(f"Recall(macro):    {metrics['macro_recall']*100:.2f}%")
    print(f"F1(macro):        {metrics['macro_f1']*100:.2f}%")

    # Save metrics JSON
    metrics_out = {
        'timestamp': timestamp,
        'classes': class_names,
        'per_class': metrics['per_class'],
        'macro': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['macro_precision'],
            'recall': metrics['macro_recall'],
            'f1': metrics['macro_f1']
        }
    }
    with open(os.path.join(base_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # Confusion matrix figure (raw counts)
    fig_cm, ax_cm = plt.subplots(figsize=(6,5))
    im = ax_cm.imshow(cm, cmap='Blues')
    ax_cm.set_title('Confusion Matrix')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels(class_names, rotation=45, ha='right')
    ax_cm.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=9)
    fig_cm.colorbar(im, ax=ax_cm, shrink=0.75)
    plt.tight_layout()
    cm_path = os.path.join(base_dir, 'confusion_matrix.png')
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)

    # Normalized confusion matrix (optional)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm[np.isnan(cm_norm)] = 0.0
    fig_cm_n, ax_cm_n = plt.subplots(figsize=(6,5))
    imn = ax_cm_n.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax_cm_n.set_title('Confusion Matrix (Normalized)')
    ax_cm_n.set_xlabel('Predicted')
    ax_cm_n.set_ylabel('True')
    ax_cm_n.set_xticks(range(len(class_names)))
    ax_cm_n.set_yticks(range(len(class_names)))
    ax_cm_n.set_xticklabels(class_names, rotation=45, ha='right')
    ax_cm_n.set_yticklabels(class_names)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax_cm_n.text(j, i, f"{cm_norm[i, j]*100:.1f}", ha='center', va='center', color='black', fontsize=8)
    fig_cm_n.colorbar(imn, ax=ax_cm_n, shrink=0.75)
    plt.tight_layout()
    cm_norm_path = os.path.join(base_dir, 'confusion_matrix_normalized.png')
    fig_cm_n.savefig(cm_norm_path, dpi=150)
    plt.close(fig_cm_n)

    # Save each image with prediction overlay
    from PIL import Image
    for i, path in enumerate(sample_paths):
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue
        t = labels_np[i]
        p = preds_np[i]
        conf = float(probs_full[i, p]) * 100.0
        fname = f"{i+1:03d}__pred-{class_names[p]}_true-{class_names[t]}_conf-{conf:.1f}.png"
        out_path = os.path.join(pred_dir, fname)
        # Use matplotlib for quick overlay (keeps code simple)
        fig_single, ax_single = plt.subplots(figsize=(3,3))
        ax_single.imshow(img)
        color = 'green' if t == p else 'red'
        ax_single.set_title(f"Pred: {class_names[p]} ({conf:.1f}%)\nTrue: {class_names[t]}", color=color, fontsize=9)
        ax_single.axis('off')
        plt.tight_layout()
        fig_single.savefig(out_path, dpi=120)
        plt.close(fig_single)

    print(f"Saved per-image predictions to {pred_dir}")
    print(f"Saved confusion matrices to {base_dir}")
    print("Done.")


if __name__ == "__main__":
    evaluate()
