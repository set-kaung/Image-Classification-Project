import os
import glob
import json
from datetime import datetime
from typing import List, Tuple

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
# Optional AVIF support (ignore if not installed)
try:
    import pillow_avif  # type: ignore  # noqa: F401
except Exception:
    pass

from main import SimpleCNN

# Configuration
TEST_ROOT = "test_dataset"
MODELS_DIR = "models"
EVAL_DIR = "evaluation"
PRED_IMG_DIR = os.path.join(EVAL_DIR, "predictions")
os.makedirs(PRED_IMG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tiff", ".tif"}

def load_best_checkpoint(models_dir: str) -> Tuple[SimpleCNN, List[str], dict]:
    best_candidates = glob.glob(os.path.join(models_dir, "best*.pth"))
    if not best_candidates:
        raise FileNotFoundError("No best*.pth checkpoint found. Train a model first.")
    best_path = max(best_candidates, key=os.path.getctime)
    ckpt = torch.load(best_path, map_location='cpu')
    model = SimpleCNN(num_classes=ckpt['num_classes'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['class_names'], ckpt | {"_path": best_path}

def collect_test_images(root: str):
    samples = []  # (path, true_class)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Test root not found: {root}")
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXT:
                samples.append((os.path.join(cls_dir, fname), cls))
    return samples

def annotate_and_save(image: Image.Image, predicted: str, true_label: str, confidence: float, dest_path: str):
    draw = ImageDraw.Draw(image)
    text = f"Pred: {predicted} ({confidence*100:.1f}%) | True: {true_label}"
    # Choose font
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    # Background rectangle
    text_bbox = draw.textbbox((5,5), text, font=font)
    draw.rectangle(text_bbox, fill=(0,0,0,160))
    color = (0, 200, 0) if predicted == true_label else (220, 20, 20)
    draw.text((8,8), text, font=font, fill=color)
    image.save(dest_path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, class_names, ckpt_meta = load_best_checkpoint(MODELS_DIR)
    model.to(device)
    print(f"Loaded best checkpoint: {os.path.basename(ckpt_meta['_path'])} | Val Acc Stored: {ckpt_meta.get('val_acc', 'NA')}%")
    print("Classes:", class_names)

    samples = collect_test_images(TEST_ROOT)
    if not samples:
        print("No test images found.")
        return
    print(f"Found {len(samples)} test images.")

    y_true = []
    y_pred = []
    per_image_results = []  # We'll keep only path + true/pred labels (no probabilities as requested)

    for idx, (path, true_cls) in enumerate(samples, 1):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_label = class_names[pred_idx]
            conf = float(probs[pred_idx])
        y_true.append(true_cls)
        y_pred.append(pred_label)
        per_image_results.append({
            'image_path': path,
            'true_class': true_cls,
            'predicted_class': pred_label
        })
        # Annotated image filename
        base = os.path.splitext(os.path.basename(path))[0]
        out_name = f"{base}__pred-{pred_label}_true-{true_cls}_conf-{conf*100:.1f}.png"
        dest = os.path.join(PRED_IMG_DIR, out_name)
        annotate_and_save(img.resize((256,256)), pred_label, true_cls, conf, dest)
        print(f"[{idx}/{len(samples)}] {path} -> {pred_label} ({conf*100:.1f}%)")

    # Confusion matrix
    labels_sorted = sorted(set(class_names) | set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=45, ha='right')
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Test Set)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cm_path = os.path.join(EVAL_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

    # Compute metrics: sensitivity (recall), specificity, accuracy, precision, f1-score
    total = cm.sum()
    overall_accuracy = float(np.trace(cm) / total) if total else 0.0

    per_class_metrics = {}
    recalls = []
    precisions = []
    specificities = []
    f1s = []
    supports = []

    for i, cls in enumerate(labels_sorted):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - (tp + fn + fp)
        support = cm[i, :].sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

        per_class_metrics[cls] = {
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'support': int(support),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'recall': sensitivity,
            'precision': precision,
            'f1': f1
        }
        recalls.append(sensitivity)
        precisions.append(precision)
        specificities.append(specificity)
        f1s.append(f1)
        supports.append(support)

    supports_arr = np.array(supports)
    weight_sum = supports_arr.sum() if supports_arr.sum() else 1

    macro_metrics = {
        'sensitivity': float(np.mean(recalls) if recalls else 0.0),
        'specificity': float(np.mean(specificities) if specificities else 0.0),
        'precision': float(np.mean(precisions) if precisions else 0.0),
        'f1': float(np.mean(f1s) if f1s else 0.0)
    }
    weighted_metrics = {
        'sensitivity': float(np.average(recalls, weights=supports_arr) if recalls else 0.0),
        'specificity': float(np.average(specificities, weights=supports_arr) if specificities else 0.0),
        'precision': float(np.average(precisions, weights=supports_arr) if precisions else 0.0),
        'f1': float(np.average(f1s, weights=supports_arr) if f1s else 0.0)
    }

    # Aggregate JSON output
    summary = {
        'timestamp': datetime.now().isoformat() + 'Z',
        'checkpoint_used': os.path.basename(ckpt_meta['_path']),
        'classes': class_names,
        'num_images': len(per_image_results),
        'confusion_matrix_labels': labels_sorted,
        'confusion_matrix': cm.tolist(),
        'overall': {
            'accuracy': overall_accuracy
        },
        'per_class': per_class_metrics,
        'macro_averages': macro_metrics,
        'weighted_averages': weighted_metrics,
        'per_image_results': per_image_results
    }
    summary_path = os.path.join(EVAL_DIR, 'test_set_evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {summary_path}")

    print("Done.")

if __name__ == '__main__':
    main()
