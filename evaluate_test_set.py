import torch
from main import SimpleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from metrics_utils import compute_confusion_metrics
import matplotlib.pyplot as plt
import math
import numpy as np

MODEL_PATH = "models/20250909_101742_best.pth"
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

    n_images = len(sample_paths)
    cols = min(6, n_images)
    rows = math.ceil(n_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel().tolist()
    else:  
        axes_flat = [axes]
    from PIL import Image
    for i, ax in enumerate(axes_flat):
        if i >= n_images:
            ax.axis('off')
            continue
        path = sample_paths[i]
        try:
            img = Image.open(path).convert('RGB')
            ax.imshow(img)
        except Exception:
            ax.text(0.5,0.5,'[load fail]', ha='center', va='center')
        t = labels_np[i]
        p = preds_np[i]
        conf = float(probs_full[i, p]) * 100.0
        color = 'green' if t == p else 'red'
        ax.set_title(f"T:{class_names[t]}\nP:{class_names[p]} {conf:.1f}%", fontsize=8, color=color)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(h_pad=0.6)
    plt.show()


if __name__ == "__main__":
    evaluate()
