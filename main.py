import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import json
from datetime import datetime
import random
from PIL import Image, ImageFile, ImageOps
import warnings
import argparse
from time import process_time

Image.MAX_IMAGE_PIXELS = 300_000_000 
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress only the specific palette transparency warning (optional)
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes"
)

def safe_image_loader(path: str):
    """Robust image loader that:
    - Opens image safely (handles truncated files)
    - Preserves orientation via EXIF
    - Converts palette / alpha images to RGB composited on white
    - Avoids PIL transparency warning by going through RGBA first
    - Downscales extremely large images to a max side (soft cap)
    - Falls back to a blank image on failure
    """
    max_side = 4096  # hard ceiling to avoid huge memory spikes
    try:
        with Image.open(path) as im:
            # EXIF orientation
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            mode = im.mode
            if mode in ("P", "LA", "RGBA"):
                # Ensure RGBA then composite onto white
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                alpha = im.split()[-1]
                bg.paste(im, mask=alpha)
                im = bg
            else:
                im = im.convert("RGB")
            # Large image soft downscale (keep aspect)
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            return im
    except Exception as e:
        print(f"[WARN] safe_image_loader failed for {path}: {e}")
        return Image.new("RGB", (256, 256), (255, 255, 255))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x



if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    parser = argparse.ArgumentParser(description="Train SimpleCNN")
    parser.add_argument('--batch', type=int, default=48, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    LR = 1e-3
    IMAGE_SIZE = 256
    if torch.backends.mps.is_available(): 
        DEVICE = torch.device("mps") 
        print("mps")
    elif torch.cuda.is_available(): 
        DEVICE = torch.device("cuda")
        print("cuda") 
    else: 
        DEVICE = torch.device("cpu")
        print("cpu")

    # DataLoader performance knobs
    NUM_WORKERS = 2  if (DEVICE.type == 'cuda') else 0 
    PIN_MEMORY = (DEVICE.type == 'cuda')  # only helpful for CUDA
    PERSISTENT = NUM_WORKERS > 0

   
   
    def _safe_downscale(img):
        if img.width * img.height > 20_000_000:
            img = img.copy()
            img.thumbnail((2048, 2048)) 
        return img

    train_transform = transforms.Compose([
        transforms.Lambda(_safe_downscale),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root="dataset/train", transform=train_transform, loader=safe_image_loader)
    val_dataset = datasets.ImageFolder(root="dataset/val", transform=val_transform, loader=safe_image_loader)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT,
    )

    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs("models", exist_ok=True)

    with open("class_labels.json", "w") as f:
        json.dump(class_names, f, indent=4)
    print("Saved class labels to class_labels.json:", class_names)

    best_val_acc = 0.0
    best_model_path = None

    print(f"Starting training on {DEVICE} for {EPOCHS} epochs | Batch: {BATCH_SIZE} | Classes: {class_names}")
    
    for epoch in range(1, EPOCHS + 1):
        start = process_time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
        train_loss = running_loss / total if total else 0.0
        train_acc = 100.0 * correct / total if total else 0.0

        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_accum += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss_accum / val_total if val_total else 0.0
        val_acc = 100.0 * val_correct / val_total if val_total else 0.0
        end = process_time()
        print(f"End of epoch {epoch:02d}, time taken: {end-start}")
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
                best_val_acc = val_acc
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                best_model_path = f"models/{timestamp}_best.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'num_classes': num_classes,
                    'class_names': class_names,
                    'val_acc': val_acc,
                    'epoch': epoch
                }, best_model_path)
                print(f"  -> New best model saved: {best_model_path}")
    last_model_path = "models/last.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names,
        'val_acc': best_val_acc,  # store best val acc observed during run
        'epoch': EPOCHS
    }, last_model_path)
    print(f"Last model saved to {last_model_path}")
    if best_model_path:
        print(f"Best model was {best_model_path} (Val Acc: {best_val_acc:.2f}%)")
    else:
        print("No validation data available to determine best model.")

    print("Training complete.")