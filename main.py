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
from PIL import Image, ImageFile
import warnings
import argparse
import time
from utils import safe_image_loader,safe_downscale

Image.MAX_IMAGE_PIXELS = 300_000_000 
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes"
)

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
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--stop-after-lr-drops', type=int, default=0, help='Stop training after this many LR reductions (0=ignore)')
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

   
    train_transform = transforms.Compose([
        transforms.Lambda(safe_downscale),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.01),
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
   
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    WEIGHT_DECAY = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)

    os.makedirs("models", exist_ok=True)

    with open("class_labels.json", "w") as f:
        json.dump(class_names, f, indent=4)
    print("Saved class labels to class_labels.json:", class_names)

    best_val_acc = 0.0
    best_model_path = None
    best_val_loss = float('inf')

    def save_checkpoint(path, model, num_classes, class_names, val_acc, best_val_loss, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'class_names': class_names,
            'val_acc': val_acc,
            'best_val_loss': best_val_loss,
            'epoch': epoch
        }, path)
        print(f"Saved checkpoint: {path}")

    start_epoch = 1
    lr_drop_target = args.stop_after_lr_drops
    lr_drop_count = 0

    if args.resume and os.path.isfile(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=DEVICE)
            state_dict = ckpt.get('model_state_dict', ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[RESUME WARN] missing: {missing} unexpected: {unexpected}")
            prev_epoch = ckpt.get('epoch')
            if isinstance(prev_epoch, int):
                start_epoch = prev_epoch + 1
            best_val_acc = ckpt.get('val_acc', best_val_acc)
            best_val_loss = ckpt.get('best_val_loss', best_val_loss)
            best_model_path = args.resume
            print(f"Resumed from {args.resume} (epoch={prev_epoch}, best_val_acc={best_val_acc:.2f}%)")
        except Exception as e:
            print(f"[RESUME ERROR] Failed to load {args.resume}: {e}")

    print(f"Starting training on {DEVICE} for {EPOCHS} epochs | Batch: {BATCH_SIZE} | Classes: {class_names}")
    
    for epoch in range(start_epoch, EPOCHS + 1):
        start = time.perf_counter()
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
    
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        elif DEVICE.type == 'mps':
            try:
                torch.mps.synchronize()
            except Exception:
                pass
        end = time.perf_counter()
        print(f"End of epoch {epoch:02d}, time (s): {end-start:.3f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr - 1e-12:
            lr_drop_count += 1
            print(f"  -> LR reduced: {prev_lr:.2e} -> {new_lr:.2e} (drops: {lr_drop_count}{'/' + str(lr_drop_target) if lr_drop_target else ''})")
            if lr_drop_target and lr_drop_count >= lr_drop_target:
                print(f"Reached target LR drops ({lr_drop_target}); stopping early after epoch {epoch}.")
                timestamp_es = datetime.now().strftime('%Y%m%d_%H%M%S')
                es_path = f"models/{timestamp_es}_earlystop_last.pth"
                save_checkpoint(es_path, model, num_classes, class_names, best_val_acc, best_val_loss, epoch)
                print(f"Early-stop model saved to {es_path}")
                break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  -> New lowest val loss: {best_val_loss:.4f}")

        if val_acc > best_val_acc:
                best_val_acc = val_acc
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                best_model_path = f"models/{timestamp}_best.pth"
                save_checkpoint(best_model_path, model, num_classes, class_names, val_acc, best_val_loss, epoch)
                print(f"  -> New best model saved: {best_model_path}")
    timestamp_end = datetime.now().strftime('%Y%m%d_%H%M%S')
    last_model_path = f"models/{timestamp_end}_last.pth"
    save_checkpoint(last_model_path, model, num_classes, class_names, best_val_acc, best_val_loss, EPOCHS)
    print(f"Last model saved to {last_model_path}")
    if best_model_path and os.path.isfile(best_model_path):
        print(f"Best model was {best_model_path} (Val Acc: {best_val_acc:.2f}%, Best Val Loss: {best_val_loss if best_val_loss != float('inf') else 'n/a'})")
    else:
        if best_val_acc > 0:
            print(f"Best validation accuracy during run: {best_val_acc:.2f}% (original best model file not tracked this session)")
        else:
            print("No validation data available to determine best model.")

    print("Training complete.")