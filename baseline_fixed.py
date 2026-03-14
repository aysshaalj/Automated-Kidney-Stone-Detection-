import os
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 0) Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 1) Config (values can be changed here)
@dataclass
class CFG:
    data_dir: str = r"./Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection A Resource for Deep Learning Research"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # NEW: val + test splits (like your HHO code)
    val_split: float = 0.15
    test_split: float = 0.15

    num_workers: int = 2
    seed: int = 42
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # NEW: outputs go into a unique run folder (no overwrite)
    out_dir: str = "runs_baseline"


# 1.1) Unique run folder helper (prevents overwriting pytorch files)
def make_run_dir(base: str = "runs_baseline", prefix: str = "baseline_cnn") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# 2) Model: simple CNN
class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x


# 3) Metrics: Accuracy, Precision, Recall, F1 (macro)
@torch.no_grad()
def compute_metrics(logits, y_true, num_classes: int):
    preds = torch.argmax(logits, dim=1)

    correct = (preds == y_true).sum().item()
    total = y_true.numel()
    acc = correct / total if total else 0.0

    precision_list, recall_list, f1_list = [], [], []
    for c in range(num_classes):
        tp = ((preds == c) & (y_true == c)).sum().item()
        fp = ((preds == c) & (y_true != c)).sum().item()
        fn = ((preds != c) & (y_true == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    macro_precision = float(np.mean(precision_list))
    macro_recall = float(np.mean(recall_list))
    macro_f1 = float(np.mean(f1_list))

    return acc, macro_precision, macro_recall, macro_f1


# 4) Train / Eval loops
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    running_loss = 0.0
    all_logits, all_y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        all_logits.append(logits)
        all_y.append(y)

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)

    loss_val = running_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(all_logits, all_y, num_classes)
    return loss_val, acc, prec, rec, f1


# 5) Build dataloaders with train/val/test split (like HHO)
def build_dataloaders(cfg: CFG):
    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data_dir = Path(cfg.data_dir)

    # Base dataset for indices/classes
    base_ds = datasets.ImageFolder(root=str(data_dir), transform=None)
    class_names = base_ds.classes
    num_classes = len(class_names)

    # Make deterministic shuffled indices
    indices = list(range(len(base_ds)))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(indices)

    test_len = int(len(indices) * cfg.test_split)
    val_len = int(len(indices) * cfg.val_split)
    train_len = len(indices) - val_len - test_len
    if train_len <= 0:
        raise ValueError("Not enough data for requested splits. Reduce val_split/test_split.")

    test_idx = indices[:test_len]
    val_idx = indices[test_len:test_len + val_len]
    train_idx = indices[test_len + val_len:]

    # Two copies so train has aug, val/test do not
    train_ds_full = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
    eval_ds_full = datasets.ImageFolder(root=str(data_dir), transform=eval_tfms)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(eval_ds_full, val_idx)
    test_ds = Subset(eval_ds_full, test_idx)

    workers = max(0, cfg.num_workers)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False, persistent_workers=(workers > 0))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=workers, pin_memory=False, persistent_workers=(workers > 0))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=workers, pin_memory=False, persistent_workers=(workers > 0))

    return train_loader, val_loader, test_loader, num_classes, class_names, len(train_ds), len(val_ds), len(test_ds)


# 6) Main
def main():
    cfg = CFG()
    print("Device:", cfg.device)
    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir.resolve()}")

    if cfg.val_split + cfg.test_split >= 0.5:
        raise ValueError("val_split + test_split is too large. Keep it <= 0.3 for small datasets.")

    # Create unique run folder (NO OVERWRITES)
    run_dir = make_run_dir(cfg.out_dir, prefix="baseline_cnn")
    print(f"Run output directory: {run_dir.resolve()}")

    ckpt_path = run_dir / "best_model.pt"
    history_csv = run_dir / "history.csv"
    summary_path = run_dir / "summary.txt"

    # Data
    train_loader, val_loader, test_loader, num_classes, class_names, n_train, n_val, n_test = build_dataloaders(cfg)
    print("Classes:", class_names, "num_classes =", num_classes)
    print(f"Train size: {n_train} | Val size: {n_val} | Test size: {n_test}")

    # Model
    model = BaselineCNN(num_classes=num_classes).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Best-epoch tracking by VAL F1
    history = []
    best_f1 = -1.0
    best_epoch = -1
    best_val_metrics = None  # (val_loss, acc, prec, rec, f1)

    print("\nTraining baseline model...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_loss, acc, prec, rec, f1 = evaluate(model, val_loader, criterion, cfg.device, num_classes)

        history.append((epoch, train_loss, val_loss, acc, prec, rec, f1))

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_acc={acc:.4f} val_prec={prec:.4f} val_rec={rec:.4f} val_f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_val_metrics = (val_loss, acc, prec, rec, f1)
            torch.save(
                {"model_state": model.state_dict(),
                 "classes": class_names,
                 "cfg": cfg.__dict__,
                 "best_epoch": best_epoch,
                 "best_val_metrics": best_val_metrics,
                 "best_val_f1": best_f1},
                ckpt_path
            )

    print(f"\nBest epoch: {best_epoch}")
    if best_val_metrics is not None:
        vl, va, vp, vr, vf = best_val_metrics
        print(f"Best VAL: loss={vl:.4f} acc={va:.4f} prec={vp:.4f} rec={vr:.4f} f1={vf:.4f}")
    print(f"Saved best model to: {ckpt_path}")

    # Save history (NO OVERWRITE due to run folder)
    np.savetxt(
        history_csv,
        np.array(history, dtype=float),
        delimiter=",",
        header="epoch,train_loss,val_loss,val_acc,val_precision,val_recall,val_f1",
        comments=""
    )
    print(f"Saved history to: {history_csv}")

    # Load best checkpoint & evaluate on test
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, cfg.device, num_classes)
    print(
        f"\nTEST RESULTS (held-out): "
        f"loss={test_loss:.4f} acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f}"
    )

    # Save summary
    vl, va, vp, vr, vf = ckpt["best_val_metrics"]
    with open(summary_path, "w") as f:
        f.write(f"Best epoch: {ckpt['best_epoch']}\n")
        f.write(f"Best val loss: {vl:.6f}\n")
        f.write(f"Best val acc: {va:.6f}\n")
        f.write(f"Best val precision: {vp:.6f}\n")
        f.write(f"Best val recall: {vr:.6f}\n")
        f.write(f"Best val F1: {vf:.6f}\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test acc: {test_acc:.6f}\n")
        f.write(f"Test precision: {test_prec:.6f}\n")
        f.write(f"Test recall: {test_rec:.6f}\n")
        f.write(f"Test F1: {test_f1:.6f}\n")

    print("\nAll results saved successfully!")
    print(f"Everything is inside: {run_dir.resolve()}")

if __name__ == "__main__":
    main()
