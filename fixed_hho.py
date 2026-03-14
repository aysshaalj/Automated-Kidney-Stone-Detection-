import os
import random
import math
import argparse
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 0) Seed for Reproducibility
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

    # Final training
    base_epochs: int = 50
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 2
    seed: int = 42

    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # HHO params
    pop_size: int = 10
    max_iters: int = 12
    eval_epochs: int = 4

    max_samples_per_class: Optional[int] = None
    skip_hho: bool = False

    # output folder (unique run id)
    out_dir: str = "runs"

# 1.1) Run folder helper (prevents overwrite)
def make_run_dir(base: str = "runs", prefix: str = "hho_cnn") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

# 2) Parameterized CNN
class HHO_CNN(nn.Module):
    def __init__(self, num_classes: int, channels_scale: float = 1.0, use_block4: bool = True, dropout: float = 0.3):
        super().__init__()
        c1 = max(8, int(32 * channels_scale))
        c2 = max(8, int(64 * channels_scale))
        c3 = max(8, int(128 * channels_scale))
        c4 = max(8, int(256 * channels_scale))

        feats = []
        feats += [nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        feats += [nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        feats += [nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.MaxPool2d(2)]

        if use_block4:
            feats += [nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            final_c = c4
        else:
            final_c = c3

        self.features = nn.Sequential(*feats)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(final_c, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

# 3) Plots / Confusion Matrix
def save_accuracy_plot(history, save_path: Path, best_epoch: int):
    history_arr = np.array(history, dtype=float)
    epochs = history_arr[:, 0]
    val_accs = history_arr[:, 3]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accs, marker='o', linewidth=2, label='Validation Accuracy')
    plt.axvline(best_epoch, linestyle='--', linewidth=2, label=f'Best Epoch = {best_epoch}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to: {save_path}")

def save_confusion_matrix(model, loader, device, classes, save_path: Path, title: str = "Confusion Matrix"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")


# 4) Metrics
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

# 5) Train / Eval
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


# 6) Dataloaders
def build_dataloaders(cfg: CFG, flip_p: float, rotation_deg: float, batch_size: int, seed: int,
                      max_samples_per_class: Optional[int] = None):

    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.RandomRotation(degrees=rotation_deg),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data_dir = Path(cfg.data_dir)

    base_ds = datasets.ImageFolder(root=str(data_dir), transform=None)
    num_classes = len(base_ds.classes)

    indices = list(range(len(base_ds)))
    if max_samples_per_class is not None and max_samples_per_class > 0:
        class_to_idx = defaultdict(list)
        for i, (_, y) in enumerate(base_ds.samples):
            class_to_idx[y].append(i)
        rng = random.Random(seed)
        indices = []
        for c in range(num_classes):
            idxs = class_to_idx[c]
            rng.shuffle(idxs)
            indices.extend(idxs[:max_samples_per_class])

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_len = int(len(indices) * cfg.test_split)
    val_len = int(len(indices) * cfg.val_split)
    train_len = len(indices) - val_len - test_len
    if train_len <= 0:
        raise ValueError("Not enough data for requested splits. Reduce val_split/test_split.")

    test_idx = indices[:test_len]
    val_idx = indices[test_len:test_len + val_len]
    train_idx = indices[test_len + val_len:]

    train_ds_full = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
    eval_ds_full = datasets.ImageFolder(root=str(data_dir), transform=eval_tfms)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(eval_ds_full, val_idx)
    test_ds = Subset(eval_ds_full, test_idx)

    workers = max(0, cfg.num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False, persistent_workers=(workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=False, persistent_workers=(workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=False, persistent_workers=(workers > 0))

    return train_loader, val_loader, test_loader, num_classes, base_ds.classes


# 7) HHO Mapping
def map_position_to_params(pos: np.ndarray) -> Dict[str, Any]:
    lr = 10 ** (math.log10(1e-5) + pos[0] * (math.log10(1e-2) - math.log10(1e-5)))
    wd = 10 ** (math.log10(1e-6) + pos[1] * (math.log10(1e-3) - math.log10(1e-6)))
    dropout = 0.5 * pos[2]
    flip_p = 0.5 * pos[3]
    rotation_deg = 10.0 * pos[4]

    batch_choices = [16, 32, 48]
    channels_choices = [1.0, 1.25, 1.5]
    batch_size = batch_choices[min(int(pos[5] * len(batch_choices)), len(batch_choices) - 1)]
    channels_scale = channels_choices[min(int(pos[6] * len(channels_choices)), len(channels_choices) - 1)]
    use_block4 = bool(pos[7] >= 0.5)

    return {
        "lr": lr,
        "weight_decay": wd,
        "dropout": dropout,
        "flip_p": flip_p,
        "rotation_deg": rotation_deg,
        "batch_size": batch_size,
        "channels_scale": channels_scale,
        "use_block4": use_block4,
    }


# 8) Fitness Eval (LAST epoch F1)
def evaluate_config(cfg: CFG, params: Dict[str, Any], seed: int) -> Tuple[float, Dict[str, Any]]:
    device = cfg.device
    train_loader, val_loader, _, num_classes, _ = build_dataloaders(
        cfg,
        flip_p=params["flip_p"],
        rotation_deg=params["rotation_deg"],
        batch_size=params["batch_size"],
        seed=seed,
        max_samples_per_class=cfg.max_samples_per_class
    )

    model = HHO_CNN(
        num_classes=num_classes,
        channels_scale=params["channels_scale"],
        use_block4=params["use_block4"],
        dropout=params["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    last_f1 = 0.0
    for _ in range(cfg.eval_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, _, _, _, f1 = evaluate(model, val_loader, criterion, device, num_classes)
        last_f1 = f1

    return -last_f1, {"f1": last_f1}

# 9) HHO Algorithm
class HHO:
    def __init__(self, dim: int, pop_size: int, max_iters: int, seed: int = 42):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)

        self.X = self.rng.random((pop_size, dim))
        self.fitness = np.full(pop_size, np.inf)
        self.best_pos = None
        self.best_fit = np.inf

    def clip(self, x):
        return np.clip(x, 0.0, 1.0)

    def optimize(self, fitness_fn):
        for i in range(self.pop_size):
            self.fitness[i] = fitness_fn(self.X[i])

        best_idx = int(np.argmin(self.fitness))
        self.best_fit = self.fitness[best_idx]
        self.best_pos = self.X[best_idx].copy()

        for t in range(1, self.max_iters + 1):
            E1 = 2 * (1 - t / self.max_iters)

            for i in range(self.pop_size):
                E0 = 2 * self.rng.random() - 1
                E = E1 * E0

                q = self.rng.random()
                r = self.rng.random()
                J = 2 * (1 - self.rng.random())

                if abs(E) >= 1:
                    if r < 0.5:
                        rand_idx = self.rng.integers(0, self.pop_size)
                        X_rand = self.X[rand_idx]
                        self.X[i] = self.clip(X_rand - r * abs(X_rand - 2 * r * self.X[i]))
                    else:
                        X_mean = np.mean(self.X, axis=0)
                        self.X[i] = self.clip(self.best_pos - r * abs(self.best_pos - 2 * r * X_mean))
                else:
                    if r >= 0.5 and q >= 0.5:
                        Y = self.clip(self.best_pos - E * abs(J * self.best_pos - self.X[i]))
                        LF = self.levy_flight(self.dim)
                        Z = self.clip(Y + LF * self.rng.random(self.dim))

                        fitY = fitness_fn(Y)
                        fitZ = fitness_fn(Z)

                        if fitY < self.fitness[i] or fitZ < self.fitness[i]:
                            if fitY < fitZ:
                                self.X[i] = Y
                                self.fitness[i] = fitY
                            else:
                                self.X[i] = Z
                                self.fitness[i] = fitZ
                        if self.fitness[i] < self.best_fit:
                            self.best_fit = self.fitness[i]
                            self.best_pos = self.X[i].copy()
                        continue

                    if r < 0.5 and q >= 0.5:
                        self.X[i] = self.clip(self.best_pos - E * abs(self.best_pos - self.X[i]))
                    elif r >= 0.5 and q < 0.5:
                        X_mean = np.mean(self.X, axis=0)
                        self.X[i] = self.clip(self.best_pos - E * abs(J * self.best_pos - X_mean))
                    else:
                        X_mean = np.mean(self.X, axis=0)
                        self.X[i] = self.clip(self.best_pos - E * abs(J * self.best_pos - self.X[i]))

                fit = fitness_fn(self.X[i])
                self.fitness[i] = fit
                if fit < self.best_fit:
                    self.best_fit = fit
                    self.best_pos = self.X[i].copy()

        return self.best_pos, self.best_fit

    def levy_flight(self, dim: int, beta: float = 1.5):
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = self.rng.normal(0, sigma_u, size=dim)
        v = self.rng.normal(0, 1, size=dim)
        step = u / (abs(v) ** (1 / beta))
        return step


# 10) Main Workflow
def run_hho_search_and_train(args: Optional[argparse.Namespace] = None):
    cfg = CFG()

    # CLI overrides
    if args is not None:
        if getattr(args, "data_dir", None):
            cfg.data_dir = args.data_dir
        if getattr(args, "img_size", None):
            cfg.img_size = args.img_size
        if getattr(args, "max_samples_per_class", None) is not None:
            cfg.max_samples_per_class = args.max_samples_per_class
        if getattr(args, "base_epochs", None):
            cfg.base_epochs = args.base_epochs
        if getattr(args, "eval_epochs", None):
            cfg.eval_epochs = args.eval_epochs
        if getattr(args, "pop_size", None):
            cfg.pop_size = args.pop_size
        if getattr(args, "max_iters", None):
            cfg.max_iters = args.max_iters
        if getattr(args, "val_split", None):
            cfg.val_split = args.val_split
        if getattr(args, "test_split", None):
            cfg.test_split = args.test_split
        if getattr(args, "num_workers", None):
            cfg.num_workers = args.num_workers
        if getattr(args, "skip_hho", False):
            cfg.skip_hho = True
        if getattr(args, "out_dir", None):
            cfg.out_dir = args.out_dir
        if getattr(args, "quick", False):
            cfg.pop_size = 3
            cfg.max_iters = 2
            cfg.eval_epochs = 1
            cfg.base_epochs = 5

    if cfg.val_split + cfg.test_split >= 0.5:
        raise ValueError("val_split + test_split is too large. Keep it <= 0.3 for small datasets.")

    print("Device:", cfg.device)
    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir.resolve()}")

    run_dir = make_run_dir(cfg.out_dir, prefix="hho_cnn")
    print(f"Run output directory: {run_dir.resolve()}")

    ckpt_path = run_dir / "hho_cnn_presentation_best.pt"
    history_csv = run_dir / "training_results.csv"
    acc_plot_path = run_dir / "accuracy_plot.png"
    cm_val_path = run_dir / "confusion_matrix_val.png"
    cm_test_path = run_dir / "confusion_matrix_test.png"
    summary_path = run_dir / "summary.txt"

    tmp_ds = datasets.ImageFolder(root=str(data_dir), transform=transforms.ToTensor())
    print("Classes:", tmp_ds.classes, "num_classes =", len(tmp_ds.classes))

    if cfg.skip_hho:
        print("\nSkipping HHO search, using fixed hyperparameters...")
        best_params = {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "dropout": 0.35,
            "flip_p": 0.3,
            "rotation_deg": 5.0,
            "batch_size": 32,
            "channels_scale": 1.25,
            "use_block4": True,
        }
        print("Using params:", best_params)
    else:
        dim = 8
        hho = HHO(dim=dim, pop_size=cfg.pop_size, max_iters=cfg.max_iters, seed=cfg.seed)

        eval_seed = cfg.seed
        def fitness_fn(pos):
            params = map_position_to_params(pos)
            fit, info = evaluate_config(cfg, params, seed=eval_seed)
            print(f"Eval params: {params} | val_f1={info['f1']:.4f}")
            return fit

        print("\nStarting HHO search...")
        best_pos, best_fit = hho.optimize(fitness_fn)
        best_params = map_position_to_params(best_pos)
        print("\nHHO best params:", best_params, "| best_val_f1 =", -best_fit)

    with open(run_dir / "best_params.txt", "w") as f:
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

    train_loader, val_loader, test_loader, num_classes, class_names = build_dataloaders(
        cfg,
        flip_p=best_params["flip_p"],
        rotation_deg=best_params["rotation_deg"],
        batch_size=best_params["batch_size"],
        seed=cfg.seed,
        max_samples_per_class=cfg.max_samples_per_class
    )

    model = HHO_CNN(
        num_classes=num_classes,
        channels_scale=best_params["channels_scale"],
        use_block4=best_params["use_block4"],
        dropout=best_params["dropout"]
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

    # ========= BEST-EPOCH TRACKING (NEW) =========
    history = []
    best_f1_final = -1.0
    best_epoch = -1
    best_val_metrics = None  # (val_loss, acc, prec, rec, f1)

    print("\nTraining final model with best params...")
    for epoch in range(1, cfg.base_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_loss, acc, prec, rec, f1 = evaluate(model, val_loader, criterion, cfg.device, num_classes)

        history.append((epoch, train_loss, val_loss, acc, prec, rec, f1))
        print(
            f"Epoch {epoch:02d}/{cfg.base_epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_acc={acc:.4f} val_prec={prec:.4f} val_rec={rec:.4f} val_f1={f1:.4f}"
        )

        if f1 > best_f1_final:
            best_f1_final = f1
            best_epoch = epoch
            best_val_metrics = (val_loss, acc, prec, rec, f1)

            torch.save(
                {"model_state": model.state_dict(),
                 "classes": class_names,
                 "cfg": cfg.__dict__,
                 "best_params": best_params,
                 "best_val_f1": best_f1_final,
                 "best_epoch": best_epoch,
                 "best_val_metrics": best_val_metrics},
                ckpt_path
            )

    print(f"\nBest epoch: {best_epoch}")
    if best_val_metrics is not None:
        vl, va, vp, vr, vf = best_val_metrics
        print(f"Best VAL: loss={vl:.4f} acc={va:.4f} prec={vp:.4f} rec={vr:.4f} f1={vf:.4f}")

    print(f"Saved best model to: {ckpt_path}")

    np.savetxt(
        history_csv,
        np.array(history, dtype=float),
        delimiter=",",
        header="epoch,train_loss,val_loss,val_acc,val_precision,val_recall,val_f1",
        comments=""
    )
    print(f"Saved history to: {history_csv}")

    save_accuracy_plot(history, acc_plot_path, best_epoch=best_epoch)

    # Load best checkpoint for final evaluation
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"\n(Loaded checkpoint) Best epoch: {ckpt['best_epoch']}")
    vl, va, vp, vr, vf = ckpt["best_val_metrics"]
    print(f"(Loaded checkpoint) Best VAL: loss={vl:.4f} acc={va:.4f} prec={vp:.4f} rec={vr:.4f} f1={vf:.4f}")

    save_confusion_matrix(model, val_loader, cfg.device, class_names, cm_val_path, title="Confusion Matrix (Validation)")

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, cfg.device, num_classes)
    print(
        f"\nTEST RESULTS (held-out): "
        f"loss={test_loss:.4f} acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f}"
    )
    save_confusion_matrix(model, test_loader, cfg.device, class_names, cm_test_path, title="Confusion Matrix (Test)")

    # Summary file includes best epoch
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


# 11) CLI arguments for different options in running the code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HHO-driven CNN search and training (best-epoch saving)")

    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset root (class subfolders)")
    parser.add_argument("--quick", action="store_true", help="Enable quick-run mode (smaller search and fewer epochs)")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="Cap samples per class to speed up")
    parser.add_argument("--img_size", type=int, default=None, help="Image size (default 224)")
    parser.add_argument("--base_epochs", type=int, default=None, help="Final training epochs")
    parser.add_argument("--eval_epochs", type=int, default=None, help="Short training epochs per fitness evaluation")
    parser.add_argument("--pop_size", type=int, default=None, help="HHO population size")
    parser.add_argument("--max_iters", type=int, default=None, help="HHO iterations")
    parser.add_argument("--val_split", type=float, default=None, help="Validation split fraction (0-1)")
    parser.add_argument("--test_split", type=float, default=None, help="Test split fraction (0-1)")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--skip_hho", action="store_true", help="Skip HHO search and use fixed hyperparameters")
    parser.add_argument("--out_dir", type=str, default=None, help="Base output directory for runs (default: runs)")

    cli_args = parser.parse_args()
    run_hho_search_and_train(cli_args)
