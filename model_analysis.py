#creating analysis charts and plots for model comparison
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from datetime import datetime

# Models
class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class HHO_CNN(nn.Module):
    """Must match your HHO training architecture exactly."""
    def __init__(self, num_classes: int, channels_scale: float = 1.0, use_block4: bool = True, dropout: float = 0.3):
        super().__init__()
        c1 = int(32 * channels_scale)
        c2 = int(64 * channels_scale)
        c3 = int(128 * channels_scale)
        c4 = int(256 * channels_scale)

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
        return self.classifier(x)


# CSV Column Standardization
def standardize_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames common column variants into a standard set:
    epoch, train_loss, val_loss, acc, precision, recall, f1

    Prevents KeyError: 'acc' etc.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {}

    # epoch
    for c in ["epoch", "epochs", "ep"]:
        if c in df.columns:
            rename[c] = "epoch"
            break

    # losses
    for c in ["train_loss", "training_loss", "loss_train", "trainloss"]:
        if c in df.columns:
            rename[c] = "train_loss"
            break
    for c in ["val_loss", "validation_loss", "loss_val", "valloss"]:
        if c in df.columns:
            rename[c] = "val_loss"
            break

    # accuracy -> acc
    for c in ["acc", "accuracy", "val_acc", "val_accuracy", "valid_acc", "valid_accuracy"]:
        if c in df.columns:
            rename[c] = "acc"
            break

    # precision / recall / f1
    for c in ["precision", "prec", "val_precision"]:
        if c in df.columns:
            rename[c] = "precision"
            break
    for c in ["recall", "val_recall"]:
        if c in df.columns:
            rename[c] = "recall"
            break
    for c in ["f1", "f1_score", "f1score", "val_f1", "val_f1_score"]:
        if c in df.columns:
            rename[c] = "f1"
            break

    df = df.rename(columns=rename)
    print("CSV columns after standardize:", list(df.columns))
    return df



# Helpers
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def normalize_cm(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def per_class_metrics_from_cm(cm: np.ndarray):
    n = cm.shape[0]
    prec = np.zeros(n, dtype=float)
    rec = np.zeros(n, dtype=float)
    f1 = np.zeros(n, dtype=float)

    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0

        prec[c], rec[c], f1[c] = p, r, f

    return prec, rec, f1


def plot_confusion_matrix_pretty(cm: np.ndarray, classes, title: str, save_path: Path):
    cm = np.asarray(cm)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(cm, cmap="Blues")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    thresh = cm.max() * 0.55 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_curve(df_base: pd.DataFrame, df_hho: pd.DataFrame, col: str, title: str, out_path: Path):
    if col not in df_base.columns or col not in df_hho.columns:
        print(f"Skipping curve '{col}' because it's missing in one of the CSVs.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df_base["epoch"], df_base[col], label="Baseline", linewidth=2, marker="o")
    plt.plot(df_hho["epoch"], df_hho[col], label="HHO", linewidth=2, marker="s")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")


def plot_average_metrics_bar(df_base: pd.DataFrame, df_hho: pd.DataFrame, out_path: Path):
    """
    Bar chart comparing AVERAGE metrics:
    mean acc/precision/recall/f1 across all epochs in each history csv.
    """
    metrics = ["acc", "precision", "recall", "f1"]

    base_vals = [safe_mean(df_base[m]) if m in df_base.columns else np.nan for m in metrics]
    hho_vals = [safe_mean(df_hho[m]) if m in df_hho.columns else np.nan for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, base_vals, width, label="Baseline (avg over epochs)")
    bars2 = plt.bar(x + width / 2, hho_vals, width, label="HHO (avg over epochs)")

    plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1"])
    plt.ylabel("Average Score")
    plt.title("Average Validation Metrics: Baseline vs HHO")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            txt = "N/A" if np.isnan(h) else f"{h:.3f}"
            y = 0.02 if np.isnan(h) else h + 0.015
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                txt,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_avg_f1_bar_only(df_base: pd.DataFrame, df_hho: pd.DataFrame, out_path: Path):
    base_avg_f1 = safe_mean(df_base["f1"]) if "f1" in df_base.columns else np.nan
    hho_avg_f1 = safe_mean(df_hho["f1"]) if "f1" in df_hho.columns else np.nan

    labels = ["Baseline", "HHO"]
    vals = [base_avg_f1, hho_avg_f1]
    x = np.arange(len(labels))

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, vals)
    plt.xticks(x, labels)
    plt.ylabel("Average F1")
    plt.title("Average F1 Comparison (Across Epochs)")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)

    for b in bars:
        h = b.get_height()
        txt = "N/A" if np.isnan(h) else f"{h:.3f}"
        y = 0.02 if np.isnan(h) else h + 0.015
        plt.text(b.get_x() + b.get_width()/2, y, txt, ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_time_to_quality(df_base: pd.DataFrame, df_hho: pd.DataFrame, thresholds=(0.85, 0.90), out_path: Path = Path("time_to_quality.png")):
    """
    Bar chart: epochs to reach F1 thresholds + epoch of best F1.
    """
    if "epoch" not in df_base.columns or "epoch" not in df_hho.columns or "f1" not in df_base.columns or "f1" not in df_hho.columns:
        print("⚠️ Skipping time-to-quality chart (missing epoch/f1).")
        return

    def best_epoch(df: pd.DataFrame) -> int:
        s = pd.to_numeric(df["f1"], errors="coerce")
        i = s.idxmax()
        return int(df.loc[i, "epoch"])

    def epoch_to_reach(df: pd.DataFrame, threshold: float):
        s = df.sort_values("epoch").copy()
        s["f1"] = pd.to_numeric(s["f1"], errors="coerce")
        hits = s[s["f1"] >= threshold]
        if len(hits) == 0:
            return None
        return int(hits.iloc[0]["epoch"])

    base_best_ep = best_epoch(df_base)
    hho_best_ep = best_epoch(df_hho)

    labels = []
    base_vals = []
    hho_vals = []

    for th in thresholds:
        labels.append(f"F1≥{th:.2f}")
        b = epoch_to_reach(df_base, th)
        h = epoch_to_reach(df_hho, th)
        base_vals.append(b if b is not None else np.nan)
        hho_vals.append(h if h is not None else np.nan)

    labels.append("Best F1 epoch")
    base_vals.append(base_best_ep)
    hho_vals.append(hho_best_ep)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(11, 6))
    bars1 = plt.bar(x - width / 2, base_vals, width, label="Baseline")
    bars2 = plt.bar(x + width / 2, hho_vals, width, label="HHO")

    plt.xticks(x, labels)
    plt.ylabel("Epoch")
    plt.title("Time-to-Quality (Lower is Better)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    def autolabel(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                txt = "N/A"
                y = 0.5
            else:
                txt = f"{int(h)}"
                y = h + 0.5
            plt.text(bar.get_x() + bar.get_width()/2, y, txt, ha="center", va="bottom", fontweight="bold")

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_generalization_gap(df_base: pd.DataFrame, df_hho: pd.DataFrame, out_path: Path):
    """
    gap = val_loss - train_loss at epoch of best F1.
    """
    needed = {"train_loss", "val_loss", "f1", "epoch"}
    if not needed.issubset(df_base.columns) or not needed.issubset(df_hho.columns):
        print("⚠️ Skipping generalization gap chart (missing train_loss/val_loss/f1/epoch).")
        return

    def best_row(df: pd.DataFrame):
        s = pd.to_numeric(df["f1"], errors="coerce")
        i = s.idxmax()
        return df.loc[i]

    b = best_row(df_base)
    h = best_row(df_hho)

    base_gap = float(b["val_loss"]) - float(b["train_loss"])
    hho_gap = float(h["val_loss"]) - float(h["train_loss"])

    x = np.arange(2)
    vals = [base_gap, hho_gap]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, vals)
    plt.xticks(x, ["Baseline", "HHO"])
    plt.ylabel("Val Loss - Train Loss")
    plt.title("Generalization Gap at Best F1 (Lower is Better)")
    plt.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        hgt = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, hgt + 0.01, f"{hgt:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_class_f1(classes, f1_base: np.ndarray, f1_hho: np.ndarray, out_path: Path):
    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(max(10, len(classes) * 1.2), 6))
    bars1 = plt.bar(x - width/2, f1_base, width, label="Baseline")
    bars2 = plt.bar(x + width/2, f1_hho, width, label="HHO")

    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylabel("F1 (per class)")
    plt.title("Per-Class F1 Comparison")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_cm_difference(cm_base_norm: np.ndarray, cm_hho_norm: np.ndarray, classes, out_path: Path):
    diff = cm_hho_norm - cm_base_norm
    plt.figure(figsize=(8, 6))
    plt.imshow(diff, aspect="auto", cmap="coolwarm")
    plt.title("Confusion Matrix Difference (HHO - Baseline, Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{diff[i, j]:+.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def build_val_loader_with_split(data_dir: Path, tfms, split_path: Path, batch_size: int = 64):
    full_ds = datasets.ImageFolder(str(data_dir), transform=tfms)
    classes = full_ds.classes
    num_classes = len(classes)

    if split_path.exists():
        d = np.load(split_path)
        val_idx = d["val_idx"].tolist()
        val_ds = Subset(full_ds, val_idx)
        print(f"Using saved split: {split_path.resolve()}")
    else:
        rng = np.random.default_rng(42)
        indices = np.arange(len(full_ds))
        rng.shuffle(indices)
        val_len = int(len(full_ds) * 0.2)
        val_idx = indices[:val_len].tolist()
        val_ds = Subset(full_ds, val_idx)
        print("WARNING: split_indices.npz not found; using fallback 20% val split with seed=42")

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader, classes, num_classes


def eval_model_preds(model: nn.Module, loader: DataLoader, device: str):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

def plot_best_f1_bar(df_base: pd.DataFrame, df_hho: pd.DataFrame, out_path: Path):
    """
    Bar chart comparing BEST F1 (max over epochs) for Baseline vs HHO.
    Writes exact numbers on bars + shows which epoch it happened.
    """
    if "f1" not in df_base.columns or "epoch" not in df_base.columns:
        print("Skipping best F1 bar (Baseline missing f1/epoch).")
        return
    if "f1" not in df_hho.columns or "epoch" not in df_hho.columns:
        print("Skipping best F1 bar (HHO missing f1/epoch).")
        return

    base_f1 = pd.to_numeric(df_base["f1"], errors="coerce")
    hho_f1 = pd.to_numeric(df_hho["f1"], errors="coerce")

    base_idx = base_f1.idxmax()
    hho_idx = hho_f1.idxmax()

    base_best_f1 = float(base_f1.loc[base_idx])
    hho_best_f1 = float(hho_f1.loc[hho_idx])

    base_best_ep = int(pd.to_numeric(df_base.loc[base_idx, "epoch"], errors="coerce"))
    hho_best_ep = int(pd.to_numeric(df_hho.loc[hho_idx, "epoch"], errors="coerce"))

    labels = [f"Baseline\n(ep {base_best_ep})", f"HHO\n(ep {hho_best_ep})"]
    vals = [base_best_f1, hho_best_f1]
    x = np.arange(len(labels))

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, vals)
    plt.xticks(x, labels)
    plt.ylabel("Best F1")
    plt.title("Best F1 Comparison (Max Over Epochs)")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)

    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x() + b.get_width()/2,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
def plot_metrics_curve_single(df: pd.DataFrame, model_name: str, out_path: Path):
    """
    Plot one model's validation metrics curve like your screenshot:
    Accuracy, Precision, Recall, F1 vs Epoch (all on same plot).
    Writes nice title + legend, and saves to out_path.
    """
    if "epoch" not in df.columns:
        print(f"Skipping {model_name} metrics curve (missing 'epoch').")
        return

    # Choose which metric columns exist
    metric_cols = []
    labels = []
    for col, lab in [("acc", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]:
        if col in df.columns:
            metric_cols.append(col)
            labels.append(lab)

    if not metric_cols:
        print(f"Skipping {model_name} metrics curve (no acc/precision/recall/f1 found).")
        return

    # Make sure numeric
    x = pd.to_numeric(df["epoch"], errors="coerce")
    plt.figure(figsize=(10, 7))

    for col, lab in zip(metric_cols, labels):
        y = pd.to_numeric(df[col], errors="coerce")
        plt.plot(x, y, linewidth=2, label=lab)  # no forced colors (matplotlib defaults)

    plt.title(f"{model_name} Validation Metrics", fontsize=18, fontweight="bold")
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_hho_metrics_grid(df_hho: pd.DataFrame, out_path: Path, title: str = "CNN + HHO - Validation Metrics"):
    """
    One figure, 2x2 grid, separate plots for:
      Accuracy, Precision, Recall, F1
    (Like your example, but ONLY these 4 metrics.)
    """
    required = ["epoch", "acc", "precision", "recall", "f1"]
    missing = [c for c in required if c not in df_hho.columns]
    if missing:
        print(f"Skipping HHO 2x2 grid (missing columns: {missing}). Columns found: {list(df_hho.columns)}")
        return

    # numeric
    x = pd.to_numeric(df_hho["epoch"], errors="coerce")
    acc = pd.to_numeric(df_hho["acc"], errors="coerce")
    prec = pd.to_numeric(df_hho["precision"], errors="coerce")
    rec = pd.to_numeric(df_hho["recall"], errors="coerce")
    f1 = pd.to_numeric(df_hho["f1"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Accuracy
    axes[0, 0].plot(x, acc, linewidth=2, label="acc")
    axes[0, 0].set_title("Accuracy", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Precision
    axes[0, 1].plot(x, prec, linewidth=2, label="precision")
    axes[0, 1].set_title("Precision", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Recall
    axes[1, 0].plot(x, rec, linewidth=2, label="recall")
    axes[1, 0].set_title("Recall", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # F1
    axes[1, 1].plot(x, f1, linewidth=2, label="f1")
    axes[1, 1].set_title("F1 Score", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# Main
def main():
    device = pick_device()
    print("Device:", device)

    # --- UPDATE if needed ---
    data_dir = Path("./Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection A Resource for Deep Learning Research")

    # Fair split file
    split_path = Path("split_indices.npz")

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # ---- Output folder (won't overwrite old runs) ----
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("comparison_outputs") / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving outputs to:", out_dir.resolve())

    # ----- Val loader -----
    val_loader, classes, num_classes = build_val_loader_with_split(
        data_dir=data_dir,
        tfms=tfms,
        split_path=split_path,
        batch_size=64
    )
    print("Classes:", classes)

    # ----- Load Baseline checkpoint -----
    baseline_ckpt_path = Path("best_fixed_model.pt")
    if not baseline_ckpt_path.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_ckpt_path.resolve()}")

    baseline_ckpt = torch.load(str(baseline_ckpt_path), map_location="cpu")
    baseline_model = BaselineCNN(num_classes=num_classes).to(device)
    baseline_state = baseline_ckpt["model_state"] if isinstance(baseline_ckpt, dict) and "model_state" in baseline_ckpt else baseline_ckpt
    baseline_model.load_state_dict(baseline_state)

    # ----- Load HHO checkpoint -----
    hho_ckpt_path = Path("hho_cnn_presentation_best.pt")
    if not hho_ckpt_path.exists():
        raise FileNotFoundError(f"Missing HHO checkpoint: {hho_ckpt_path.resolve()}")

    hho_ckpt = torch.load(str(hho_ckpt_path), map_location="cpu")
    best_params = hho_ckpt.get("best_params", {}) if isinstance(hho_ckpt, dict) else {}

    channels_scale = float(best_params.get("channels_scale", 1.0))
    use_block4 = bool(best_params.get("use_block4", True))
    dropout = float(best_params.get("dropout", 0.3))

    hho_model = HHO_CNN(
        num_classes=num_classes,
        channels_scale=channels_scale,
        use_block4=use_block4,
        dropout=dropout
    ).to(device)

    hho_state = hho_ckpt["model_state"] if isinstance(hho_ckpt, dict) and "model_state" in hho_ckpt else hho_ckpt
    hho_model.load_state_dict(hho_state)

    # Predictions + Confusion
    y_true_b, y_pred_b = eval_model_preds(baseline_model, val_loader, device)
    y_true_h, y_pred_h = eval_model_preds(hho_model, val_loader, device)

    cm_base = confusion_matrix_np(y_true_b, y_pred_b, num_classes)
    cm_hho = confusion_matrix_np(y_true_h, y_pred_h, num_classes)

    cm_base_norm = normalize_cm(cm_base)
    cm_hho_norm = normalize_cm(cm_hho)

    # Confusion matrices (counts + normalized %)
    plot_confusion_matrix_pretty(cm_base, classes, "Baseline Confusion Matrix", out_dir / "baseline_confusion_matrix_counts.png")
    plot_confusion_matrix_pretty(cm_hho, classes, "HHO Confusion Matrix", out_dir / "hho_confusion_matrix_counts.png")

    plot_confusion_matrix_pretty((cm_base_norm * 100).round(2), classes, "Baseline Confusion Matrix (Normalized %)", out_dir / "baseline_confusion_matrix_norm_percent.png")
    plot_confusion_matrix_pretty((cm_hho_norm * 100).round(2), classes, "HHO Confusion Matrix (Normalized %)", out_dir / "hho_confusion_matrix_norm_percent.png")

    plot_cm_difference(cm_base_norm, cm_hho_norm, classes, out_dir / "comparison_cm_difference.png")

    # Per-class metrics csv + per-class F1 bar
    prec_b, rec_b, f1_b = per_class_metrics_from_cm(cm_base)
    prec_h, rec_h, f1_h = per_class_metrics_from_cm(cm_hho)

    per_class_df = pd.DataFrame({
        "class": classes,
        "baseline_precision": prec_b,
        "baseline_recall": rec_b,
        "baseline_f1": f1_b,
        "hho_precision": prec_h,
        "hho_recall": rec_h,
        "hho_f1": f1_h,
        "f1_delta_hho_minus_baseline": (f1_h - f1_b),
    })
    per_class_df.to_csv(out_dir / "comparison_per_class_metrics.csv", index=False)
    print(f"Saved: {out_dir / 'comparison_per_class_metrics.csv'}")

    plot_per_class_f1(classes, f1_b, f1_h, out_dir / "comparison_per_class_f1.png")

    # Curves from CSV histories
    baseline_history = Path("history.csv")
    hho_history = Path("training_results.csv")

    if not baseline_history.exists():
        raise FileNotFoundError(f"Missing baseline history CSV: {baseline_history.resolve()}")
    if not hho_history.exists():
        raise FileNotFoundError(f"Missing HHO history CSV: {hho_history.resolve()}")

    df_base = standardize_history_columns(pd.read_csv(baseline_history))
    df_hho = standardize_history_columns(pd.read_csv(hho_history))

    # Convert numerics safely (only if column exists)
    for df in (df_base, df_hho):
        for c in ["epoch", "train_loss", "val_loss", "acc", "precision", "recall", "f1"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Curves (skip missing columns automatically)
    plot_curve(df_base, df_hho, "train_loss", "Training Loss Comparison", out_dir / "comparison_train_loss.png")
    plot_curve(df_base, df_hho, "val_loss", "Validation Loss Comparison", out_dir / "comparison_val_loss.png")
    plot_curve(df_base, df_hho, "acc", "Accuracy Comparison", out_dir / "comparison_accuracy.png")
    plot_curve(df_base, df_hho, "precision", "Precision Comparison", out_dir / "comparison_precision.png")
    plot_curve(df_base, df_hho, "recall", "Recall Comparison", out_dir / "comparison_recall.png")
    plot_curve(df_base, df_hho, "f1", "F1-Score Comparison", out_dir / "comparison_f1.png")
    plot_best_f1_bar(df_base, df_hho, out_dir / "comparison_best_f1_bar.png")
    plot_metrics_curve_single(df_base, "Baseline CNN", out_dir / "baseline_validation_metrics_curve.png")
    plot_metrics_curve_single(df_hho, "CNN + HHO", out_dir / "hho_validation_metrics_curve.png")
    plot_hho_metrics_grid(df_hho, out_dir / "hho_metrics_grid_2x2.png")




    # Average metrics bar charts (won't crash if 'acc' was originally 'accuracy')
    plot_average_metrics_bar(df_base, df_hho, out_dir / "comparison_avg_metrics_bar.png")
    plot_avg_f1_bar_only(df_base, df_hho, out_dir / "comparison_avg_f1_bar.png")

    # Extra comparisons
    plot_time_to_quality(df_base, df_hho, thresholds=(0.85, 0.90), out_path=out_dir / "comparison_time_to_quality.png")
    plot_generalization_gap(df_base, df_hho, out_path=out_dir / "comparison_generalization_gap.png")

    # Summary CSV
    summary = pd.DataFrame([
        {
            "model": "baseline",
            "avg_acc": safe_mean(df_base["acc"]) if "acc" in df_base.columns else np.nan,
            "avg_precision": safe_mean(df_base["precision"]) if "precision" in df_base.columns else np.nan,
            "avg_recall": safe_mean(df_base["recall"]) if "recall" in df_base.columns else np.nan,
            "avg_f1": safe_mean(df_base["f1"]) if "f1" in df_base.columns else np.nan,
        },
        {
            "model": "hho",
            "avg_acc": safe_mean(df_hho["acc"]) if "acc" in df_hho.columns else np.nan,
            "avg_precision": safe_mean(df_hho["precision"]) if "precision" in df_hho.columns else np.nan,
            "avg_recall": safe_mean(df_hho["recall"]) if "recall" in df_hho.columns else np.nan,
            "avg_f1": safe_mean(df_hho["f1"]) if "f1" in df_hho.columns else np.nan,
        }
    ])
    summary.to_csv(out_dir / "comparison_avg_metrics_summary.csv", index=False)
    print(f"Saved: {out_dir / 'comparison_avg_metrics_summary.csv'}")

    print("\nDone. Outputs saved in:")
    print("   ", out_dir.resolve())


if __name__ == "__main__":
    main()
