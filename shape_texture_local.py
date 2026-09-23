"""
Shape vs. Texture — Local Python Script
========================================
Local (non-Colab) version of Sections 0, 19, 20 and 21 of the CNN
"Shape vs. Texture" notebook.

Assumes:
  1) You already have the Fruit Recognition dataset downloaded on this
     machine, with one subfolder per fruit class (ImageFolder format) —
     see the tutorial for how to get it.
  2) (Optional) You have a previously trained checkpoint (.pth file) you'd
     like to reload. If it's not found, the shape/texture probe below just
     runs on a freshly initialized model instead — the kernel-size
     experiment (Section 20) always trains its own models regardless.

Run from a terminal in this folder with:
    python shape_texture_local.py

Charts are both shown on screen and saved as PNG files in this folder, and
the results tables are printed to the terminal.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd

# ============================================================
# CONFIG — edit these for your machine
# ============================================================
DATASET_PATH = "fruit_dataset/train/train"               # folder with one subfolder per class
CHECKPOINT_PATH = None    # set to None to always start fresh

SEED = 42
IMAGE_SIZE = 64
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 64

KERNEL_SIZES = (3, 5, 7)   # Section 20 experiment conditions
EXPERIMENT_EPOCHS = 5      # change to wherever YOUR train/val curves plateau
N_RUNS = 3                 # repeats per kernel size, each with a different seed


# ============================================================
# SECTION 0 — Setup (local equivalent of the Colab "quick resume" cell)
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ShapeTextureCNN(nn.Module):
    def __init__(self, num_classes, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=pad)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def denormalize(img_tensor, mean=NORM_MEAN, std=NORM_STD):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return img_tensor * std_t + mean_t


def setup():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.isdir(DATASET_PATH) or not os.listdir(DATASET_PATH):
        raise RuntimeError(
            f"Dataset folder not found or empty at '{DATASET_PATH}'. "
            "Update DATASET_PATH at the top of this script — see the setup "
            "tutorial for how to download the dataset."
        )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    full_dataset_train_tf = datasets.ImageFolder(root=DATASET_PATH, transform=train_transform)
    full_dataset_eval_tf = datasets.ImageFolder(root=DATASET_PATH, transform=eval_transform)
    class_names = full_dataset_eval_tf.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes:", class_names)

    targets = np.array(full_dataset_train_tf.targets)
    indices = np.arange(len(full_dataset_train_tf))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=targets, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=targets[temp_idx], random_state=SEED)

    train_dataset = Subset(full_dataset_train_tf, train_idx)
    val_dataset = Subset(full_dataset_eval_tf, val_idx)
    test_dataset = Subset(full_dataset_eval_tf, test_idx)

    # num_workers=0 for simplicity/cross-platform safety. Once you've confirmed
    # this runs cleanly, you can raise it (Windows requires the
    # `if __name__ == "__main__":` guard already present at the bottom of this
    # file for num_workers > 0 to work correctly).
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ShapeTextureCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("Loaded saved checkpoint from", CHECKPOINT_PATH)
    else:
        print("No checkpoint found - using a freshly initialized model "
              "(fine for the kernel-size experiment, which trains its own "
              "models anyway; the shape/texture probe below will just be "
              "less meaningful on an untrained model).")

    return {
        "device": device, "model": model, "criterion": criterion, "optimizer": optimizer,
        "full_dataset_eval_tf": full_dataset_eval_tf, "class_names": class_names,
        "num_classes": num_classes, "test_idx": test_idx,
        "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
    }


# ============================================================
# SECTION 19 — Shape vs. texture bias probe
# ============================================================
def to_edge_map(img_tensor):
    img = denormalize(img_tensor).clamp(0, 1).permute(1, 2, 0).numpy()
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = np.stack([edges] * 3, axis=-1).astype(np.float32) / 255.0
    edges_tensor = torch.tensor(edges_rgb).permute(2, 0, 1)
    return transforms.Normalize(NORM_MEAN, NORM_STD)(edges_tensor)


def shuffle_patches(img_tensor, patch_size=8):
    c, h, w = img_tensor.shape
    img = img_tensor.clone()
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patches.append(img[:, i:i + patch_size, j:j + patch_size].clone())
    random.shuffle(patches)
    idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            img[:, i:i + patch_size, j:j + patch_size] = patches[idx]
            idx += 1
    return img


def evaluate_on_transformed(model, device, full_dataset_eval_tf, test_idx, transform_fn, n_samples=200):
    model.eval()
    correct, total = 0, 0
    sample_idx = random.sample(list(test_idx), min(n_samples, len(test_idx)))
    with torch.no_grad():
        for idx in sample_idx:
            img, label = full_dataset_eval_tf[idx]
            transformed = transform_fn(img).unsqueeze(0).to(device)
            output = model(transformed)
            pred = output.argmax(dim=1).item()
            correct += int(pred == label)
            total += 1
    return 100 * correct / total


def run_shape_texture_probe(ctx):
    model, device = ctx["model"], ctx["device"]
    full_dataset_eval_tf, test_idx = ctx["full_dataset_eval_tf"], ctx["test_idx"]

    baseline_acc = evaluate_on_transformed(model, device, full_dataset_eval_tf, test_idx, lambda x: x)
    edge_acc = evaluate_on_transformed(model, device, full_dataset_eval_tf, test_idx, to_edge_map)
    shuffled_acc = evaluate_on_transformed(model, device, full_dataset_eval_tf, test_idx,
                                            lambda x: shuffle_patches(x, patch_size=8))

    print(f"Accuracy on normal images:                {baseline_acc:.2f}%")
    print(f"Accuracy on edge-only (shape cue):         {edge_acc:.2f}%")
    print(f"Accuracy on patch-shuffled (texture cue):  {shuffled_acc:.2f}%")

    class_names = ctx["class_names"]
    demo_idx = random.choice(test_idx)
    img, label = full_dataset_eval_tf[demo_idx]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(denormalize(img).permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].set_title("Original")
    axes[1].imshow(denormalize(to_edge_map(img)).permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1].set_title("Edge-only (shape)")
    axes[2].imshow(denormalize(shuffle_patches(img)).permute(1, 2, 0).clamp(0, 1).numpy())
    axes[2].set_title("Patch-shuffled (texture)")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Example: {class_names[label]}")
    plt.tight_layout()
    plt.savefig("shape_texture_probe_example.png", dpi=150)
    print("Saved shape_texture_probe_example.png")
    plt.show()

    return baseline_acc, edge_acc, shuffled_acc


# ============================================================
# SECTION 20 — Kernel-size ablation experiment
# ============================================================
def run_epoch_generic(model, criterion, device, loader, train_mode, optimizer=None):
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    progress_every = max(1, n_batches // 4)
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            if (batch_idx + 1) % progress_every == 0 or (batch_idx + 1) == n_batches:
                mode_label = "train" if train_mode else "eval"
                print(f"    [{mode_label}] batch {batch_idx + 1}/{n_batches} "
                      f"- running loss {total_loss / total:.4f} "
                      f"- running acc {100 * correct / total:.1f}%")
    return total_loss / total, 100 * correct / total


def run_experiment(kernel_size, ctx, epochs, run_seed):
    device, criterion = ctx["device"], ctx["criterion"]
    train_loader, val_loader, test_loader = ctx["train_loader"], ctx["val_loader"], ctx["test_loader"]
    full_dataset_eval_tf, test_idx = ctx["full_dataset_eval_tf"], ctx["test_idx"]

    # Fresh model + optimizer every call - fully independent of the baseline
    # model, any loaded checkpoint, and every other run in this loop.
    set_seed(run_seed)
    exp_model = ShapeTextureCNN(num_classes=ctx["num_classes"], kernel_size=kernel_size).to(device)
    exp_optimizer = optim.Adam(exp_model.parameters(), lr=0.001)
    print(f"  -> training a NEW model: kernel_size={kernel_size}, seed={run_seed}")

    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch_generic(exp_model, criterion, device, train_loader, True, exp_optimizer)
        va_loss, va_acc = run_epoch_generic(exp_model, criterion, device, val_loader, False)
        print(f"     epoch {epoch + 1}/{epochs} - train_acc={tr_acc:.1f}% val_acc={va_acc:.1f}%")

    _, test_acc_ = run_epoch_generic(exp_model, criterion, device, test_loader, False)
    edge_acc_ = evaluate_on_transformed(exp_model, device, full_dataset_eval_tf, test_idx, to_edge_map)
    shuffled_acc_ = evaluate_on_transformed(exp_model, device, full_dataset_eval_tf, test_idx,
                                             lambda x: shuffle_patches(x, 8))

    return {
        "kernel_size": kernel_size, "seed": run_seed,
        "test_acc": test_acc_, "edge_acc": edge_acc_, "shuffled_acc": shuffled_acc_,
    }


def run_kernel_size_experiment(ctx):
    experiment_results = []
    for k in KERNEL_SIZES:
        for run in range(N_RUNS):
            print(f"kernel_size={k}, run {run + 1}/{N_RUNS}")
            result = run_experiment(k, ctx, epochs=EXPERIMENT_EPOCHS, run_seed=SEED + run)
            experiment_results.append(result)
            print(f"  DONE | test_acc={result['test_acc']:.2f}% "
                  f"edge_acc={result['edge_acc']:.2f}% "
                  f"shuffled_acc={result['shuffled_acc']:.2f}%\n")

    results_df = pd.DataFrame(experiment_results)
    print("\nRaw results - one row per training run:")
    print(results_df.to_string(index=False))

    summary_df = results_df.groupby("kernel_size")[["test_acc", "edge_acc", "shuffled_acc"]].agg(["mean", "std"])
    print("\nAggregated (mean +/- std across runs):")
    print(summary_df)
    summary_df.to_csv("kernel_size_experiment_summary.csv")
    results_df.to_csv("kernel_size_experiment_raw.csv", index=False)
    print("Saved kernel_size_experiment_summary.csv and kernel_size_experiment_raw.csv")

    means = results_df.groupby("kernel_size")[["test_acc", "edge_acc", "shuffled_acc"]].mean()
    stds = results_df.groupby("kernel_size")[["test_acc", "edge_acc", "shuffled_acc"]].std()

    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = range(len(means.index))
    width = 0.35
    ax.bar([p - width / 2 for p in x_pos], means["edge_acc"], width,
           yerr=stds["edge_acc"], label="Edge-only (shape)", capsize=4)
    ax.bar([p + width / 2 for p in x_pos], means["shuffled_acc"], width,
           yerr=stds["shuffled_acc"], label="Patch-shuffled (texture)", capsize=4)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([str(k) for k in means.index])
    ax.set_xlabel("Kernel size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Shape vs. texture accuracy by kernel size (mean +/- std over runs)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("kernel_size_bar_chart.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    for col, label in [("test_acc", "Normal test accuracy"),
                        ("edge_acc", "Edge-only (shape)"),
                        ("shuffled_acc", "Patch-shuffled (texture)")]:
        ax.errorbar(means.index, means[col], yerr=stds[col], marker="o", capsize=4, label=label)
    ax.set_xlabel("Kernel size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy trend across kernel sizes")
    ax.set_xticks(list(means.index))
    ax.legend()
    plt.tight_layout()
    plt.savefig("kernel_size_trend_chart.png", dpi=150)
    plt.show()

    return results_df, summary_df


# ============================================================
# SECTION 21 — Results summary
# ============================================================
def print_results_summary(ctx, baseline_acc, edge_acc, shuffled_acc, results_df):
    print("=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Classes ({ctx['num_classes']}):", ctx["class_names"])
    print(f"Baseline (unmodified) accuracy on probe subset: {baseline_acc:.2f}%")
    print(f"Shape-cue (edge-only) accuracy: {edge_acc:.2f}%")
    print(f"Texture-cue (patch-shuffled) accuracy: {shuffled_acc:.2f}%")
    if results_df is not None:
        print("\nKernel-size experiment summary:")
        print(results_df.groupby("kernel_size")[["test_acc", "edge_acc", "shuffled_acc"]].mean())


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ctx = setup()                                                  # Section 0
    baseline_acc, edge_acc, shuffled_acc = run_shape_texture_probe(ctx)  # Section 19
    results_df, summary_df = run_kernel_size_experiment(ctx)        # Section 20
    print_results_summary(ctx, baseline_acc, edge_acc, shuffled_acc, results_df)  # Section 21
