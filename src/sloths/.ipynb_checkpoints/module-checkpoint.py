import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
from pytorch_lightning import Trainer
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

# Import local modules from the sloths package
from sloths.ImageClassifier import ImageClassifier
from sloths.MetricsCallback import MetricsCallback
from sloths.visualizations import generate_visualizations


# =========================================================
# 1. Data Preparation
# =========================================================

def prepare_data(root_dir: str = "data", seed: int = 42):
    """Prepare datasets, splits, and generate summary reports."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    dataset = ImageFolder(root=root_dir, transform=transform)
    class_names = dataset.classes
    print("Detected classes:", class_names)

    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Create reports directory
    os.makedirs("reports/figures", exist_ok=True)

    # Class distribution
    counts = pd.Series([dataset.targets.count(i) for i in range(len(class_names))], index=class_names)
    class_dist = pd.DataFrame({"Class": class_names, "Count": counts.values})
    class_dist.to_csv("reports/class_distribution.csv", index=False)

    # Image dimension summary
    sizes = [Image.open(dataset.samples[i][0]).size for i in range(len(dataset))]
    widths, heights = zip(*sizes)
    dim_df = pd.DataFrame({"Width": widths, "Height": heights})
    dim_summary = dim_df.describe().round(2)
    dim_summary.to_csv("reports/image_dimensions_summary.csv")

    # Train/validation split summary
    split_df = pd.DataFrame({"Set": ["Train", "Validation"], "Samples": [len(train_dataset), len(val_dataset)]})
    split_df.to_csv("reports/train_validation_distribution.csv", index=False)

    return train_loader, val_loader, class_names, class_dist, widths, heights


# =========================================================
# 2. Model Training
# =========================================================

def train_model(train_loader, val_loader, class_names, max_epochs: int = 3):
    """Train the model using PyTorch Lightning."""
    metrics_cb = MetricsCallback()
    model = ImageClassifier(num_classes=len(class_names))

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        deterministic=True,
        callbacks=[metrics_cb],
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_loader, val_loader)
    return model, metrics_cb


# =========================================================
# 3. Evaluation
# =========================================================

def evaluate_model(model, val_loader, class_names):
    """Evaluate the model and compute metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    frac_pos, mean_pred = calibration_curve(all_labels, all_probs, n_bins=8)
    calib_error = np.abs(frac_pos - mean_pred).mean()

    # Compute per-sample loss for misclassified examples
    losses, samples, labels_np, pred_probs = [], [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            loss_batch = nn.CrossEntropyLoss(reduction="none")(logits, y)
            preds_batch = torch.argmax(logits, dim=1)
            mask = preds_batch != y
            if mask.any():
                losses.extend(loss_batch[mask].numpy())
                samples.extend(x[mask].numpy())
                labels_np.extend(y[mask].numpy())
                pred_probs.extend(probs[mask].numpy())

    losses = np.array(losses)
    samples = np.array(samples)
    labels_np = np.array(labels_np)
    pred_probs = np.array(pred_probs)

    # Select top-k misclassified samples per class
    k = 6
    top_samples = []
    for c in range(len(class_names)):
        idx_class = np.where(labels_np == c)[0]
        if len(idx_class) > 0:
            idx_top = idx_class[np.argsort(losses[idx_class])[-(k // 2):]]
            top_samples.extend(idx_top)

    return cm, mean_pred, frac_pos, calib_error, top_samples, samples, labels_np, losses, pred_probs


# =========================================================
# 4. Main Pipeline
# =========================================================

def main():
    """Complete pipeline: data preparation, training, and evaluation."""
    train_loader, val_loader, class_names, class_dist, widths, heights = prepare_data()

    model, metrics_cb = train_model(train_loader, val_loader, class_names)

    results = evaluate_model(model, val_loader, class_names)
    cm, mean_pred, frac_pos, calib_error, top_samples, samples, labels_np, losses, pred_probs = results

    # Choose a color palette interactively
    palette = input("Choose a color palette ('default', 'seaborn', 'ggplot', etc.): ")

    # Generate all visualizations
    generate_visualizations(
        train_loader=train_loader,
        class_names=class_names,
        class_dist=class_dist,
        widths=widths,
        heights=heights,
        metrics_cb=metrics_cb,
        cm=cm,
        mean_pred=mean_pred,
        frac_pos=frac_pos,
        calib_error=calib_error,
        top_samples=top_samples,
        samples=samples,
        labels_np=labels_np,
        losses=losses,
        pred_probs=pred_probs,
        color_palette=palette
    )


if __name__ == "__main__":
    main()
