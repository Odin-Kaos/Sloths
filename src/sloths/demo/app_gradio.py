"""
Image Classification Experiment Dashboard using Gradio and PyTorch Lightning.

This script provides an interactive dashboard for analyzing image datasets and
training classification models. It supports visualization of dataset
characteristics, training performance, confusion matrices, and calibration
curves. The interface is built using Gradio and integrates with PyTorch
Lightning for reproducible training workflows.

Modules used:
    - gradio: for interactive web interface
    - torch, torchvision: for deep learning and image preprocessing
    - pytorch_lightning: for structured training and validation
    - sklearn: for evaluation metrics and calibration
    - matplotlib, seaborn: for visualization
    - PIL: for image manipulation
"""

import gradio as gr
import torch
from torch import nn
import numpy as np
import random
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from sklearn.calibration import calibration_curve
from sloths.ImageClassifier import ImageClassifier
from sloths.MetricsCallback import MetricsCallback


# ====================================================
# Convert matplotlib figure to PIL.Image
# ====================================================
def fig_to_pil(fig, figsize=None):
    """
    Convert a Matplotlib figure to a PIL Image.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure to convert.
    figsize : tuple of float, optional
        Figure size in inches (width, height). If provided, resizes the figure before conversion.

    Returns
    -------
    PIL.Image.Image
        The converted image in RGB mode.
    """
    if figsize:
        fig.set_size_inches(*figsize)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)
    return img


# ====================================================
# Analyze dataset
# ====================================================
def analyze_dataset(data_dir, seed, img_size, color_palette):
    """
    Analyze the image dataset and generate visual summaries.

    This function inspects a dataset in the specified directory, visualizing
    class distribution, sample images, and image dimension statistics. It also
    computes and plots color channel histograms per class.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory containing class subfolders.
    seed : int
        Random seed for reproducibility.
    img_size : int
        Size (in pixels) to which each image will be resized for visualization.
    color_palette : str
        Matplotlib style name for visualization (e.g., "default", "seaborn", "ggplot").

    Returns
    -------
    list
        A list containing a descriptive message followed by PIL images of the
        generated figures.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes

    figs = []

    try:
        if color_palette != "default":
            plt.style.use(color_palette)
        else:
            plt.style.use("classic")
    except:
        plt.style.use("classic")

    # Sample images per class
    fig1, axes1 = plt.subplots(len(class_names), 2, figsize=(6, 2*len(class_names)))
    for i, cls in enumerate(class_names):
        indices = [j for j, (_, label) in enumerate(dataset) if label == i]
        samples = np.random.choice(indices, size=min(2, len(indices)), replace=False)
        for k, idx in enumerate(samples):
            img, _ = dataset[idx]
            axes1[i, k].imshow(img.permute(1, 2, 0))
            axes1[i, k].axis("off")
            if k == 0:
                axes1[i, k].set_ylabel(cls)
    fig1.suptitle("Sample images per class")
    figs.append(fig_to_pil(fig1))

    # Class distribution
    counts = [list(dataset.targets).count(i) for i in range(len(class_names))]
    fig2, ax2 = plt.subplots()
    wedges, texts, autotexts = ax2.pie(counts, labels=class_names, autopct="%1.1f%%", startangle=90)
    centre_circle = plt.Circle((0,0),0.9,fc='white')
    fig2.gca().add_artist(centre_circle)
    ax2.set_title("Class Distribution")
    figs.append(fig_to_pil(fig2))

    # Image dimensions
    widths, heights = [], []
    for path, _ in dataset.imgs:
        with Image.open(path) as im:
            w, h = im.size
            widths.append(w)
            heights.append(h)

    # Width distribution
    width_bins = np.arange(min(widths), max(widths)+10, 10)
    width_hist, _ = np.histogram(widths, bins=width_bins)
    fig_width, ax_width = plt.subplots(figsize=(8,4))
    ax_width.bar(width_bins[:-1], width_hist, width=10, color="skyblue", edgecolor="black")
    ax_width.set_xlabel("Width (pixels)")
    ax_width.set_ylabel("Number of images")
    ax_width.set_title("Width Distribution")
    figs.append(fig_to_pil(fig_width))

    # Height distribution
    height_bins = np.arange(min(heights), max(heights)+10, 10)
    height_hist, _ = np.histogram(heights, bins=height_bins)
    fig_height, ax_height = plt.subplots(figsize=(8,4))
    ax_height.bar(height_bins[:-1], height_hist, width=10, color="salmon", edgecolor="black")
    ax_height.set_xlabel("Height (pixels)")
    ax_height.set_ylabel("Number of images")
    ax_height.set_title("Height Distribution")
    figs.append(fig_to_pil(fig_height))

    # Color histograms
    fig4, axes4 = plt.subplots(len(class_names), 1, figsize=(6, 2*len(class_names)))
    if len(class_names) == 1:
        axes4 = [axes4]
    for i, cls in enumerate(class_names):
        indices = [j for j, (_, label) in enumerate(dataset) if label == i]
        sample_idx = np.random.choice(indices, size=min(10, len(indices)), replace=False)
        all_pixels = []
        for idx in sample_idx:
            img, _ = dataset[idx]
            all_pixels.append(img.numpy())
        all_pixels = np.concatenate(all_pixels, axis=1).reshape(3, -1)
        axes4[i].hist(all_pixels[0], bins=30, alpha=0.5, color="r", label="R")
        axes4[i].hist(all_pixels[1], bins=30, alpha=0.5, color="g", label="G")
        axes4[i].hist(all_pixels[2], bins=30, alpha=0.5, color="b", label="B")
        axes4[i].set_title(cls)
        axes4[i].legend()
    fig4.suptitle("Color Channel Histograms per Class")
    figs.append(fig_to_pil(fig4))

    message = f"Dataset analysis completed. {len(dataset)} images, {len(class_names)} classes."
    return [message] + figs


# ====================================================
# Main experiment function
# ====================================================
def run_experiment(data_dir, seed, img_size, batch_size, epochs, color_palette):
    """
    Train an image classification model and visualize performance metrics.

    This function initializes and trains a PyTorch Lightning model on a dataset,
    using a specified train-validation split. It computes predictions on the
    validation set and generates visualizations such as confusion matrices and
    calibration curves.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory.
    seed : int
        Random seed for reproducibility.
    img_size : int
        Image resizing dimension in pixels.
    batch_size : int
        Number of samples per training batch.
    epochs : int
        Number of training epochs.
    color_palette : str
        Matplotlib style name for visualization.

    Returns
    -------
    list
        A list containing a descriptive message followed by PIL images of the
        generated training result figures.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ImageClassifier(num_classes=len(class_names))
    metrics_cb = MetricsCallback()

    trainer = Trainer(max_epochs=epochs, accelerator="auto", deterministic=True,
                      callbacks=[metrics_cb], num_sanity_val_steps=0, enable_progress_bar=False)
    trainer.fit(model, train_loader, val_loader)

    all_preds, all_labels, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    figs = []
    try:
        if color_palette != "default":
            plt.style.use(color_palette)
        else:
            plt.style.use("classic")
    except:
        plt.style.use("classic")

    train_acc = metrics_cb.train_acc[-1] if hasattr(metrics_cb, "train_acc") else None
    val_acc = metrics_cb.val_acc[-1] if hasattr(metrics_cb, "val_acc") else None
    train_loss = metrics_cb.train_loss[-1] if hasattr(metrics_cb, "train_loss") else None
    val_loss = metrics_cb.val_loss[-1] if hasattr(metrics_cb, "val_loss") else None

    # Summary table
    fig0, ax0 = plt.subplots(figsize=(6,3))
    table_data = [
        ["Train", f"{train_acc:.4f}" if train_acc else "N/A", f"{train_loss:.4f}" if train_loss else "N/A"],
        ["Validation", f"{val_acc:.4f}" if val_acc else "N/A", f"{val_loss:.4f}" if val_loss else "N/A"]
    ]
    table = ax0.table(cellText=table_data, colLabels=["Set", "Accuracy", "Loss"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    for key, cell in table.get_celld().items():
        cell.set_height(0.2)
    ax0.axis("off")
    ax0.set_title("Final Accuracy and Loss", fontsize=16)
    figs.append(fig_to_pil(fig0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig1, ax1 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax1, cmap="Blues", colorbar=False)
    ax1.set_title("Confusion Matrix")
    figs.append(fig_to_pil(fig1))

    # Calibration curve (binary only)
    if len(class_names) == 2:
        fig2, ax2 = plt.subplots()
        probs_for_class1 = all_probs[:, 1]
        frac_pos, mean_pred = calibration_curve(all_labels, probs_for_class1, n_bins=10)
        ax2.plot(mean_pred, frac_pos, marker="o", label="Model")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2.set_xlabel("Mean predicted probability")
        ax2.set_ylabel("Fraction of positives")
        ax2.set_title("Calibration Curve")
        ax2.legend()
        figs.append(fig_to_pil(fig2))
    else:
        figs.append(Image.new("RGB", (400, 300), color=(255, 255, 255)))

    message = f"Training completed. {epochs} epochs with {len(class_names)} classes."
    return [message] + figs


# ====================================================
# Gradio interface
# ====================================================
with gr.Blocks(title="Sloths Pain Experiment Dashboard") as demo:
    """
    Launch the Gradio dashboard for dataset analysis and model experimentation.

    The interface provides controls for:
        - Dataset inspection (sample visualization, class distribution, etc.)
        - Training configuration (batch size, epochs, image size)
        - Model training and evaluation visualizations
    """
    gr.Markdown("## Image Classification Experiments with PyTorch Lightning")

    with gr.Row():
        data_dir = gr.Textbox(label="Data directory", value="data")
        seed = gr.Number(label="Random seed", value=42)

    with gr.Row():
        img_size = gr.Slider(64, 256, value=128, step=16, label="Image size (px)")
        color_palette = gr.Dropdown(
            ["default", "seaborn", "ggplot", "bmh", "dark_background"],
            value="default",
            label="Visualization style"
        )

    analyze_btn = gr.Button("Analyse dataset")
    analyze_output_text = gr.Textbox(label="Dataset analysis output")
    analyze_figs = [gr.Image(label=f"Figure {i+1}") for i in range(4)]

    analyze_btn.click(
        fn=analyze_dataset,
        inputs=[data_dir, seed, img_size, color_palette],
        outputs=[analyze_output_text] + analyze_figs
    )

    with gr.Row():
        batch_size = gr.Slider(8, 64, value=32, step=4, label="Batch size")
        epochs = gr.Slider(1, 10, value=3, step=1, label="Training epochs")

    run_btn = gr.Button("Run experiment")
    output_text = gr.Textbox(label="Experiment output")
    output_images = [gr.Image(label=f"Figure {i+1}") for i in range(3)]

    run_btn.click(
        fn=run_experiment,
        inputs=[data_dir, seed, img_size, batch_size, epochs, color_palette],
        outputs=[output_text] + output_images
    )

demo.launch()


# ====================================================
# Example of Use
# ====================================================
# The following example demonstrates how to call the key functions directly
# without using the Gradio interface. Adjust the `data_dir` path to your dataset.

# Example:
# if __name__ == "__main__":
#     data_dir = "path/to/your/dataset"
#     seed = 42
#     img_size = 128
#     batch_size = 32
#     epochs = 3
#     color_palette = "seaborn"
#
#     # Analyze dataset
#     analysis_results = analyze_dataset(data_dir, seed, img_size, color_palette)
#     print(analysis_results[0])  # Prints summary message
#     analysis_results[1].show()  # Display first figure (sample images)
#
#     # Run training experiment
#     experiment_results = run_experiment(data_dir, seed, img_size, batch_size, epochs, color_palette)
#     print(experiment_results[0])  # Prints training summary message
#     experiment_results[1].show()  # Display first result figure (accuracy/loss table)





