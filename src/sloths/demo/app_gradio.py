import os, random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from torch import nn

from sloths.ImageClassifier import ImageClassifier
from sloths.MetricsCallback import MetricsCallback


# ====================================================
# Utility functions
# ====================================================
def fig_to_pil(fig):
    """Convert a matplotlib figure to a PIL image."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


# ====================================================
# Dataset Analysis
# ====================================================
def analyze_dataset(data_dir, seed, img_size, color_palette):
    """Analyze dataset structure and visualize key properties."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes

    # Set color style
    try:
        plt.style.use(color_palette if color_palette != "default" else "classic")
    except Exception:
        plt.style.use("classic")

    figs = []

    # --- 1. Sample images ---
    fig1, axes = plt.subplots(1, len(class_names), figsize=(12, 4))
    for i, cls in enumerate(class_names):
        indices = [j for j, (_, label) in enumerate(dataset) if label == i]
        idx = random.choice(indices)
        img, _ = dataset[idx]
        axes[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axes[i].set_title(cls)
        axes[i].axis("off")
    figs.append(fig_to_pil(fig1))

    # --- 2. Class distribution ---
    counts = [len([1 for _, label in dataset if label == i]) for i in range(len(class_names))]
    fig2, ax2 = plt.subplots()
    ax2.bar(class_names, counts)
    ax2.set_title("Class Distribution")
    figs.append(fig_to_pil(fig2))

    # --- 3. Image dimension summary ---
    sizes = [Image.open(path).size for path, _ in dataset.samples]
    widths, heights = zip(*sizes)
    fig3, ax3 = plt.subplots()
    ax3.hist(widths, bins=20, alpha=0.5, label="Width")
    ax3.hist(heights, bins=20, alpha=0.5, label="Height")
    ax3.legend()
    ax3.set_title("Image Dimension Distribution")
    figs.append(fig_to_pil(fig3))

    # --- 4. Color channel histograms ---
    fig4, axes4 = plt.subplots(1, len(class_names), figsize=(12, 4))
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
# Model Training and Visualization
# ====================================================
def run_experiment(data_dir, seed, img_size, batch_size, epochs, color_palette):
    """Train an image classifier and visualize performance metrics."""
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

    # Collect predictions
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
        plt.style.use(color_palette if color_palette != "default" else "classic")
    except Exception:
        plt.style.use("classic")

    # --- 1. Accuracy and loss table ---
    fig0, ax0 = plt.subplots(figsize=(6, 3))
    table_data = [
        ["Train", f"{metrics_cb.train_acc[-1]:.4f}" if hasattr(metrics_cb, "train_acc") else "N/A",
         f"{metrics_cb.train_loss[-1]:.4f}" if hasattr(metrics_cb, "train_loss") else "N/A"],
        ["Validation", f"{metrics_cb.val_acc[-1]:.4f}" if hasattr(metrics_cb, "val_acc") else "N/A",
         f"{metrics_cb.val_loss[-1]:.4f}" if hasattr(metrics_cb, "val_loss") else "N/A"]
    ]
    table = ax0.table(cellText=table_data, colLabels=["Set", "Accuracy", "Loss"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    ax0.axis("off")
    ax0.set_title("Final Accuracy and Loss")
    figs.append(fig_to_pil(fig0))

    # --- 2. Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    fig1, ax1 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax1, cmap="Blues", colorbar=False)
    ax1.set_title("Confusion Matrix")
    figs.append(fig_to_pil(fig1))

    # --- 3. Calibration curve (binary only) ---
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
# Gradio Interface
# ====================================================
def main():
    """Launch the interactive Gradio dashboard."""
    with gr.Blocks(title="Sloths Pain Experiment Dashboard") as demo:
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


if __name__ == "__main__":
    main()


