import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def generate_visualizations(train_loader, class_names, class_dist, widths, heights,
                            metrics_cb, cm, mean_pred, frac_pos, calib_error,
                            top_samples, samples, labels_np, losses, pred_probs,
                            color_palette="default"):
    """
    Generate and save visualizations for dataset exploration and model performance.

    This function creates multiple plots summarizing dataset characteristics,
    training metrics, and model evaluation results. All figures are saved
    under `reports/figures/`.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        DataLoader containing the training dataset.
    class_names : list of str
        List of class labels.
    class_dist : pandas.DataFrame
        DataFrame with 'Class' and 'Count' columns describing class distribution.
    widths : list or np.ndarray
        List of image widths in the dataset.
    heights : list or np.ndarray
        List of image heights in the dataset.
    metrics_cb : object
        Callback-like object containing train/val loss and accuracy lists.
    cm : np.ndarray
        Confusion matrix for model predictions.
    mean_pred : np.ndarray
        Mean predicted probabilities for calibration curve.
    frac_pos : np.ndarray
        Fraction of positive samples for calibration curve.
    calib_error : float
        Calibration error value to display on the plot.
    top_samples : list of int
        Indices of the most misclassified samples.
    samples : torch.Tensor
        Tensor containing sample images.
    labels_np : np.ndarray
        Ground truth labels corresponding to samples.
    losses : np.ndarray
        Loss values for each sample.
    pred_probs : np.ndarray
        Predicted probabilities for each sample.
    color_palette : str, optional
        Matplotlib color palette name. Default is "default".

    Returns
    -------
    None

    Examples
    --------
    >>> from visualizations import generate_visualizations
    >>> generate_visualizations(train_loader, class_names, class_dist, widths, heights,
    ...                         metrics_cb, cm, mean_pred, frac_pos, calib_error,
    ...                         top_samples, samples, labels_np, losses, pred_probs,
    ...                         color_palette="viridis")
    """
    if color_palette != "default":
        try:
            plt.style.use(color_palette)
        except OSError:
            available = plt.style.available
            print(f"'{color_palette}' is not a valid Matplotlib style. Using default instead.\n"
                  f"Available styles: {available}")
            plt.style.use("default")


    def imshow(img, title=None):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if title:
            plt.title(title)
        plt.axis("off")

    # =========================================================
    # Sample images
    # =========================================================
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        imshow(images[i], class_names[labels[i]])
    plt.suptitle("Sample images from dataset")
    plt.tight_layout()
    plt.savefig("reports/figures/sample_images.png")
    plt.show()

    # =========================================================
    # Class distribution + Image dimensions
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].pie(
        class_dist["Count"],
        labels=class_dist["Class"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    axes[0].set_title("Class distribution (donut chart)")
    axes[1].hist(widths, bins=20, alpha=0.5, label="Width")
    axes[1].hist(heights, bins=20, alpha=0.5, label="Height")
    axes[1].legend()
    axes[1].set_title("Image dimension distribution")
    plt.tight_layout()
    plt.savefig("reports/figures/class_and_dimensions.png")
    plt.show()

    # =========================================================
    # Histograms of average R, G, B values per class
    # =========================================================
    class_avg_colors = {cls: [] for cls in range(len(class_names))}
    for imgs, labels in train_loader:
        imgs = imgs * 0.5 + 0.5
        imgs = imgs * 255.0
        batch_avg = imgs.mean(dim=[2, 3])
        for i, label in enumerate(labels):
            class_avg_colors[label.item()].append(batch_avg[i])

    for cls in class_avg_colors:
        if len(class_avg_colors[cls]) > 0:
            class_avg_colors[cls] = torch.stack(class_avg_colors[cls]).numpy()
        else:
            class_avg_colors[cls] = np.zeros((0, 3))

    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 4 * n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]

    colors = ["red", "green", "blue"]
    labels = ["Red", "Green", "Blue"]
    for cls, ax in enumerate(axes):
        data = class_avg_colors[cls]
        for i in range(3):
            ax.hist(
                data[:, i], bins=30, color=colors[i], alpha=0.5,
                range=(0, 255), label=f"{labels[i]} channel"
            )
        ax.set_title(f"Class: {class_names[cls]}")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.xlabel("Average value (0-255)")
    plt.suptitle("Histogram of average channel values per class")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("reports/figures/avg_color_channels_per_class.png")
    plt.show()

    # =========================================================
    # Training vs Validation curves
    # =========================================================
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_cb.train_loss, label="Train Loss")
    plt.plot(metrics_cb.val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(metrics_cb.train_acc, label="Train Acc")
    plt.plot(metrics_cb.val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/training_curves.png")
    plt.show()

    # =========================================================
    # Confusion matrix
    # =========================================================
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("reports/figures/confusion_matrix.png")
    plt.show()

    # =========================================================
    # Calibration curve
    # =========================================================
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.text(0.6, 0.2, f"Calib. Error = {calib_error:.3f}")
    plt.savefig("reports/figures/calibration_curve.png")
    plt.show()

    # =========================================================
    # Top misclassified samples
    # =========================================================
    plt.figure(figsize=(8, 4))
    for i, idx in enumerate(top_samples):
        plt.subplot(2, 3, i + 1)
        img = samples[idx] / 2 + 0.5
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(f"{labels_np[labels_np[idx]]}\nLoss={losses[idx]:.2f}, Prob={pred_probs[idx]:.2f}")
        plt.axis("off")
    plt.suptitle("Top misclassified samples with predicted probability")
    plt.tight_layout()
    plt.savefig("reports/figures/high_loss_samples_with_prob.png")
    plt.show()
