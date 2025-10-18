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
from visualizations import generate_visualizations
from ImageClassifier import ImageClassifier  # tu modelo
from MetricsCallback import MetricsCallback  # tu callback de métricas
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve


# ============================================
# FUNCIONES DE ENTRENAMIENTO Y EXPERIMENTO
# ============================================

def run_experiment(
    data_dir: str,
    seed: int,
    img_size: int,
    batch_size: int,
    epochs: int,
    color_palette: str
):
    # ===============================
    # 1️⃣ Semilla
    # ===============================
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ===============================
    # 2️⃣ Dataset y DataLoaders
    # ===============================
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ===============================
    # 3️⃣ Resúmenes iniciales
    # ===============================
    counts = pd.Series([dataset.targets.count(i) for i in range(len(class_names))], index=class_names)
    class_dist = pd.DataFrame({"Class": class_names, "Count": counts.values})

    sizes = [Image.open(dataset.samples[i][0]).size for i in range(len(dataset))]
    widths, heights = zip(*sizes)

    os.makedirs("reports/figures", exist_ok=True)

    # ===============================
    # 4️⃣ Modelo y Callback
    # ===============================
    metrics_cb = MetricsCallback()
    model = ImageClassifier(num_classes=len(class_names))

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        deterministic=True,
        callbacks=[metrics_cb],
        num_sanity_val_steps=0,
        enable_progress_bar=False
    )

    trainer.fit(model, train_loader, val_loader)

    # ===============================
    # 5️⃣ Evaluación y top misclassifications
    # ===============================
    all_preds, all_labels, all_probs = [], [], []
    losses, samples_arr, labels_np, pred_probs = [], [], [], []

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

            loss_batch = nn.CrossEntropyLoss(reduction="none")(logits, y)
            mask = preds != y

            if mask.any():
                losses.extend(loss_batch[mask].cpu().numpy())
                samples_arr.extend(x[mask].cpu().numpy())
                labels_np.extend(y[mask].cpu().numpy())
                pred_probs.extend(probs[mask].cpu().numpy())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    losses = np.array(losses)
    samples_arr = np.array(samples_arr)
    labels_np = np.array(labels_np)
    pred_probs = np.array(pred_probs)

    # Top-k misclassified por clase
    k = 6
    top_samples = []
    for c in range(len(class_names)):
        idx_class = np.where(labels_np == c)[0]
        if len(idx_class) > 0:
            idx_top = idx_class[np.argsort(losses[idx_class])[-(k//2):]]
            top_samples.extend(idx_top)

    # ===============================
    # 6️⃣ Generar visualizaciones
    # ===============================
    cm = confusion_matrix(all_labels, all_preds)

    # Normaliza tipos de entrada
    if isinstance(all_probs, list):
        try:
            # Si la lista contiene tensores
            if isinstance(all_probs[0], torch.Tensor):
                all_probs = torch.cat(all_probs, dim=0).cpu().numpy()
            else:
                all_probs = np.array(all_probs)
        except Exception as e:
            print("⚠️ Could not convert all_probs list:", e)
            all_probs = np.array([])
    
    if isinstance(all_labels, list):
        all_labels = np.array(all_labels)
    
    if all_probs is not None and len(all_probs) > 0 and len(np.unique(all_labels)) == 2:
        try:
            probs_for_class1 = all_probs[:, 1] if all_probs.ndim > 1 else all_probs
            frac_pos, mean_pred = calibration_curve(all_labels, probs_for_class1, n_bins=10)
            calib_error = np.mean(np.abs(frac_pos - mean_pred))
        except Exception as e:
            print("⚠️ Could not compute calibration curve:", e)
            mean_pred, frac_pos, calib_error = None, None, None
    else:
        print("ℹ️ Skipping calibration plot (no valid data or not binary).")
        mean_pred, frac_pos, calib_error = None, None, None


    print("all_probs shape:", all_probs.shape)
    print("unique labels:", np.unique(all_labels))

    try:
        frac_pos, mean_pred = calibration_curve(all_labels, all_probs, n_bins=10)
        calib_error = np.mean(np.abs(frac_pos - mean_pred))
    except Exception:
        mean_pred, frac_pos, calib_error = None, None, None
    
    report_path = generate_visualizations(
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
        samples=samples_arr,
        labels_np=labels_np,
        losses=losses,
        pred_probs=pred_probs,
        color_palette=color_palette
    )

    return f"✅ Entrenamiento completado. Visualizaciones generadas en {report_path}"



# ============================================
# INTERFAZ DE GRADIO
# ============================================

with gr.Blocks(title="🐨 Sloths Pain Experiment Dashboard") as demo:
    gr.Markdown("## 🧠 Experimentos de Clasificación con PyTorch Lightning")

    with gr.Row():
        data_dir = gr.Textbox(label="📁 Directorio de datos", value="data")
        seed = gr.Number(label="🔢 Semilla aleatoria", value=42)
    
    with gr.Row():
        img_size = gr.Slider(64, 256, value=128, step=16, label="🖼️ Tamaño de imagen (px)")
        batch_size = gr.Slider(8, 64, value=32, step=4, label="📦 Batch size")
        epochs = gr.Slider(1, 10, value=3, step=1, label="🔁 Número de épocas")
    
    color_palette = gr.Dropdown(
        ["default", "seaborn", "ggplot", "bmh", "dark_background"],
        value="default",
        label="🎨 Estilo de visualización"
    )

    run_btn = gr.Button("🚀 Ejecutar experimento")
    output = gr.Textbox(label="Salida del experimento")

    run_btn.click(
        fn=run_experiment,
        inputs=[data_dir, seed, img_size, batch_size, epochs, color_palette],
        outputs=output
    )

demo.launch()


