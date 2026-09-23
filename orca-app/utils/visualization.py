"""
visualization.py:  Plotting utilities (Organic Heal light theme)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

BG_COLOR   = "#FAF9F6"
CARD_COLOR = "#FFFFFF"
TEXT_COLOR = "#2D3142"
INACTIVE   = "#DDD8D0"
CLASS_COLORS = {"AMD": "#F28482", "DME": "#E29578", "Normal": "#84A59D"}


def plot_confidence_bars(probabilities, class_names, predicted_idx):
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    fig.patch.set_facecolor(CARD_COLOR)
    ax.set_facecolor(CARD_COLOR)

    colors = [
        CLASS_COLORS.get(cls, "#84A59D") if i == predicted_idx else INACTIVE
        for i, cls in enumerate(class_names)
    ]
    bars = ax.barh(class_names, probabilities, color=colors, height=0.48)

    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%", va="center", ha="left",
            color=TEXT_COLOR, fontsize=10, fontweight="600",
        )

    ax.set_xlim(0, 1.2)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.tick_params(colors=TEXT_COLOR, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color(TEXT_COLOR)
        lbl.set_fontsize(10)

    fig.tight_layout(pad=0.3)
    return fig


def plot_uncertainty(result: dict):
    """
    Bar chart showing per-class mean probability with MC Dropout error bars.
    Returns None if std_per_class is not available.
    """
    if "std_per_class" not in result:
        return None

    n = len(result["probabilities"])
    labels   = ["AMD", "DME", "Normal"][:n]
    means    = np.array(result["probabilities"])
    stds     = np.array(result["std_per_class"])
    pred_idx = result["predicted_idx"]

    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    fig.patch.set_facecolor(CARD_COLOR)
    ax.set_facecolor(CARD_COLOR)

    colors = [
        CLASS_COLORS.get(lbl, "#84A59D") if i == pred_idx else INACTIVE
        for i, lbl in enumerate(labels)
    ]

    bars = ax.barh(labels, means, color=colors, height=0.45, xerr=stds,
                   error_kw={"ecolor": "#9499A8", "capsize": 3, "linewidth": 1.2})

    ax.set_xlim(0, 1.1)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.tick_params(colors=TEXT_COLOR, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color(TEXT_COLOR)
        lbl.set_fontsize(9.5)

    ax.set_title("MC Dropout uncertainty (error bars = std)", fontsize=8,
                 color="#6B7280", pad=4)
    fig.tight_layout(pad=0.3)
    return fig


def plot_preprocessing_steps(steps_imgs):
    n = len(steps_imgs)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.7, 2.7))
    fig.patch.set_facecolor(BG_COLOR)

    for ax, (title, img) in zip(axes, steps_imgs.items()):
        display = img.convert("RGB") if img.mode != "L" else img
        cmap    = None if img.mode != "L" else "gray"
        ax.imshow(display, cmap=cmap)
        ax.set_title(title, color=TEXT_COLOR, fontsize=8.5, pad=4, fontweight="500")
        ax.axis("off")

    for i in range(n - 1):
        fig.text(
            (i + 1) / n - 0.005, 0.5, "›",
            ha="center", va="center",
            color="#84A59D", fontsize=18, fontweight="bold",
            transform=fig.transFigure,
        )

    fig.tight_layout(pad=0.4)
    return fig


def build_gradcam_figure(original_pil, heatmap):
    gray     = original_pil.convert("L").resize((512, 512), Image.Resampling.LANCZOS)
    gray_arr = np.array(gray)

    colormap    = cm.get_cmap("RdYlGn_r")
    heatmap_rgb = colormap(heatmap)[:, :, :3]
    gray_rgb    = np.stack([gray_arr / 255.0] * 3, axis=-1)
    overlay     = np.clip(0.55 * gray_rgb + 0.45 * heatmap_rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    fig.patch.set_facecolor(BG_COLOR)

    for ax, (data, cmap, title) in zip(axes, [
        (gray_arr,    "gray", "Original scan"),
        (heatmap_rgb, None,   "Attention map"),
        (overlay,     None,   "Overlay"),
    ]):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="500")
        ax.axis("off")

    sm   = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention", color=TEXT_COLOR, fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=7.5)
    cbar.outline.set_visible(False)

    fig.tight_layout(pad=0.4)
    return fig


def pil_to_thumbnail(pil_image, size=(72, 72)):
    thumb = pil_image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    return thumb
