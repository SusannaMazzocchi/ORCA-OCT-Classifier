"""
predictor.py
------------
Image preprocessing, inference, and saliency map generation.

OOD detection uses two complementary signals:
  1. MC Dropout  — runs N stochastic forward passes and measures prediction
                   variance (epistemic uncertainty / Gal & Ghahramani 2016).
  2. Entropy     — Shannon entropy of the mean probability distribution
                   (aleatoric uncertainty).

Model-specific preprocessing:
  CustomCNN / EfficientNet / MIRAGE:
      Grayscale → resize 512×512 → normalise (mean=0.5, std=0.5)
  MedGemma:
      RGB → SigLIP processor (resize 384×384, normalise to [-1, 1])
      The SigLIP processor is loaded from the same HF repo as the encoder
      and is cached locally after the first download.

Saliency maps:
  CustomCNN / EfficientNet  — Grad-CAM (pytorch-grad-cam)
  MIRAGE                    — Attention Rollout (Abnar & Zuidema, 2020)
  MedGemma                  — Attention Rollout on SigLIP vision encoder
"""

import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

INPUT_SIZE = 512
NORM_MEAN  = [0.5]
NORM_STD   = [0.5]

_INFERENCE_TRANSFORM = T.Compose([
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# Entropy thresholds (3-class problem; max entropy = log(3) ≈ 1.099 nats)
ENTROPY_THRESHOLD = 0.70
STD_THRESHOLD     = 0.12

# SigLIP input size used by MedGemma's vision encoder
SIGLIP_SIZE = 384


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(pil_image: Image.Image, model_name: str = "EfficientNet") -> tuple:
    """
    Apply the full preprocessing pipeline to a PIL image.

    Returns (tensor, steps_imgs) where:
      - tensor      is ready for model inference
      - steps_imgs  is an ordered dict of PIL images for the pipeline expander

    For MedGemma the tensor is a dict {'pixel_values': Tensor(1, 3, 384, 384)}
    so that the MedGemmaClassifier.forward() receives the right input format.
    For all other models the tensor is Tensor(1, 1, 512, 512).
    """
    if model_name == "MedGemma":
        return _preprocess_medgemma(pil_image)
    else:
        return _preprocess_standard(pil_image)


def _preprocess_standard(pil_image: Image.Image) -> tuple:
    """Standard grayscale pipeline for CNN / MIRAGE models."""
    gray    = pil_image.convert("L")
    resized = gray.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)

    arr    = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0)
    tensor = _INFERENCE_TRANSFORM(tensor)
    tensor = tensor.unsqueeze(0)   # (1, 1, 512, 512)

    arr_display = np.clip(arr, 0, 1)
    steps_imgs = {
        "Original":        pil_image,
        "Grayscale":       gray,
        "Resize 512×512":  resized,
        "Normalised":      Image.fromarray((arr_display * 255).astype(np.uint8)),
    }
    return tensor, steps_imgs


def _preprocess_medgemma(pil_image: Image.Image) -> tuple:
    """
    MedGemma / SigLIP preprocessing pipeline.

    SigLIP expects:
      - 3-channel RGB input
      - Resized to 384×384 (SigLIP-So400m native resolution in MedGemma-4b)
      - Pixel values normalised to [-1, 1]  (mean=0.5, std=0.5 on [0,1] scale)

    We replicate the processor's transform manually so we can also produce
    the intermediate visualisation steps for the pipeline expander.
    """
    rgb     = pil_image.convert("RGB")
    resized = rgb.resize((SIGLIP_SIZE, SIGLIP_SIZE), Image.Resampling.LANCZOS)

    arr = np.array(resized, dtype=np.float32) / 255.0        # [0, 1]
    # SigLIP normalisation: (x - 0.5) / 0.5  →  [-1, 1]
    arr_norm = (arr - 0.5) / 0.5

    tensor = torch.tensor(arr_norm).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 384, 384)

    steps_imgs = {
        "Original":          pil_image,
        "RGB convert":       rgb,
        f"Resize {SIGLIP_SIZE}×{SIGLIP_SIZE}": resized,
        "Normalised [-1,1]": Image.fromarray(
            np.clip((arr * 255), 0, 255).astype(np.uint8)
        ),
    }
    return tensor, steps_imgs


# ══════════════════════════════════════════════════════════════════════════════
# Uncertainty helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector (in nats)."""
    p = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def _enable_mc_dropout(model: nn.Module) -> None:
    """
    Keep BatchNorm in eval mode but re-enable all Dropout layers for
    stochastic MC passes. Works for CNNs, MIRAGE, and MedGemma head.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _run_forward(model: nn.Module, tensor, device: torch.device) -> torch.Tensor:
    """
    Unified forward pass that handles both standard tensor input
    and MedGemma's pixel_values tensor.
    """
    if isinstance(tensor, dict):
        # MedGemma path
        pixel_values = tensor["pixel_values"].to(device)
        return model(pixel_values)
    else:
        return model(tensor.to(device))


# ══════════════════════════════════════════════════════════════════════════════
# Standard single-pass inference
# ══════════════════════════════════════════════════════════════════════════════

def predict_image(
    model:       nn.Module,
    tensor,
    class_names: list,
    device:      torch.device,
) -> dict:
    """Single deterministic forward pass."""
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = _run_forward(model, tensor, device)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    elapsed    = time.perf_counter() - t0
    pred_idx   = int(probs.argmax())
    confidence = float(probs.max())
    entropy    = compute_entropy(probs)
    return {
        "predicted_class":  class_names[pred_idx],
        "predicted_idx":    pred_idx,
        "probabilities":    [float(p) for p in probs],
        "confidence":       confidence,
        "entropy":          entropy,
        "mean_std":         0.0,
        "mc_dropout":       False,
        "inference_time_s": elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MC Dropout inference
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_uncertainty(
    model:       nn.Module,
    tensor,
    class_names: list,
    device:      torch.device,
    n_passes:    int = 20,
) -> dict:
    """
    Monte Carlo Dropout inference.
    For MedGemma, only the MLP head dropout is stochastic (the frozen
    vision encoder has no dropout layers active at eval time).
    """
    _enable_mc_dropout(model)
    all_probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_passes):
            logits = _run_forward(model, tensor, device)
            p      = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            all_probs.append(p)
    elapsed = time.perf_counter() - t0
    model.eval()

    all_probs  = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)
    std_probs  = all_probs.std(axis=0)
    pred_idx   = int(mean_probs.argmax())
    confidence = float(mean_probs.max())
    entropy    = compute_entropy(mean_probs)
    mean_std   = float(std_probs.mean())
    return {
        "predicted_class":  class_names[pred_idx],
        "predicted_idx":    pred_idx,
        "probabilities":    [float(p) for p in mean_probs],
        "std_per_class":    [float(s) for s in std_probs],
        "confidence":       confidence,
        "entropy":          entropy,
        "mean_std":         mean_std,
        "mc_dropout":       True,
        "n_passes":         n_passes,
        "inference_time_s": elapsed,
    }


def is_ood(result: dict, conf_threshold: float) -> tuple[bool, list[str]]:
    """
    OOD detection: checks confidence, entropy, and MC variance.
    Returns (flagged: bool, reasons: list[str]).
    """
    reasons = []
    if result["confidence"] < conf_threshold:
        reasons.append(
            f"Low confidence: {result['confidence'] * 100:.1f}% "
            f"(threshold: {conf_threshold * 100:.0f}%)"
        )
    if result["entropy"] > ENTROPY_THRESHOLD:
        max_entropy = np.log(3)
        reasons.append(
            f"High prediction entropy: {result['entropy']:.3f} nats "
            f"(threshold: {ENTROPY_THRESHOLD:.2f}, max possible: {max_entropy:.3f})"
        )
    if result["mc_dropout"] and result["mean_std"] > STD_THRESHOLD:
        reasons.append(
            f"High MC Dropout variance: mean std = {result['mean_std']:.3f} "
            f"(threshold: {STD_THRESHOLD:.2f})"
        )
    return len(reasons) > 0, reasons


# ══════════════════════════════════════════════════════════════════════════════
# Saliency maps
# ══════════════════════════════════════════════════════════════════════════════

def _get_target_layer(model: nn.Module, model_name: str):
    if model_name == "CustomCNN":
        return [model.features[8]]
    elif model_name == "EfficientNet":
        return [model.base.features[-1]]
    else:
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Conv2d):
                return [layer]
        return None


def _attention_rollout_mirage(model: nn.Module, tensor: torch.Tensor,
                               device: torch.device) -> np.ndarray:
    """
    Attention Rollout for MIRAGE (ViT-Base encoder).
    Hooks into model.model.encoder blocks and rolls out CLS attention.
    Returns a (512, 512) saliency map normalised to [0, 1].
    """
    tensor_dev = tensor.to(device)
    attn_maps  = []
    hooks = []
    for block in model.model.encoder:
        def _hook(module, inp, out, _store=attn_maps):
            x  = inp[0]
            B, N, C = x.shape
            nh = module.num_heads
            hd = module.head_dim
            qkv = module.qkv(x).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            attn = attn.softmax(dim=-1)
            _store.append(attn.detach().cpu())
        h = block.attn.register_forward_hook(_hook)
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        _ = model(tensor_dev)
    for h in hooks:
        h.remove()

    return _rollout_to_map(attn_maps, out_size=INPUT_SIZE)


def _attention_rollout_siglip(model: nn.Module, tensor: torch.Tensor,
                               device: torch.device) -> np.ndarray:
    """
    Attention Rollout for the SigLIP vision encoder inside MedGemma.

    SigLIP uses a standard ViT architecture accessed via Hugging Face
    transformers. The attention weights are exposed through the
    output_attentions flag of the vision encoder.

    Returns a (512, 512) saliency map normalised to [0, 1].
    """
    try:
        pixel_values = tensor.to(device) if not isinstance(tensor, dict) \
                       else tensor["pixel_values"].to(device)

        vision_encoder = model.encoder   # SiglipVisionModel

        model.eval()
        with torch.no_grad():
            outputs = vision_encoder(
                pixel_values=pixel_values,
                output_attentions=True,
            )

        # outputs.attentions: tuple of (B, num_heads, N, N) per layer
        attn_maps = [a.cpu() for a in outputs.attentions]
        # SigLIP has no explicit CLS token; patch tokens cover the full image.
        # We use the mean attention over the first token as an approximation.
        rollout = _rollout_to_map_no_cls(attn_maps, out_size=INPUT_SIZE)
        return rollout

    except Exception:
        return None


def _rollout_to_map(attn_maps: list, out_size: int) -> np.ndarray:
    """
    Standard Attention Rollout with CLS token (for MIRAGE).
    attn_maps: list of (1, num_heads, N, N) tensors.
    Returns (out_size, out_size) float32 array in [0, 1].
    """
    rollout = None
    for attn in attn_maps:
        a = attn[0].mean(dim=0).numpy()
        a = a + np.eye(a.shape[0])
        a = a / a.sum(axis=-1, keepdims=True)
        rollout = a if rollout is None else a @ rollout

    if rollout is None:
        return None

    num_patches_1d = int(round(np.sqrt(rollout.shape[1] - 1)))
    patch_attn     = rollout[0, 1:].reshape(num_patches_1d, num_patches_1d)
    return _normalise_and_resize(patch_attn, out_size)


def _rollout_to_map_no_cls(attn_maps: list, out_size: int) -> np.ndarray:
    """
    Attention Rollout without a CLS token (for SigLIP in MedGemma).
    Uses mean attention of the first patch token as the saliency signal.
    attn_maps: list of (1, num_heads, N, N) tensors.
    Returns (out_size, out_size) float32 array in [0, 1].
    """
    rollout = None
    for attn in attn_maps:
        a = attn[0].mean(dim=0).numpy()
        a = a + np.eye(a.shape[0])
        a = a / a.sum(axis=-1, keepdims=True)
        rollout = a if rollout is None else a @ rollout

    if rollout is None:
        return None

    # Use mean over all token queries
    patch_attn = rollout.mean(axis=0)
    num_patches_1d = int(round(np.sqrt(patch_attn.shape[0])))
    patch_attn = patch_attn[:num_patches_1d ** 2].reshape(num_patches_1d, num_patches_1d)
    return _normalise_and_resize(patch_attn, out_size)


def _normalise_and_resize(patch_attn: np.ndarray, out_size: int) -> np.ndarray:
    p_min, p_max = patch_attn.min(), patch_attn.max()
    if p_max > p_min:
        patch_attn = (patch_attn - p_min) / (p_max - p_min)
    else:
        patch_attn = np.zeros_like(patch_attn)
    pil = Image.fromarray((patch_attn * 255).astype(np.uint8), mode="L")
    pil = pil.resize((out_size, out_size), Image.Resampling.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def generate_gradcam(
    model:        nn.Module,
    tensor,
    model_name:   str,
    device:       torch.device,
    target_class: int = None,
):
    """
    Generate a saliency map appropriate for the given model.

    MIRAGE     → Attention Rollout on ViT encoder
    MedGemma   → Attention Rollout on SigLIP vision encoder
    CNN models → Grad-CAM (pytorch-grad-cam)

    Returns np.ndarray (H, W) in [0, 1], or None on failure.
    """
    if model_name == "MIRAGE":
        try:
            return _attention_rollout_mirage(model, tensor, device)
        except Exception:
            return None

    if model_name == "MedGemma":
        try:
            return _attention_rollout_siglip(model, tensor, device)
        except Exception:
            return None

    # ── CNN models: Grad-CAM ──────────────────────────────────────────────────
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        return None

    target_layers = _get_target_layer(model, model_name)
    if target_layers is None:
        return None

    try:
        model.eval()
        tensor_dev = tensor.to(device)
        targets    = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=tensor_dev, targets=targets)
        return grayscale_cam[0]
    except Exception:
        return None
