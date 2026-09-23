"""
loader.py
---------
Model architecture definitions and weight loading.
Supports CustomCNN, EfficientNet-B0, MIRAGE (Vision Transformer),
and MedGemma (SigLIP vision encoder + fine-tuned MLP head).
Falls back to a dummy model if weights are not found.
"""

import os
import json
import torch
import torch.nn as nn
import streamlit as st

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "class_names":    ["AMD", "DME", "Normal"],
    "class_mapping":  {"AMD": 0, "DME": 1, "Normal": 2},
    "input_size":     512,
    "normalize_mean": 0.5,
    "normalize_std":  0.5,
    "ood_threshold":  0.65,
}

# ── MedGemma vision encoder ───────────────────────────────────────────────────
# google/siglip-so400m-patch14-384 is a public, NON-GATED SigLIP encoder.
# It uses the same So400m architecture as the vision tower inside medgemma-4b-it:
#   hidden_size=1152, num_heads=16, num_layers=27, patch_size=14
# The MLP head (best_MedGemma.pth) only receives pooler_output (1152-d),
# which is resolution-independent — the same embedding is produced regardless
# of whether the vision encoder runs at 384×384 or 896×896.
# Download size ≈ 3.5 GB (cached in ~/.cache/huggingface/hub/ after first run).
SIGLIP_HF_ID       = "google/siglip-so400m-patch14-384"
MEDGEMMA_EMBED_DIM = 1152   # SigLIP-So400m pooler_output dimension
SIGLIP_IMG_SIZE    = 384    # native input resolution for this checkpoint


# ══════════════════════════════════════════════════════════════════════════════
# CustomCNN
# ══════════════════════════════════════════════════════════════════════════════

class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# EfficientNet
# ══════════════════════════════════════════════════════════════════════════════

class EfficientNetOCT(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        import torchvision.models as tv_models
        self.base = tv_models.efficientnet_b0(weights=None)
        self.base.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)


# ══════════════════════════════════════════════════════════════════════════════
# MIRAGE — Vision Transformer (ViT-Base, patch 32, grayscale input)
# ══════════════════════════════════════════════════════════════════════════════

class _MirageAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.qkv        = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj       = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop  = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x    = self.proj(x)
        return x


class _MirageMLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(embed_dim, mlp_dim, bias=True)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(mlp_dim, embed_dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class _MirageEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = _MirageAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = _MirageMLP(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MirageInputAdapter(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, num_patches_1d: int):
        super().__init__()
        self.pos_emb = nn.Parameter(
            torch.zeros(1, embed_dim, num_patches_1d, num_patches_1d)
        )
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True
        )

    def forward(self, x):
        tokens = self.proj(x)
        tokens = tokens + self.pos_emb
        B, D, Hp, Wp = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens


class _MirageBackbone(nn.Module):
    def __init__(self, in_channels=1, patch_size=32, img_size=512,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        num_patches_1d = img_size // patch_size
        mlp_dim        = int(embed_dim * mlp_ratio)
        self.global_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_adapters = nn.ModuleDict({
            "bscan": _MirageInputAdapter(in_channels, patch_size, embed_dim, num_patches_1d)
        })
        self.encoder = nn.ModuleList([
            _MirageEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        B = x.shape[0]
        tokens = self.input_adapters["bscan"](x)
        cls    = self.global_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.encoder:
            tokens = block(tokens)
        return tokens


class MIRAGE(nn.Module):
    def __init__(self, num_classes: int = 3, embed_dim: int = 768, dropout: float = 0.0):
        super().__init__()
        self.model = _MirageBackbone(
            in_channels=1, patch_size=32, img_size=512,
            embed_dim=embed_dim, depth=12, num_heads=12,
            mlp_ratio=4.0, dropout=dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        tokens = self.model(x)
        cls    = tokens[:, 0]
        cls    = self.norm(cls)
        return self.head(cls)


# ══════════════════════════════════════════════════════════════════════════════
# MedGemma — SigLIP vision encoder + fine-tuned MLP classification head
#
# The checkpoint (best_MedGemma.pth) contains only the MLP head weights.
# The vision encoder is loaded from the public HF repo:
#   google/siglip-so400m-patch14-384
# Same So400m architecture (hidden_size=1152, 27 layers, patch_size=14).
# No HF token required. ~3.5 GB download, cached after first run.
#
# Pipeline:
#   RGB image (384×384, normalised to [-1, 1])
#   → SiglipVisionModel (frozen)
#   → pooler_output  shape (B, 1152)
#   → _MedGemmaHead  shape (B, 3)
#
# MLP head layout (matches best_MedGemma.pth state-dict):
#   net.0  Linear(1152 → 512)
#   net.1  LayerNorm(512)
#   net.2  ReLU              (no params)
#   net.3  Dropout(0.3)      (no params)
#   net.4  Linear(512 → 3)
# ══════════════════════════════════════════════════════════════════════════════

class _MedGemmaHead(nn.Module):
    def __init__(self, embed_dim: int = 1152, hidden_dim: int = 512,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MedGemmaClassifier(nn.Module):
    """
    Full MedGemma inference pipeline:
      frozen SigLIP vision encoder  →  trained MLP classification head.
    """
    def __init__(self, vision_encoder: nn.Module, head_state_dict: dict,
                 num_classes: int = 3):
        super().__init__()
        self.encoder = vision_encoder
        self.head    = _MedGemmaHead(
            embed_dim=MEDGEMMA_EMBED_DIM,
            hidden_dim=512,
            num_classes=num_classes,
        )
        self.head.load_state_dict(head_state_dict, strict=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values : (B, 3, 384, 384) normalised to [-1, 1]
        Returns      : (B, num_classes) logits
        """
        with torch.no_grad():
            outputs = self.encoder(pixel_values=pixel_values)
        embedding = outputs.pooler_output   # (B, 1152)
        return self.head(embedding)


# ══════════════════════════════════════════════════════════════════════════════
# Dummy (fallback when weights are missing)
# ══════════════════════════════════════════════════════════════════════════════

class _DummyModel(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.randn(x.shape[0], self.num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        cfg["ood_threshold"] = 0.65
        return cfg
    st.warning(f"config.json not found at {config_path}. Using default configuration.")
    return DEFAULT_CONFIG.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model(model_name: str, weights_dir: str, hf_token: str = None):
    """
    Load a model by name.
    Returns (model, is_real) — is_real is False only when a dummy is returned.
    """
    filename_map = {
        "CustomCNN":    ["best_CustomCNN.pth", "best_CostumCNN.pth"],
        "EfficientNet": ["best_EfficientNet.pth"],
        "MIRAGE":       ["MIRAGE_Base_OCT5k.pth"],
        "MedGemma":     ["best_MedGemma.pth"],
    }

    if model_name == "MedGemma":
        return _load_medgemma(weights_dir)

    if model_name == "CustomCNN":
        model = CustomCNN(num_classes=3)
    elif model_name == "EfficientNet":
        model = EfficientNetOCT(num_classes=3)
    elif model_name == "MIRAGE":
        model = MIRAGE(num_classes=3)
    else:
        return _DummyModel(num_classes=3).eval(), False

    for fname in filename_map.get(model_name, []):
        candidate = os.path.join(weights_dir, fname)
        if not os.path.exists(candidate):
            continue
        try:
            checkpoint = torch.load(candidate, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=True)
            model.to(DEVICE).eval()
            return model, True
        except Exception as e:
            st.warning(f"Could not load weights for {model_name}: {e}. Running in demo mode.")
            break

    return _DummyModel(num_classes=3).eval(), False


def _load_medgemma(weights_dir: str):
    """
    Load the MedGemma classifier.

    Step 1 — Vision encoder:
        Downloaded from google/siglip-so400m-patch14-384 (public, no token needed).
        ~3.5 GB, cached in ~/.cache/huggingface/hub/ after first download.

    Step 2 — Classification head:
        Loaded from best_MedGemma.pth (592,899 trainable parameters).
    """
    head_path = os.path.join(weights_dir, "best_MedGemma.pth")
    if not os.path.exists(head_path):
        st.warning("best_MedGemma.pth not found in weights/. Running in demo mode.")
        return _DummyModel(num_classes=3).eval(), False

    try:
        from transformers import SiglipVisionModel

        with st.spinner(
            "Loading SigLIP vision encoder (~3.5 GB on first run, cached after that)..."
        ):
            vision_encoder = SiglipVisionModel.from_pretrained(
                SIGLIP_HF_ID,
                torch_dtype=torch.float32,
            )

        vision_encoder = vision_encoder.to(DEVICE).eval()

        # Freeze all encoder parameters — it is used for feature extraction only
        for param in vision_encoder.parameters():
            param.requires_grad = False

        # Load the fine-tuned classification head
        head_state = torch.load(head_path, map_location=DEVICE, weights_only=False)

        model = MedGemmaClassifier(
            vision_encoder=vision_encoder,
            head_state_dict=head_state,
            num_classes=3,
        )
        model.to(DEVICE).eval()
        return model, True

    except Exception as e:
        st.error(
            f"Failed to load MedGemma vision encoder: {e}\n\n"
            "Common causes:\n"
            "• No internet connection on first run (cache not built yet)\n"
            "• transformers or huggingface_hub not installed "
            "(run: pip install transformers huggingface_hub)"
        )
        return _DummyModel(num_classes=3).eval(), False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
