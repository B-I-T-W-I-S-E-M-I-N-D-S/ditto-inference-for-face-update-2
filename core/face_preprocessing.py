"""
core/face_preprocessing.py
==========================
Face preprocessing for canonical face normalization.

Two modes (auto-selected based on availability):
  - "bisenet"   : Semantic segmentation via BiSeNet-V1 (ResNet18 backbone).
                  Precisely removes hair, ears, background.
                  Requires: torch, torchvision + pretrained weights .pth
                  Weights: https://github.com/zllrunning/face-parsing.PyTorch
                           (download 79999_iter.pth to checkpoints/bisenet/)
  - "heuristic" : Lightweight ellipse-based soft mask. Zero extra dependencies.
                  Automatically used as fallback if BiSeNet weights are absent.

Input/output convention (matches AppearanceExtractor):
    image : np.ndarray  float32  shape (1, 3, 256, 256)  range [0, 1]  BCHW RGB
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic mask
# ─────────────────────────────────────────────────────────────────────────────

def _build_heuristic_mask(h: int = 256, w: int = 256) -> np.ndarray:
    """
    Soft elliptical face-region mask, shape (H, W, 1), float32 in [0, 1].

    The ellipse is tuned for Ditto's 256×256 aligned-face crops:
      - Centre slightly below mid-height to include chin but exclude forehead hair
      - Semi-axes crop out ears (sides) and top-of-hair (top)
    """
    canvas = np.zeros((h, w), dtype=np.float32)
    cx = w // 2
    cy = int(h * 0.54)          # ~138 px: includes chin, trims top hair
    ax = int(w * 0.40)          # ~102 px: trims ears on each side
    ay = int(h * 0.44)          # ~112 px: trims hair crown + neck
    cv2.ellipse(canvas, (cx, cy), (ax, ay),
                angle=0, startAngle=0, endAngle=360,
                color=1.0, thickness=-1)

    # Soft boundary via Gaussian blur
    k = max(3, (min(h, w) // 20) | 1)          # ~13 px kernel for 256 px image
    canvas = cv2.GaussianBlur(canvas, (k * 4 + 1, k * 4 + 1), k * 2)
    canvas = np.clip(canvas, 0.0, 1.0)
    return canvas[:, :, None]                   # (H, W, 1)


_HEURISTIC_MASK_CACHE: dict[tuple, np.ndarray] = {}


def _get_heuristic_mask(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _HEURISTIC_MASK_CACHE:
        _HEURISTIC_MASK_CACHE[key] = _build_heuristic_mask(h, w)
    return _HEURISTIC_MASK_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# BiSeNet label sets  (face-parsing-pytorch / CelebAMask-HQ convention)
# ─────────────────────────────────────────────────────────────────────────────
#  0: background   1: skin        2: l_brow     3: r_brow     4: l_eye
#  5: r_eye        6: eye_glass   7: l_ear      8: r_ear      9: ear_ring
# 10: nose        11: mouth      12: u_lip     13: l_lip     14: neck
# 15: necklace    16: cloth      17: hair      18: hat

_BISENET_FACE_LABELS = frozenset({1, 2, 3, 4, 5, 6, 10, 11, 12, 13})
# Kept: skin, brows, eyes, eyeglasses, nose, mouth, lips
# Removed: ears (7,8,9), neck (14), hair (17,18), background (0), cloth (16)


# ─────────────────────────────────────────────────────────────────────────────
# BiSeNet model (lazy-loaded; ResNet18 backbone, 19-class head)
# ─────────────────────────────────────────────────────────────────────────────

def _try_load_bisenet(model_path: str, device: str) -> object | None:
    """
    Attempt to load the BiSeNet-V1 face-parsing model.
    Returns the model on success, None on any failure (missing dep / weights).
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18

        class _ConvBnRelu(nn.Module):
            def __init__(self, ic, oc, k=3, s=1, p=1, bias=False):
                super().__init__()
                self.conv = nn.Conv2d(ic, oc, k, s, p, bias=bias)
                self.bn   = nn.BatchNorm2d(oc)
                self.relu = nn.ReLU(inplace=True)
            def forward(self, x):
                return self.relu(self.bn(self.conv(x)))

        class _ARM(nn.Module):
            """Attention Refinement Module."""
            def __init__(self, ic, oc):
                super().__init__()
                self.conv = _ConvBnRelu(ic, oc)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.gate = nn.Sequential(
                    nn.Conv2d(oc, oc, 1, bias=False),
                    nn.BatchNorm2d(oc),
                    nn.Sigmoid(),
                )
            def forward(self, x):
                x = self.conv(x)
                w = self.gate(self.pool(x))
                return x * w

        class _FFM(nn.Module):
            """Feature Fusion Module."""
            def __init__(self, ic, oc):
                super().__init__()
                self.conv = _ConvBnRelu(ic, oc)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.se   = nn.Sequential(
                    nn.Conv2d(oc, oc // 4, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(oc // 4, oc, 1, bias=False),
                    nn.Sigmoid(),
                )
            def forward(self, sp, cp):
                x = torch.cat([sp, cp], dim=1)
                x = self.conv(x)
                w = self.se(self.pool(x))
                return x + x * w

        class BiSeNetV1(nn.Module):
            def __init__(self, n_classes: int = 19):
                super().__init__()
                backbone = resnet18(weights=None)
                # Spatial path
                self.sp = nn.Sequential(
                    _ConvBnRelu(3,  64, 3, 2, 1),
                    _ConvBnRelu(64, 128, 3, 2, 1),
                    _ConvBnRelu(128, 256, 3, 2, 1),
                )
                # Context path  — reuse ResNet18 stages
                self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1,
                                            backbone.relu, backbone.maxpool,
                                            backbone.layer1)
                self.layer2 = backbone.layer2
                self.layer3 = backbone.layer3
                self.layer4 = backbone.layer4
                # ARM on last two stages
                self.arm3 = _ARM(256, 128)
                self.arm4 = _ARM(512, 128)
                # Global average context
                self.gap   = nn.AdaptiveAvgPool2d(1)
                self.gap_c = _ConvBnRelu(512, 128, 1, 1, 0)
                # FFM
                self.ffm = _FFM(256 + 128, 256)
                # Classifier head
                self.cls  = nn.Conv2d(256, n_classes, 1)
                self.cls3 = nn.Conv2d(128, n_classes, 1)
                self.cls4 = nn.Conv2d(128, n_classes, 1)

            def forward(self, x):
                H, W = x.shape[2], x.shape[3]
                # Spatial path
                sp = self.sp(x)                     # /8
                # Context path
                f1 = self.layer1(x)
                f2 = self.layer2(f1)
                f3 = self.layer3(f2)
                f4 = self.layer4(f3)
                gap = self.gap_c(self.gap(f4))
                # ARM + upsample
                a4 = self.arm4(f4) + gap
                a4 = nn.functional.interpolate(a4, scale_factor=2, mode='bilinear', align_corners=True)
                a3 = self.arm3(f3) + a4
                a3 = nn.functional.interpolate(a3, size=sp.shape[2:], mode='bilinear', align_corners=True)
                # FFM
                out = self.ffm(sp, a3)
                out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
                return self.cls(out)

        model = BiSeNetV1(n_classes=19)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        # Handle various checkpoint formats
        if isinstance(state, dict):
            sd = state.get("state_dict", state.get("model", state))
        else:
            sd = state
        # Strip common prefixes
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("module.", "").replace("model.", "")
            new_sd[nk] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if len(missing) > 30:
            raise RuntimeError(
                f"Too many missing keys ({len(missing)}) — "
                "checkpoint architecture may not match. "
                "Please use the 79999_iter.pth from face-parsing.PyTorch."
            )
        model.eval().to(device)
        return model

    except Exception as exc:
        warnings.warn(
            f"[FacePreprocessor] BiSeNet load failed: {exc}\n"
            "Falling back to heuristic ellipse mask.",
            stacklevel=3,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Background fill
# ─────────────────────────────────────────────────────────────────────────────

def fill_background(
    image: np.ndarray,
    mask: np.ndarray,
    method: Literal["mean", "blur", "black"] = "mean",
) -> np.ndarray:
    """
    Fill the unmasked (background) region of `image`.

    Args:
        image  : float32  (1, 3, H, W)  [0, 1]  BCHW
        mask   : float32  (H, W, 1)     [0, 1]  face=1, background=0
        method : fill strategy

    Returns:
        float32  (1, 3, H, W)  [0, 1]
    """
    hwc = image[0].transpose(1, 2, 0)          # (H, W, 3)
    w = mask[:, :, 0]                           # (H, W)

    if method == "mean":
        total = float(w.sum()) + 1e-6
        mean_color = (hwc * w[:, :, None]).sum(axis=(0, 1)) / total  # (3,)
        fill = np.broadcast_to(mean_color[None, None, :], hwc.shape).copy()
    elif method == "blur":
        k_sz = max(3, (min(hwc.shape[:2]) // 8) | 1)
        fill = cv2.GaussianBlur(hwc, (k_sz * 4 + 1, k_sz * 4 + 1), k_sz * 2)
    else:  # "black"
        fill = np.zeros_like(hwc)

    result = np.clip(hwc * mask + fill * (1.0 - mask), 0.0, 1.0)  # (H, W, 3)
    return result.transpose(2, 0, 1)[None].astype(np.float32)       # (1, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class FacePreprocessor:
    """
    Removes hair, ears, and background from a 256×256 cropped face image.

    Args:
        mode        : "bisenet" (preferred) or "heuristic" (fallback).
                      When mode="bisenet" but weights are unavailable,
                      automatically degrades to "heuristic".
        bisenet_path: Path to BiSeNet .pth weights.
                      Defaults to "checkpoints/bisenet/79999_iter.pth".
        device      : "cuda" or "cpu".
        fill_method : How to fill removed regions ("mean", "blur", "black").
        blur_edge_k : Extra Gaussian blur on the BiSeNet mask boundary (pixels).
                      Smooths the hard segmentation edge. 0 = disable.
    """

    def __init__(
        self,
        mode: Literal["bisenet", "heuristic"] = "bisenet",
        bisenet_path: str = "checkpoints/bisenet/79999_iter.pth",
        device: str = "cpu",
        fill_method: Literal["mean", "blur", "black"] = "mean",
        blur_edge_k: int = 7,
    ):
        self.fill_method = fill_method
        self.blur_edge_k = blur_edge_k
        self.device = device

        self._bisenet = None
        self._mode = "heuristic"

        if mode == "bisenet":
            if os.path.isfile(bisenet_path):
                model = _try_load_bisenet(bisenet_path, device)
                if model is not None:
                    self._bisenet = model
                    self._mode = "bisenet"
                    try:
                        import torch
                        self._torch = torch
                    except ImportError:
                        pass
                    print(f"[FacePreprocessor] BiSeNet loaded from {bisenet_path}")
            else:
                warnings.warn(
                    f"[FacePreprocessor] BiSeNet weights not found at '{bisenet_path}'. "
                    "Using heuristic ellipse mask instead.\n"
                    "To enable BiSeNet: download 79999_iter.pth from\n"
                    "  https://github.com/zllrunning/face-parsing.PyTorch\n"
                    f"and place it at: {bisenet_path}",
                    stacklevel=2,
                )
        else:
            print("[FacePreprocessor] Using heuristic ellipse mask.")

    @property
    def mode(self) -> str:
        return self._mode

    def _bisenet_mask(self, image_bchw: np.ndarray) -> np.ndarray:
        """
        Run BiSeNet on a float32 BCHW image and return a soft mask (H, W, 1).
        """
        import torch
        _, _, H, W = image_bchw.shape

        # Normalise to ImageNet stats expected by ResNet backbone
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
        inp  = (image_bchw - mean) / std

        t = torch.from_numpy(inp).to(self.device)
        with torch.no_grad():
            logits = self._bisenet(t)           # (1, 19, H, W)

        labels = logits[0].argmax(dim=0).cpu().numpy()  # (H, W)

        # Binary mask: 1 = face region
        binary = np.zeros((H, W), dtype=np.float32)
        for lbl in _BISENET_FACE_LABELS:
            binary[labels == lbl] = 1.0

        # Soft boundary
        if self.blur_edge_k > 0:
            k = (self.blur_edge_k * 2 + 1)
            binary = cv2.GaussianBlur(binary, (k, k), self.blur_edge_k * 0.5)
            binary = np.clip(binary, 0.0, 1.0)

        return binary[:, :, None]               # (H, W, 1)

    def __call__(self, image_bchw: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bchw : float32  (1, 3, H, W)  [0, 1]  RGB

        Returns:
            float32  (1, 3, H, W)  [0, 1]  — face region preserved, rest filled
        """
        _, _, H, W = image_bchw.shape

        if self._mode == "bisenet":
            mask = self._bisenet_mask(image_bchw)
        else:
            mask = _get_heuristic_mask(H, W)

        return fill_background(image_bchw, mask, method=self.fill_method)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_face(
    image_bchw: np.ndarray,
    preprocessor: FacePreprocessor,
) -> np.ndarray:
    """Convenience wrapper. Equivalent to ``preprocessor(image_bchw)``."""
    return preprocessor(image_bchw)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT-STAGE POST-PROCESSING
# These functions operate on the FINAL rendered uint8 frames so that
# the *visible* output video has hair / ears / background removed.
# ─────────────────────────────────────────────────────────────────────────────

def _build_render_oval_mask(h: int = 512, w: int = 512) -> np.ndarray:
    """
    Tight face-oval mask for the rendered 512×512 frame.

    The rendered frame already has the face filling most of the canvas.
    This mask aggressively removes:
      - Top ~25 % (hair crown)
      - Side ~20 % (ears)
      - Very bottom (chin tip / neck)

    Returns float32 (H, W, 1) in [0, 1].
    """
    canvas = np.zeros((h, w), dtype=np.float32)
    cx = w // 2
    cy = int(h * 0.58)        # shift centre down → cut more hair from top
    ax = int(w * 0.34)        # narrow x → cut ears
    ay = int(h * 0.42)        # narrower y → cut hair crown + chin/neck
    cv2.ellipse(canvas, (cx, cy), (ax, ay),
                angle=0, startAngle=0, endAngle=360,
                color=1.0, thickness=-1)
    # Smooth edge (15 px feather on 512 px image)
    canvas = cv2.GaussianBlur(canvas, (61, 61), 15)
    return np.clip(canvas, 0.0, 1.0)[:, :, None]   # (H, W, 1)


_RENDER_OVAL_CACHE: dict[tuple, np.ndarray] = {}


def _get_render_oval_mask(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _RENDER_OVAL_CACHE:
        _RENDER_OVAL_CACHE[key] = _build_render_oval_mask(h, w)
    return _RENDER_OVAL_CACHE[key]


def estimate_skin_color(img_rgb: np.ndarray, M_c2o: np.ndarray | None = None) -> np.ndarray:
    """
    Estimate mean skin color from the face region of a source image.

    Args:
        img_rgb : uint8 HWC RGB — full-resolution source frame
        M_c2o   : 3×3 homography (crop→original).  When provided, the face
                  centre is located precisely; otherwise a centre-of-image
                  heuristic is used.

    Returns:
        np.ndarray  shape (3,)  uint8  — mean skin color [R, G, B]
    """
    h, w = img_rgb.shape[:2]

    if M_c2o is not None:
        # Transform crop centre (256, 256) → original image coords
        crop_centre = np.array([256.0, 256.0, 1.0])
        pt = M_c2o @ crop_centre
        cx, cy = int(pt[0] / pt[2]), int(pt[1] / pt[2])
    else:
        cx, cy = w // 2, h // 2

    # Sample a 10 % radius region around face centre (should be mostly skin)
    r = max(10, min(h, w) // 10)
    x1, x2 = max(0, cx - r), min(w, cx + r)
    y1, y2 = max(0, cy - r), min(h, cy + r)

    if x2 > x1 and y2 > y1:
        region = img_rgb[y1:y2, x1:x2].astype(np.float32)
        skin = region.mean(axis=(0, 1))
    else:
        skin = np.array([200.0, 175.0, 150.0])   # neutral fallback

    return np.clip(skin, 0, 255).astype(np.uint8)


def clean_output_frame(
    render_img: np.ndarray,
    skin_color: np.ndarray,
    face_preprocessor: "FacePreprocessor | None" = None,
) -> np.ndarray:
    """
    Post-process a rendered 512×512 face frame:
      1. Apply a tight face-oval mask  →  removes rendered hair / ears
      2. Fill masked-out pixels with `skin_color`

    This is the function that makes the final *output video* look clean.

    Args:
        render_img        : uint8 HWC (H, W, 3) — raw rendered face frame
        skin_color        : uint8 (3,) [R, G, B] — fill colour
        face_preprocessor : optional FacePreprocessor; when provided its
                            BiSeNet mask is used instead of the oval heuristic
                            for a more precise boundary.

    Returns:
        uint8 HWC (H, W, 3)
    """
    H, W = render_img.shape[:2]

    # ── Step 1: get face mask ────────────────────────────────────────────────
    if face_preprocessor is not None and face_preprocessor.mode == "bisenet":
        # Run BiSeNet on the rendered frame for precise segmentation
        bchw = (render_img.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)
        mask = face_preprocessor._bisenet_mask(bchw)          # (H, W, 1)
    else:
        mask = _get_render_oval_mask(H, W)                    # (H, W, 1)

    # ── Step 2: fill non-face pixels with skin color ─────────────────────────
    img_f   = render_img.astype(np.float32)                   # (H, W, 3)
    fill_f  = skin_color.astype(np.float32)[None, None, :]    # (1, 1, 3)
    result  = img_f * mask + fill_f * (1.0 - mask)
    return np.clip(result, 0, 255).astype(np.float32)


def make_skin_background(h: int, w: int, skin_color: np.ndarray) -> np.ndarray:
    """
    Create a solid skin-colored HWC uint8 frame of size (h, w).

    Used to replace the original source frame so that PutBack composites
    the rendered face onto a clean skin-colored background instead of the
    original scene (which contains hair / background objects).
    """
    bg = np.empty((h, w, 3), dtype=np.uint8)
    bg[:] = skin_color
    return bg

