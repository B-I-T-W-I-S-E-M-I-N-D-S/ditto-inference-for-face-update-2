"""
core/canonical_template.py
==========================
Canonical face geometry manager for Ditto.

Loads a template face image once at initialisation, extracts its keypoints
via Source2Info, and exposes a blend method that merges template geometry
with user geometry at a configurable strength (alpha).

Key design choices:
  - Motion/lip keypoints use a lower alpha (mouth_alpha) to preserve lip-sync.
  - Keypoint deltas are clamped to avoid extreme deformation artefacts.
  - The sc (source-condition) vector is also blended so the DiT receives a
    consistent condition that matches the overridden geometry.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.atomic_components.source2info import Source2Info


# ─────────────────────────────────────────────────────────────────────────────
# Mouth / lip keypoint indices (21-point LivePortrait convention used by Ditto)
# exp shape: (1, 63) = 21 pts × 3 coords
# ─────────────────────────────────────────────────────────────────────────────
_LIP_INDICES = [6, 12, 14, 17, 19, 20]   # rows in the 21×3 layout


def _make_lip_weight(alpha_face: float, alpha_mouth: float,
                     dtype=np.float32) -> np.ndarray:
    """
    Build a per-element weight vector for the exp array (shape 1×63).
    Lip keypoints use alpha_mouth; all others use alpha_face.
    """
    w = np.full((21, 3), alpha_face, dtype=dtype)
    w[_LIP_INDICES] = alpha_mouth
    return w.reshape(1, -1)                 # (1, 63)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_rgb(image_path: str) -> np.ndarray:
    """Load image as RGB uint8 (keep original resolution for face detection)."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _img_to_bchw256(img_rgb: np.ndarray) -> np.ndarray:
    """uint8 HWC → float32 BCHW [0,1]."""
    return (img_rgb.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)


def _scalar_blend(t_val: np.ndarray, u_val: np.ndarray,
                  alpha: float) -> np.ndarray:
    """alpha * template + (1 - alpha) * user."""
    return alpha * t_val + (1.0 - alpha) * u_val


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class CanonicalTemplate:
    """
    Manages a canonical (template) face geometry for Ditto's inference.

    Args:
        template_path : Path to the template face image (any format/size).
        source2info   : Initialised Source2Info instance (shared with pipeline).
        alpha         : Blend strength for geometry. 0 = pure user, 1 = pure
                        template. Default: 0.7.
        mouth_alpha   : Blend strength for the mouth/lip exp keypoints.
                        Lower = better lip-sync fidelity. Default: 0.3.
        clamp_val     : Max absolute value of per-keypoint delta between
                        template and user (in normalised 3D space). Prevents
                        extreme shape warping. Default: 0.3. Set to None to
                        disable clamping.
        crop_kwargs   : Crop parameters forwarded to Source2Info (same as
                        pipeline defaults).
    """

    def __init__(
        self,
        template_path: str,
        source2info: "Source2Info",
        alpha: float = 0.7,
        mouth_alpha: float = 0.3,
        clamp_val: float | None = 0.3,
        crop_kwargs: dict | None = None,
    ):
        self.alpha = float(alpha)
        self.mouth_alpha = float(mouth_alpha)
        self.clamp_val = clamp_val

        # Build per-element weight for exp blending
        self._exp_w = _make_lip_weight(self.alpha, self.mouth_alpha)

        crop_kw = crop_kwargs or {
            "crop_scale": 2.3,
            "crop_vx_ratio": 0.0,
            "crop_vy_ratio": -0.125,
            "crop_flag_do_rot": True,
        }

        print(f"[CanonicalTemplate] Loading template from: {template_path}")
        template_rgb = _load_rgb(template_path)

        # Extract template geometry using the SAME Source2Info model
        try:
            t_info = source2info(template_rgb, last_lmk=None, **crop_kw)
        except Exception as exc:
            raise RuntimeError(
                f"[CanonicalTemplate] Failed to extract template geometry "
                f"from {template_path}: {exc}"
            ) from exc

        self._t_x_s_info: dict = t_info["x_s_info"]
        self._t_sc: np.ndarray = self._t_x_s_info["kp"].flatten()

        print(
            f"[CanonicalTemplate] Template ready. "
            f"alpha={self.alpha:.2f}  mouth_alpha={self.mouth_alpha:.2f}  "
            f"clamp={self.clamp_val}"
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def blend_x_s_info(self, user_x_s_info: dict) -> dict:
        """
        Return a blended keypoint dict:
          blended = alpha * template + (1 - alpha) * user

        Each key is blended independently. The exp (expression/deformation)
        array uses mouth_alpha for lip-related entries to preserve lip-sync.
        Keypoint ('kp') deltas are clamped when clamp_val is set.

        Args:
            user_x_s_info : dict from MotionExtractor for the input face.

        Returns:
            dict with same keys; values are blended np.ndarrays.
        """
        t = self._t_x_s_info
        u = user_x_s_info
        blended: dict = {}

        for key in u:
            tv = t.get(key)
            uv = u[key]

            if tv is None:
                # Key absent in template → keep user value unchanged
                blended[key] = uv
                continue

            if key == "exp":
                # Per-element weight for lip-sync preservation
                blended[key] = self._exp_w * tv + (1.0 - self._exp_w) * uv

            elif key == "kp":
                # Clamp delta to avoid extreme shape warping
                delta = tv - uv
                if self.clamp_val is not None:
                    delta = np.clip(delta, -self.clamp_val, self.clamp_val)
                blended[key] = uv + self.alpha * delta

            else:
                # scale, pitch, yaw, roll, t  — simple linear blend
                blended[key] = _scalar_blend(tv, uv, self.alpha)

        return blended

    def blend_sc(self, user_sc: np.ndarray) -> np.ndarray:
        """
        Blend the source-condition vector (sc = kp.flatten()) that feeds the
        DiT. Consistent blending keeps the audio→motion model coherent with
        the overridden geometry.

        Args:
            user_sc : np.ndarray (63,)

        Returns:
            np.ndarray (63,)  — blended sc
        """
        delta = self._t_sc - user_sc
        if self.clamp_val is not None:
            delta = np.clip(delta, -self.clamp_val, self.clamp_val)
        return user_sc + self.alpha * delta
