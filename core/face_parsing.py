import numpy as np
import cv2
from .face_preprocessing import FacePreprocessor, _get_heuristic_mask

class FaceParser:
    """
    Extracts the inner facial regions (skin, eyes, nose, lips) and removes hair, ears, and background.
    """
    def __init__(self, bisenet_path="checkpoints/bisenet/79999_iter.pth", device="cuda"):
        self.preprocessor = FacePreprocessor(
            mode="bisenet",
            bisenet_path=bisenet_path,
            device=device,
            fill_method="mean",
            blur_edge_k=7
        )

    def get_mask(self, image_hwc: np.ndarray) -> np.ndarray:
        """
        Get the mask for the inner face.
        Args:
            image_hwc: np.ndarray, float32 or uint8, (H, W, 3)
        Returns:
            mask: np.ndarray, float32, (H, W, 1), range [0, 1]
        """
        H, W = image_hwc.shape[:2]
        
        # Format for preprocessor
        if image_hwc.dtype == np.uint8:
            bchw = (image_hwc.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)
        else:
            bchw = image_hwc[None].transpose(0, 3, 1, 2)
            
        if self.preprocessor.mode == "bisenet":
            mask = self.preprocessor._bisenet_mask(bchw)
        else:
            mask = _get_heuristic_mask(H, W)
            
        return mask

    def apply_mask(self, image_hwc: np.ndarray, mask: np.ndarray, bg_color: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
        """
        Apply the mask to the image.
        Args:
            image_hwc: np.ndarray, uint8, (H, W, 3)
            mask: np.ndarray, float32, (H, W, 1)
            bg_color: np.ndarray, (3,) color to fill the background
        """
        img_f = image_hwc.astype(np.float32)
        bg_f = bg_color.astype(np.float32)[None, None, :]
        result = img_f * mask + bg_f * (1.0 - mask)
        return np.clip(result, 0, 255).astype(np.uint8)
