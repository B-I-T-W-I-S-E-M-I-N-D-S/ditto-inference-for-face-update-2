import cv2
import numpy as np

class SkinInpainter:
    """
    Fills removed areas (hair, background) with plausible skin texture and extends the skin canvas.
    """
    def __init__(self, inpaint_radius=5, extend_radius=30):
        self.inpaint_radius = inpaint_radius
        self.extend_radius = extend_radius

    def inpaint(self, image_hwc: np.ndarray, mask: np.ndarray, extend=False) -> np.ndarray:
        """
        Args:
            image_hwc: np.ndarray, uint8, (H, W, 3)
            mask: np.ndarray, float32, (H, W, 1), 1 is face, 0 is background
            extend: bool, if True, expands the skin region.
        Returns:
            inpainted: np.ndarray, uint8, (H, W, 3)
        """
        mask_uint8 = (mask[:, :, 0] > 0.5).astype(np.uint8) * 255
        
        if extend:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.extend_radius, self.extend_radius))
            extended_mask = cv2.dilate(mask_uint8, kernel)
            inpaint_target = cv2.bitwise_xor(extended_mask, mask_uint8)
            final_mask = extended_mask
        else:
            inpaint_target = cv2.bitwise_not(mask_uint8)
            final_mask = mask_uint8
            
        # Extract median skin color from the valid face region
        skin_pixels = image_hwc[mask_uint8 == 255]
        if len(skin_pixels) > 0:
            skin_color = np.median(skin_pixels, axis=0).astype(np.uint8)
        else:
            skin_color = np.array([128, 128, 128], dtype=np.uint8)
            
        # Pre-fill the target region with median skin color to assist Telea inpainting
        pre_filled = image_hwc.copy()
        pre_filled[inpaint_target == 255] = skin_color
        
        # Telea inpainting blends the edges
        inpainted = cv2.inpaint(pre_filled, inpaint_target, self.inpaint_radius, cv2.INPAINT_TELEA)
        
        # Output result with neutral background (black) for regions outside the final extended mask
        result = image_hwc.copy()
        result[final_mask == 255] = inpainted[final_mask == 255]
        result[final_mask == 0] = 0
        
        return result
