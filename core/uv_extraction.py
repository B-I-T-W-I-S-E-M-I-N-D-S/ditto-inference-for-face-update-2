import cv2
import numpy as np
from scipy.interpolate import griddata

class UVExtractor:
    """
    Extracts a 2D UV texture map from a 2D rendered face image by predicting its 3D landmarks
    and using cylindrical unwrapping to create a canonical UV layout.
    """
    def __init__(self, landmark478_model, uv_size=(512, 512)):
        """
        Args:
            landmark478_model: Instance of core.aux_models.mediapipe_landmark478.Landmark478
            uv_size: Tuple (W, H) for the output UV map.
        """
        self.landmark478 = landmark478_model
        self.uv_size = uv_size

    def extract_uv(self, image_hwc: np.ndarray) -> np.ndarray:
        """
        Extracts UV map from the 2D face image.
        Args:
            image_hwc: np.ndarray, uint8, (H, W, 3)
        Returns:
            uv_map: np.ndarray, uint8, (H_uv, W_uv, 3)
        """
        # 1. Get landmarks
        # The landmark478 model expects RGB uint8 image.
        # It returns mesh of shape (1, 478, 3) normalized to [0, 1] relative to the image size, 
        # or it handles the raw pixel coordinates depending on the internal crop.
        # From mediapipe_landmark478.py, Landmark478.__call__ returns lmk of shape (1, 478, 3)
        # where the values are already normalized by (image_width, image_height, image_width).
        
        lmk = self.landmark478(image_hwc)
        if lmk is None:
            # If face not detected, return black image
            return np.zeros((self.uv_size[1], self.uv_size[0], 3), dtype=np.uint8)
            
        landmarks = lmk[0]  # (478, 3)
        h, w = image_hwc.shape[:2]
        
        # Scale to pixel coordinates
        pts_3d = landmarks.copy()
        pts_3d[:, 0] *= w
        pts_3d[:, 1] *= h
        pts_3d[:, 2] *= w  # Z is usually scaled by W
        
        # 2. Cylindrical unwrap
        # theta = arctan2(X - CX, Z - CZ)
        cx, cy, cz = pts_3d.mean(axis=0)
        theta = np.arctan2(pts_3d[:, 0] - cx, pts_3d[:, 2] - cz)
        y = pts_3d[:, 1]
        
        # Normalize theta to [0, uv_w - 1]
        theta_min, theta_max = theta.min(), theta.max()
        if theta_max == theta_min:
            u = np.zeros_like(theta)
        else:
            u = (theta - theta_min) / (theta_max - theta_min) * (self.uv_size[0] - 1)
            
        # Normalize y to [0, uv_h - 1]
        y_min, y_max = y.min(), y.max()
        if y_max == y_min:
            v = np.zeros_like(y)
        else:
            v = (y - y_min) / (y_max - y_min) * (self.uv_size[1] - 1)
            
        uv_pts = np.stack([u, v], axis=1)  # (478, 2)
        img_pts = pts_3d[:, :2]            # (478, 2)
        
        # 3. Interpolate mapping grid
        grid_x, grid_y = np.meshgrid(np.arange(self.uv_size[0]), np.arange(self.uv_size[1]))
        
        # map_x[v, u] = img_x
        # map_y[v, u] = img_y
        map_x = griddata(uv_pts, img_pts[:, 0], (grid_x, grid_y), method='linear')
        map_y = griddata(uv_pts, img_pts[:, 1], (grid_x, grid_y), method='linear')
        
        # Nearest neighbor fallback for edges
        map_x_nearest = griddata(uv_pts, img_pts[:, 0], (grid_x, grid_y), method='nearest')
        map_y_nearest = griddata(uv_pts, img_pts[:, 1], (grid_x, grid_y), method='nearest')
        
        nan_mask = np.isnan(map_x)
        map_x[nan_mask] = map_x_nearest[nan_mask]
        map_y[nan_mask] = map_y_nearest[nan_mask]
        
        # Still nan? Set to 0
        map_x[np.isnan(map_x)] = 0
        map_y[np.isnan(map_y)] = 0
        
        # 4. Remap pixels
        uv_map = cv2.remap(
            image_hwc, 
            map_x.astype(np.float32), 
            map_y.astype(np.float32), 
            cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0)
        )
        
        return uv_map
