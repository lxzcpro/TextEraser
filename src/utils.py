import numpy as np
import cv2

def visualize_mask(image, mask, alpha=0.5):
    """
    Overlay mask on image for visualization.
    Automatically resizes mask to match image dimensions.
    """
    image = np.array(image)
    mask = np.array(mask)
    
    h, w = image.shape[:2]
    mh, mw = mask.shape[:2]

    if (h, w) != (mh, mw):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]
    
    return cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
