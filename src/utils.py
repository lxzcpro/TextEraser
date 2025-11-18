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

def combine_masks(masks):
    """Combine multiple masks into one"""
    if not masks:
        return None
    combined = np.zeros_like(masks[0])
    for mask in masks:
        combined = np.logical_or(combined, mask).astype(np.uint8)
    return combined

def resize_image(image, max_size=1024):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image