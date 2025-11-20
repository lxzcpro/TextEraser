import torch
import numpy as np
import cv2

from ultralytics import YOLO
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class YOLOSegmenter:
    def __init__(self, model_name='yolov8x-seg.pt'):
        self.model = YOLO(model_name)
    
    def segment(self, image):
        """Return list of (mask, bbox, class_id) tuples"""
        results = self.model(image)[0]
        segments = []
        
        if results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                bbox = results.boxes.xyxy[i].cpu().numpy()
                class_id = int(results.boxes.cls[i])
                segments.append({
                    'mask': (mask_resized > 0.5).astype(np.uint8),
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': self.model.names[class_id]
                })
        
        return segments

class SAM2Segmenter:
    def __init__(self, model_cfg='sam2.1_hiera_l.yaml', checkpoint=''):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the Automatic Generator
        self.mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
            "facebook/sam2.1-hiera-large",
            points_per_side=32,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            device=self.device
        )

    def segment(self, image):
        """
        Generates masks and filters out background-like huge segments.
        """
        if hasattr(self.mask_generator, 'generate'):
            masks = self.mask_generator.generate(image)
        else:
            masks = self.mask_generator.predict(image)

        segments = []
        img_h, img_w = image.shape[:2]
        total_area = img_h * img_w

        for m in masks:
            # SAM returns [x, y, w, h]
            x, y, w, h = m['bbox']
            
            # Convert to [x1, y1, x2, y2]
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Ignore masks that are too large (> 75% of image)
            if m['area'] > total_area * 0.75:
                continue
                
            # Ignore masks that are too small (< 0.5% of image)
            if m['area'] < total_area * 0.005:
                continue

            segments.append({
                'mask': m['segmentation'].astype(np.uint8),
                'bbox': np.array([x1, y1, x2, y2]),
                'score': m.get('predicted_iou', 1.0),
                'area': m['area']
            })
            
        # Sort by area (smallest to largest) to prefer specific objects over containers
        segments.sort(key=lambda s: s['area'])
        
        return segments