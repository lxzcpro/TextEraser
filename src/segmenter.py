import numpy as np
from ultralytics import YOLO
import cv2

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