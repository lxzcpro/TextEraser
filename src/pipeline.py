import numpy as np
import cv2
import torch
import gc
# Note: We import classes but DO NOT instantiate them globally
from .segmenter import YOLOWorldDetector, SAM2Predictor
from .matcher import CLIPMatcher
from .painter import SDXLInpainter

class ObjectRemovalPipeline:
    def __init__(self):
        print("Initializing Pipeline in LOW MEMORY mode...")
        # No models loaded at startup!
        pass
    
    def _clear_ram(self):
        """Helper to force clear RAM & VRAM"""
        gc.collect()
        torch.cuda.empty_cache()

    def get_candidates(self, image, text_query):
        """
        Step 1: Detect & Segment & Rank
        Strategy: Load one model at a time, use it, then delete it.
        """
        candidates = []
        box_candidates = []
        
        # --- PHASE 1: YOLO (Detect) ---
        print("Loading YOLO...")
        detector = YOLOWorldDetector()
        try:
            box_candidates = detector.detect(image, text_query)
        finally:
            del detector # Delete model immediately
            self._clear_ram()
            
        if not box_candidates:
            return [], "No objects detected."

        # --- PHASE 2: SAM2 (Segment) ---
        print("Loading SAM2...")
        segmenter = SAM2Predictor()
        segments_to_score = []
        try:
            segmenter.set_image(image)
            # Process top 3 boxes -> up to 9 masks
            for cand in box_candidates[:3]: 
                bbox = cand['bbox']
                mask_variations = segmenter.predict_from_box(bbox)
                for i, (mask, sam_score) in enumerate(mask_variations):
                    segments_to_score.append({
                        'mask': mask,
                        'bbox': bbox,
                        'area': mask.sum(),
                        'label': f"{cand['label']} (Var {i+1})"
                    })
        finally:
            # Critical cleanup for SAM2
            if hasattr(segmenter, 'clear_memory'):
                segmenter.clear_memory()
            del segmenter
            self._clear_ram()

        # --- PHASE 3: CLIP (Rank) ---
        print("Loading CLIP...")
        matcher = CLIPMatcher()
        ranked_candidates = []
        try:
            ranked_candidates = matcher.get_top_k_segments(
                image, 
                segments_to_score, 
                text_query, 
                k=len(segments_to_score)
            )
        finally:
            del matcher
            self._clear_ram()
            
        return ranked_candidates, f"Found {len(ranked_candidates)} options."

    def inpaint_selected(self, image, selected_mask, inpaint_prompt="", shadow_expansion=0):
        """
        Step 2: Inpaint
        """
        # Shadow / Edge Logic (CPU ops)
        if shadow_expansion > 0:
            kernel_h = int(shadow_expansion * 1.5)
            kernel_w = int(shadow_expansion * 0.5)
            kernel = np.ones((kernel_h, kernel_w), np.uint8)
            selected_mask = cv2.dilate(selected_mask.astype(np.uint8), kernel, iterations=1)

        kernel = np.ones((10, 10), np.uint8)
        final_mask = cv2.dilate(selected_mask.astype(np.uint8), kernel, iterations=1)
        
        result = None
        
        # --- PHASE 4: SDXL (Inpaint) ---
        print("Loading SDXL...")
        inpainter = SDXLInpainter()
        try:
            result = inpainter.inpaint(image, final_mask, prompt=inpaint_prompt)
        finally:
            del inpainter
            self._clear_ram()
            
        return result