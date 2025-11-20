import numpy as np
import cv2
from .segmenter import SAM2Segmenter
from .matcher import CLIPMatcher
from .painter import SDXLInpainter
from .utils import visualize_mask

class ObjectRemovalPipeline:
    def __init__(self):
        print("Initializing models...")
        self.segmenter = SAM2Segmenter()
        self.matcher = CLIPMatcher()
        self.inpainter = SDXLInpainter()
        print("Pipeline ready.")
    
    def process(self, image, text_query, inpaint_prompt=""):
        """
        Main processing function for object removal.
        """
        # 1. Segment
        segments = self.segmenter.segment(image)
        if not segments:
            return image, None, "No segments found"
        
        # 2. Match with Top-K Strategy
        # We get top 5 candidates to handle "Part-Whole" ambiguity (e.g. tire vs car)
        candidates = self.matcher.get_top_k_segments(image, segments, text_query, k=5)
        if not candidates:
            return image, None, "No match found"
            
        # 3. Merge Masks (The "Cat Tail" Fix)
        best_candidate = candidates[0]
        final_mask = best_candidate['mask'].copy()
        
        print(f"Top Match Score: {best_candidate['weighted_score']:.3f}")

        # Merge other candidates if they are close in score or physically overlap
        for i in range(1, len(candidates)):
            cand = candidates[i]
            score_ratio = cand['weighted_score'] / best_candidate['weighted_score']
            
            # Check intersection
            intersection = np.logical_and(final_mask, cand['mask']).sum()
            
            # Rule: Merge if score is similar (>85%) OR if they overlap pixels
            if score_ratio > 0.85 or intersection > 0:
                print(f"Merging Rank {i+1} (Score ratio: {score_ratio:.2f}, Overlap: {intersection > 0})")
                final_mask = np.logical_or(final_mask, cand['mask'])

        # 4. Dilate Final Mask
        # Expands mask slightly to cover edges/seams
        kernel = np.ones((15, 15), np.uint8)
        final_mask = cv2.dilate(final_mask.astype(np.uint8), kernel, iterations=1)

        # 5. Inpaint
        result = self.inpainter.inpaint(image, final_mask, prompt=inpaint_prompt)
        
        return result, final_mask, "Success"