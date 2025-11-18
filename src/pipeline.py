import numpy as np
from .segmenter import YOLOSegmenter
from .matcher import CLIPMatcher
from .painter import SDInpainter
from .utils import resize_image

class ObjectRemovalPipeline:
    def __init__(self):
        print("Initializing models...")
        self.segmenter = YOLOSegmenter()
        self.matcher = CLIPMatcher()
        self.inpainter = SDInpainter()
        print("Models loaded successfully!")
    
    def process(self, image, text_query, inpaint_prompt="background"):
        """
        Main pipeline for object removal
        Args:
            image: numpy array (H, W, 3)
            text_query: str, e.g., "remove the bottle"
            inpaint_prompt: str, prompt for inpainting
        """
        # Resize for processing
        original_shape = image.shape[:2]
        image = resize_image(image, max_size=1024)
        
        # Step 1: Segment objects
        segments = self.segmenter.segment(image)
        if not segments:
            return image, None, "No objects detected"
        
        # Step 2: Match text query to segment
        matched_segment = self.matcher.match_segments(image, segments, text_query)
        if matched_segment is None:
            return image, None, "No matching object found"
        
        # Step 3: Inpaint to remove object
        result = self.inpainter.inpaint(image, matched_segment['mask'], inpaint_prompt)
        
        # Resize back if needed
        if result.shape[:2] != original_shape:
            import cv2
            result = cv2.resize(result, (original_shape[1], original_shape[0]))
        
        return result, matched_segment['mask'], f"Removed: {matched_segment['class_name']}"