import torch
import numpy as np
import gc
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPMatcher:
    def __init__(self, model_name='openai/clip-vit-large-patch14'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load directly to CPU first
        self.model = CLIPModel.from_pretrained(model_name).to("cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def get_top_k_segments(self, image, segments, text_query, k=5):
        if not segments: return []
        
        # 1. Clean Text
        ignore = ['remove', 'delete', 'erase', 'the', 'a', 'an']
        words = [w for w in text_query.lower().split() if w not in ignore]
        clean_text = " ".join(words) if words else text_query
        
        # 2. Crop (CPU)
        pil_image = Image.fromarray(image)
        crops = []
        valid_segments = []
        
        h_img, w_img = image.shape[:2]
        total_img_area = h_img * w_img
        
        for seg in segments:
            if 'bbox' not in seg: continue
            
            # Safe numpy cast
            bbox = np.array(seg['bbox']).astype(int)
            x1, y1, x2, y2 = bbox
            
            # Adaptive Context Padding (30%)
            w_box, h_box = x2 - x1, y2 - y1
            pad_x = int(w_box * 0.3)
            pad_y = int(h_box * 0.3)
            
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w_img, x2 + pad_x)
            crop_y2 = min(h_img, y2 + pad_y)
            
            crops.append(pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2)))
            valid_segments.append(seg)
            
        if not crops: return []

        # 3. Inference (Brief GPU usage)
        try:
            self.model.to(self.device)
            inputs = self.processor(
                text=[clean_text], images=crops, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # FIX: Use raw logits for meaningful scores. 
                # (Softmax forces sum=1, concealing bad matches)
                probs = outputs.logits_per_image.cpu().numpy().flatten()
        except Exception as e:
            print(f"CLIP Error: {e}")
            return []
        finally:
            # Move back to CPU immediately
            self.model.to("cpu") 

        # 4. Score & Sort
        final_results = []
        for i, score in enumerate(probs):
            seg = valid_segments[i]
            if 'area' in seg:
                area_ratio = seg['area'] / total_img_area
            else:
                w, h = seg['bbox'][2]-seg['bbox'][0], seg['bbox'][3]-seg['bbox'][1]
                area_ratio = (w*h) / total_img_area
            
            # Logits are roughly 15-30 range. Add small boost for area.
            weighted_score = float(score) + (area_ratio * 2.0)
            
            final_results.append({
                'mask': seg.get('mask', None),
                'bbox': seg['bbox'],
                'original_score': float(score),
                'weighted_score': weighted_score,
                'label': seg.get('label', 'object')
            })

        final_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        return final_results[:k]