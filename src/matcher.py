import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPMatcher:
    def __init__(self, model_name='openai/clip-vit-large-patch14'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def get_top_k_segments(self, image, segments, text_query, k=5):
        """
        Returns top K segments based on CLIP score + Area Weight.
        """
        if not segments: return []
        
        # 1. Clean Text
        ignore = ['remove', 'delete', 'erase', 'the', 'a', 'an']
        words = [w for w in text_query.lower().split() if w not in ignore]
        clean_text = " ".join(words) if words else text_query
        
        pil_image = Image.fromarray(image)
        crops = []
        valid_segments = []
        
        # Prepare crops
        h, w = image.shape[:2]
        total_img_area = h * w
        
        for seg in segments:
            x1, y1, x2, y2 = seg['bbox'].astype(int)
            # Pad slightly
            pad = 10
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)
            
            crops.append(pil_image.crop((x1, y1, x2, y2)))
            valid_segments.append(seg)
            
        if not crops: return []

        # 2. Inference
        inputs = self.processor(
            text=[clean_text], images=crops, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Standardize scores
            probs = outputs.logits_per_image.softmax(dim=0).cpu().numpy().flatten()

        # 3. Re-Scoring with Area Weight
        final_results = []
        for i, score in enumerate(probs):
            seg = valid_segments[i]
            area_ratio = seg['area'] / total_img_area
            
            # HEURISTIC: Boost score for larger objects.
            # If searching for general terms (bus, car, cat), bigger is usually better.
            # We add 20% of the area_ratio to the score.
            weighted_score = score + (area_ratio * 0.2) 
            
            final_results.append({
                'mask': seg['mask'],
                'bbox': seg['bbox'],
                'original_score': float(score),
                'weighted_score': float(weighted_score)
            })

        # 4. Sort and take Top K
        final_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        return final_results[:k]