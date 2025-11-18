import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPMatcher:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def match_segments(self, image, segments, text_query):
        if not segments:
            return None
        
        ignore_words = ['remove', 'delete', 'erase', 'the', 'a', 'an']
        query_words = text_query.lower().split()
        clean_query = " ".join([w for w in query_words if w not in ignore_words])
        
        # If query becomes empty (e.g. user just typed "remove"), fallback to original
        target_text = clean_query if clean_query else text_query
        
        print(f"Debug: CLIP searching for object: '{target_text}'") 

        pil_image = Image.fromarray(image)
        best_score = -float('inf')
        best_segment = None
        
        crops = []
        valid_segments = []

        for seg in segments:
            x1, y1, x2, y2 = seg['bbox'].astype(int)
            # Check bounds to prevent crash
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 < 5 or y2 - y1 < 5: continue # Skip tiny/invalid boxes
            
            crops.append(pil_image.crop((x1, y1, x2, y2)))
            valid_segments.append(seg)
            
        if not crops: return None

        # Batch inference
        inputs = self.processor(
            text=[target_text], 
            images=crops, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits_per_image: [num_crops, 1]
            probs = outputs.logits_per_image.softmax(dim=0) 
            
            # Get the index of the highest match
            best_idx = probs.argmax().item()
            best_score = probs[best_idx].item()
            best_segment = valid_segments[best_idx]

        return best_segment