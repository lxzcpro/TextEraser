import torch
import numpy as np
import gc
from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor

class YOLOWorldDetector:
    def __init__(self, model_name='yolov8s-worldv2.pt'):

        self.model = YOLO(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def detect(self, image, text_query):
        clean_text = text_query.replace("remove", "").replace("delete", "").strip()
        if not clean_text: clean_text = "object"
        
        boxes = []
        try:

            self.model.to('cpu')
            self.model.set_classes([clean_text])
            
            if self.device == 'cuda':
                self.model.to('cuda')
                
            results = self.model.predict(image, conf=0.05, iou=0.5, verbose=False)[0]
            
            if results.boxes:
                for box in results.boxes.data:
                    x1, y1, x2, y2 = box[:4].cpu().numpy()
                    conf = float(box[4])
                    boxes.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'score': conf,
                        'label': clean_text
                    })
        except Exception as e:
            print(f"YOLO Error: {e}")
        finally:

            self.model.to('cpu')
            
        boxes.sort(key=lambda x: x['score'], reverse=True)
        return boxes

class SAM2Predictor:
    def __init__(self, checkpoint="facebook/sam2.1-hiera-large"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(checkpoint)
        except:
            self.predictor = SAM2ImagePredictor.from_pretrained(checkpoint, device='cpu')

    def set_image(self, image):
        self.predictor.model.to(self.device)
        self.predictor.set_image(image)

    def predict_from_box(self, bbox):
        box_input = np.array(bbox)[None, :]

        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_input,
            multimask_output=True 
        )
        sorted_results = sorted(zip(masks, scores), key=lambda x: x[1], reverse=True)
        return [(m.astype(np.uint8), s) for m, s in sorted_results]
        
    def clear_memory(self):

        self.predictor.reset_predictor()
        self.predictor.model.to('cpu')
        del self.predictor
        torch.cuda.empty_cache()
        gc.collect()