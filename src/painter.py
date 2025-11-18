import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

class SDInpainter:
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use float16 for GPU to save VRAM and speed up inference
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
    
    def inpaint(self, image, mask, prompt="background"):
        pil_image = Image.fromarray(image).convert('RGB')
        
        # Dilate mask to ensure the object edge is covered
        mask = self._dilate_mask(mask)
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
        
        # 1. Keep aspect ratio, resize ensuring dimensions are multiples of 8
        w, h = pil_image.size
        factor = 512 / max(w, h) # Scale based on the longest side
        new_w = int(w * factor) - (int(w * factor) % 8)
        new_h = int(h * factor) - (int(h * factor) % 8)
        
        resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        resized_mask = pil_mask.resize((new_w, new_h), Image.NEAREST)
        
        # 2. Inpaint
        output = self.pipe(
            prompt=prompt,
            negative_prompt="artifacts, low quality, distortion, object", # Add negative prompt for better quality
            image=resized_image,
            mask_image=resized_mask,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
        
        # 3. Resize back to original resolution
        result = output.resize((w, h), Image.LANCZOS)
        
        return np.array(result)
    
    def _dilate_mask(self, mask, kernel_size=9): 
        # Increased kernel size slightly for better blending
        import cv2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)