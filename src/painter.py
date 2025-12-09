import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

class SDXLInpainter:
    def __init__(self, model_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)
        
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
    
    def inpaint(self, image, mask, prompt=""):
        pil_image = Image.fromarray(image).convert('RGB')
        

        mask = self._dilate_mask(mask, kernel_size=15)
        

        import cv2
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
        
        w, h = pil_image.size
        target_size = 1024
        scale = target_size / max(w, h)
        new_w = int(w * scale) - (int(w * scale) % 8)
        new_h = int(h * scale) - (int(h * scale) % 8)
        
        resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        resized_mask = pil_mask.resize((new_w, new_h), Image.NEAREST)
        
        if not prompt or prompt == "background":
            final_prompt = "clean background, empty space, seamless texture, high quality"

            guidance_scale = 4.5 
        else:
            final_prompt = prompt
            guidance_scale = 7.5

        neg_prompt = (
            "object, subject, person, animal, cat, dog, "
            "glass, transparent, crystal, bottle, cup, reflection, "
            "complex, 3d render, artifacts, shadow, distortion, blur, watermark"
        )

        output = self.pipe(
            prompt=final_prompt,
            negative_prompt=neg_prompt,
            image=resized_image,
            mask_image=resized_mask,
            num_inference_steps=40,
            guidance_scale=guidance_scale,
            strength=0.99,
        ).images[0]
        
        result = output.resize((w, h), Image.LANCZOS)
        
        return np.array(result)
    
    def _dilate_mask(self, mask, kernel_size=15):
        import cv2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)