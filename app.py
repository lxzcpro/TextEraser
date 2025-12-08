import gradio as gr
import numpy as np
import argparse
import os
from src.pipeline import ObjectRemovalPipeline
from src.utils import visualize_mask

# --- ZeroGPU Compatibility Shim ---
# Allows code to run on local CPU/GPU without crashing on 'import spaces'
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(duration=120):
            def decorator(func):
                return func
            return decorator

# Initialize pipeline (Models use lazy-loading to save memory)
pipeline = ObjectRemovalPipeline()

def ensure_uint8(image):
    """Normalize image to uint8 (0-255)"""
    if image is None: return None
    image = np.array(image)
    if image.dtype != np.uint8:
        if image.max() <= 1.0: image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

@spaces.GPU(duration=120)
def step1_detect(image, text_query):
    """Detect objects and return candidates for user selection"""
    if image is None or not text_query:
        return [], [], "Please upload image and enter text."
    
    # 1. Detect & Rank candidates via Pipeline
    candidates, msg = pipeline.get_candidates(image, text_query)
    
    if not candidates:
        return [], [], f"Error: {msg}"
    
    masks = [c['mask'] for c in candidates]
    
    # 2. Visualize masks for Gallery
    gallery_imgs = []
    for i, mask in enumerate(masks):
        viz = visualize_mask(image, mask)
        score = candidates[i].get('weighted_score', 0)
        label = f"Option {i+1} (Score: {score:.2f})"
        gallery_imgs.append((ensure_uint8(viz), label))
        
    return masks, gallery_imgs, "Select the best match below."

def on_select(evt: gr.SelectData):
    """Capture user selection from Gallery"""
    return evt.index

@spaces.GPU(duration=120)
def step2_remove(image, masks, selected_idx, prompt, shadow_exp):
    """Inpaint the selected mask"""
    if not masks or selected_idx is None:
        return None, "Please select an object first."
    
    target_mask = masks[selected_idx]
    
    # 3. Inpaint with Shadow Fix logic
    result = pipeline.inpaint_selected(image, target_mask, prompt, shadow_expansion=shadow_exp)
    
    return ensure_uint8(result), "Success!"

# CSS for better layout and full image visibility in Gallery
css = """
.gradio-container {min-height: 0px !important}
button.gallery-item {object-fit: contain !important}
"""

with gr.Blocks(title="TextEraser", css=css, theme=gr.themes.Soft()) as demo:
    # State to hold masks between steps
    mask_state = gr.State([])
    idx_state = gr.State(0) 

    gr.Markdown("## TextEraser: Interactive Object Removal")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="numpy", height=400)
            text_query = gr.Textbox(label="What to remove?", placeholder="e.g. 'bottle', 'shadow'")
            btn_detect = gr.Button("1. Detect Objects", variant="primary")
        
        with gr.Column(scale=1):
            # Interactive Gallery (Adaptable size)
            gallery = gr.Gallery(
                label="Candidates (Select One)", 
                columns=2, 
                height=400, 
                allow_preview=True, 
                object_fit="contain" 
            )
            status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            shadow_slider = gr.Slider(0, 40, value=10, label="Shadow Fix (Expand Mask Downwards)")
            inpaint_prompt = gr.Textbox(label="Background Description", value="background")
            btn_remove = gr.Button("2. Remove Selected", variant="stop")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="Final Result", height=400)

    # Event Wiring
    btn_detect.click(
        fn=step1_detect,
        inputs=[input_image, text_query],
        outputs=[mask_state, gallery, status]
    )
    
    gallery.select(fn=on_select, inputs=None, outputs=idx_state)
    
    btn_remove.click(
        fn=step2_remove,
        inputs=[input_image, mask_state, idx_state, inpaint_prompt, shadow_slider],
        outputs=[output_image, status]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public link (Colab)")
    args = parser.parse_args()
    
    # queue() is required for ZeroGPU
    demo.queue().launch(share=args.share)