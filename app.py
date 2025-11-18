import gradio as gr
import numpy as np
import torch
from src.pipeline import ObjectRemovalPipeline
from src.utils import visualize_mask

# Initialize pipeline globally to load models only once
print("Loading pipeline...")
pipeline = ObjectRemovalPipeline()

def ensure_uint8(image):
    """
    Ensures the image is in valid uint8 format (0-255) for Gradio display.
    """
    if image is None:
        return None
        
    image = np.array(image)
    
    # 1. Handle NaN/Inf (Exploding gradients often cause this)
    if not np.isfinite(image).all():
        print("Warning: Image contains NaN or Inf. Replacing with black.")
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)

    # 2. Normalize Float (0.0-1.0) to Int (0-255)
    if image.dtype != np.uint8:
        # If image is in 0-1 range (common in torch/diffusers)
        if image.max() <= 1.0:
            image = (image * 255.0)
        
        # Clip to safe range and cast
        image = np.clip(image, 0, 255).astype(np.uint8)
        
    return image

def remove_object(image, text_query, inpaint_prompt, progress=gr.Progress()):
    """
    Gradio wrapper with progress tracking and error handling.
    """
    if image is None:
        return None, None, "Error: Please upload an image first."
    
    if not text_query:
        return image, None, "Error: Please specify what to remove."

    try:
        # 1. Segmentation Phase
        progress(0.2, desc="Segmenting & Matching Object...")
        
        # Note: We call the pipeline. 
        # Ideally, you would break the pipeline.process method apart to update progress 
        # between segmentation and inpainting, but this works for now.
        result, mask, message = pipeline.process(
            image, 
            text_query, 
            inpaint_prompt if inpaint_prompt else "background"
        )
        
        # 2. Visualization Phase
        progress(0.9, desc="Post-processing...")
        mask_viz = None
        if mask is not None:
            mask_viz = visualize_mask(image, mask)
        else:
            # If no mask found, return original image as preview
            mask_viz = image 

        mask_viz = ensure_uint8(mask_viz)
        result = ensure_uint8(result)

        return result, mask_viz, message

    except torch.cuda.OutOfMemoryError:
        return None, None, "Error: GPU Out of Memory. Try a smaller image."
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Define Custom CSS for a cleaner look (Optional)
css = """
footer {visibility: hidden}
.gradio-container {min-height: 0px !important}
"""

with gr.Blocks(title="Object Removal", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Text-Guided Object Removal Pipeline")
    gr.Markdown("Identify objects via CLIP and remove them using Stable Diffusion.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="numpy", height=400)
            text_query = gr.Textbox(
                label="Target Object", 
                placeholder="e.g., 'bottle', 'cell', 'petri dish'",
                info="What should be removed?"
            )
            inpaint_prompt = gr.Textbox(
                label="Inpaint Prompt (Context)", 
                placeholder="background",
                value="background",
                info="What should fill the empty space?"
            )
            submit_btn = gr.Button("Run Pipeline", variant="primary")
        
        with gr.Column(scale=1):
            # Result tabs to switch between final result and debug mask
            with gr.Tabs():
                with gr.TabItem("Final Result"):
                    output_image = gr.Image(label="Inpainted Result", height=400)
                with gr.TabItem("Segmentation Debug"):
                    mask_preview = gr.Image(label="Detected Mask Overlay", height=400)
            
            status_text = gr.Textbox(label="Pipeline Logs", interactive=False)
    
    # Examples allow users to test without uploading
    # Ensure these files actually exist in your folder, or comment this out
    # gr.Examples(
    #     examples=[["examples/lab_bench.jpg", "remove the pipette", "table surface"]],
    #     inputs=[input_image, text_query, inpaint_prompt],
    # )
    
    submit_btn.click(
        fn=remove_object,
        inputs=[input_image, text_query, inpaint_prompt],
        outputs=[output_image, mask_preview, status_text]
    )

if __name__ == "__main__":
    # queue() is essential for handling GPU workloads and preventing timeouts
    demo.queue().launch(share=True)