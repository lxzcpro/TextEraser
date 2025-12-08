# Attributions & Acknowledgements

## Third-Party Models & Libraries
This project integrates several open-source state-of-the-art models and libraries.

| Component | Model / Library | License | Source |
|-----------|-----------------|---------|--------|
| **Segmentation** | **SAM 2** (Segment Anything Model 2) | Apache 2.0 | [Meta AI](https://github.com/facebookresearch/segment-anything-2) |
| **Detection** | **YOLO-World** | AGPL-3.0 | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| **Matching** | **CLIP** (ViT-L/14) | MIT | [OpenAI](https://github.com/openai/CLIP) |
| **Inpainting** | **Stable Diffusion XL (Inpainting)** | CreativeML Open RAIL++-M | [Stability AI](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
| **UI Framework** | **Gradio** | Apache 2.0 | [Gradio](https://www.gradio.app/) |

## AI Assistance Declaration
Generative AI tools (ChatGPT/Gemini) were used to assist in the development of this project:
1.  **Refining Architecture**: Assisted in designing the modular directory structure.
2.  **Debugging**: Helped troubleshoot memory management logic (garbage collection and model offloading) for GPU efficiency.
All AI-generated suggestions were manually reviewed, verified, and integrated by the author.