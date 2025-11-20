# TextEraser ✨

**Text-Guided Precise Object Removal with SAM2 + CLIP + Stable Diffusion XL**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-orange.svg)](https://gradio.app/)

> Final Project for **COMPSCI372: Intro to Applied Machine Learning** @ Duke University (Fall 2025)

TextEraser is an intelligent object removal pipeline that combines state-of-the-art computer vision models to seamlessly remove objects from images using natural language descriptions. Simply describe what you want to remove, and let AI do the rest!

---

## Features

🎯 **Natural Language Control** - Remove objects using simple text descriptions like "remove the bottle" or "delete the car"

🤖 **State-of-the-Art Models**
- **SAM2** (Segment Anything Model 2) for precise object segmentation
- **CLIP** for intelligent text-to-image matching
- **Stable Diffusion XL** for photorealistic inpainting

🔄 **Smart Multi-Part Object Handling** - Automatically merges related segments (e.g., cat + tail)

🎨 **Interactive Web Interface** - User-friendly Gradio interface with real-time progress tracking

🐛 **Debug Visualization** - See exactly what's being detected and removed

---

## Demo

<!-- Add your screenshots here -->
<!--
![Demo Example](docs/demo.gif)

**Input:** "Remove the coffee cup"
![Before](docs/before.jpg) ![After](docs/after.jpg)
-->

### Try It Yourself

```bash
python app.py
```

Then open the Gradio interface and:
1. Upload an image
2. Enter what to remove (e.g., "bottle", "person", "car")
3. Optionally specify what should fill the space (e.g., "grass", "background")
4. Click "Run Pipeline" and watch the magic happen!

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 12GB+ VRAM)
- ~10GB free disk space for model downloads

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/lxzcpro/TextEraser.git
   cd TextEraser
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

   On first run, the following models will be automatically downloaded from HuggingFace:
   - `facebook/sam2.1-hiera-large` (~2.3GB)
   - `openai/clip-vit-large-patch14` (~890MB)
   - `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (~6.9GB)

4. **Access the web interface**

   The Gradio interface will launch automatically. Look for the public URL in the console output.

---

## How It Works

TextEraser uses a **three-stage pipeline** to intelligently remove objects:

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐
│ Segmentation│ --> │ Matching │ --> │ Inpainting  │
│    (SAM2)   │     │  (CLIP)  │     │   (SDXL)    │
└─────────────┘     └──────────┘     └─────────────┘
```

### Stage 1: Segmentation
- **SAM2** generates candidate object masks across the entire image
- Filters out noise (too small) and background (too large)
- Returns 30-100 candidate segments sorted by size

### Stage 2: Matching
- **CLIP** compares each segment against your text query
- Scores are weighted by object size to prefer prominent objects
- Top 5 candidates are selected for potential merging

### Stage 3: Refinement & Inpainting
- Similar segments are merged to capture multi-part objects
- Mask is dilated and blurred for smooth edges
- **Stable Diffusion XL** fills the masked region with contextually appropriate content

For more technical details, see [CLAUDE.md](CLAUDE.md).

---

## Usage Examples

### Basic Object Removal

```python
from src.pipeline import ObjectRemovalPipeline
from PIL import Image
import numpy as np

# Initialize pipeline (loads all models)
pipeline = ObjectRemovalPipeline()

# Load image
image = np.array(Image.open("photo.jpg"))

# Remove object
result, mask, message = pipeline.process(
    image=image,
    text_query="remove the bottle",
    inpaint_prompt="table surface"
)

# Save result
Image.fromarray(result).save("result.jpg")
```

### Advanced: Custom Segmentation

```python
# Use YOLOv8 instead of SAM2
from src.segmenter import YOLOSegmenter
from src.matcher import CLIPMatcher
from src.painter import SDXLInpainter

segmenter = YOLOSegmenter()  # Class-aware segmentation
matcher = CLIPMatcher()
painter = SDXLInpainter()

# Build custom pipeline...
```

---

## Project Structure

```
TextEraser/
├── app.py                  # Gradio web interface
├── requirements.txt        # Python dependencies
├── src/
│   ├── pipeline.py         # Main orchestration
│   ├── segmenter.py        # SAM2 & YOLOv8 implementations
│   ├── matcher.py          # CLIP text-to-segment matching
│   ├── painter.py          # Stable Diffusion inpainting
│   └── utils.py            # Helper functions
├── models/                 # Model weights cache (auto-populated)
├── CLAUDE.md              # Detailed technical documentation
└── README.md              # You are here!
```

---

## Configuration & Tuning

### Speed vs Quality Trade-off

**Faster Inference (Lower Quality):**
```python
# In src/painter.py, line 110
num_inference_steps=20  # Default: 40
guidance_scale=3.5      # Default: 4.5-7.5
```

**Higher Quality (Slower):**
```python
num_inference_steps=80
guidance_scale=9.0
```

### Segmentation Sensitivity

**More Segments (Better Coverage, Slower):**
```python
# In src/segmenter.py, line 38-40
points_per_side=64        # Default: 32
pred_iou_thresh=0.70      # Default: 0.80
```

**Fewer Segments (Faster, May Miss Small Objects):**
```python
points_per_side=16
pred_iou_thresh=0.90
```

---

## Requirements

### Core Dependencies

- `torch` - PyTorch deep learning framework
- `transformers` - HuggingFace models (CLIP)
- `diffusers` - Stable Diffusion pipeline
- `ultralytics` - YOLOv8 (optional)
- `gradio` - Web interface
- `Pillow` - Image processing
- `numpy` - Numerical operations

See [requirements.txt](requirements.txt) for complete list.

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 12GB+ VRAM (RTX 3060+) |
| RAM | 16GB | 32GB |
| Storage | 15GB | 20GB |
| CPU | 4 cores | 8+ cores |

**Note:** CPU-only mode is supported but significantly slower (~10x).

---

## Troubleshooting

### Out of Memory (OOM) Errors

```python
# Add memory optimization in src/painter.py after model load
self.pipe.enable_attention_slicing()
self.pipe.enable_vae_slicing()
```

Or resize input images to smaller dimensions before processing.

### Wrong Object Detected

- Check the "Segmentation Debug" tab to see what was detected
- Try more specific queries: "red car" instead of "car"
- Adjust the area weight in `src/matcher.py:61`

### Incomplete Removal / Edge Artifacts

- Increase mask dilation: `src/pipeline.py:52`
- Increase Gaussian blur: `src/painter.py:78`

For more issues, see the [debugging guide in CLAUDE.md](CLAUDE.md#debugging--troubleshooting).

---

## Limitations

- **Speed**: Full pipeline takes 10-30 seconds per image on GPU
- **Accuracy**: CLIP may struggle with rare/abstract objects
- **Quality**: Inpainting quality depends on background complexity
- **Multi-object**: Currently removes one object type per run
- **Text Queries**: Works best with concrete nouns (e.g., "cat" vs "the furry thing")

---

## Future Improvements

- [ ] Batch processing for multiple images
- [ ] Video object removal support
- [ ] Fine-tuned CLIP for laboratory/scientific objects
- [ ] Real-time preview with lightweight models
- [ ] Multiple object removal in single pass
- [ ] Undo/redo functionality in UI
- [ ] Custom negative prompts per use case

---

## Academic Context

This project was developed as the final project for **COMPSCI372: Introduction to Applied Machine Learning** at Duke University (Fall 2025). It demonstrates:

- Integration of multiple state-of-the-art models
- Practical applications of segmentation, vision-language models, and generative AI
- End-to-end ML pipeline design and deployment
- User interface design for ML applications

---

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{texteraser2025,
  author = {Zhang, Xuting},
  title = {TextEraser: Text-Guided Object Removal with SAM2 and Stable Diffusion},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/lxzcpro/TextEraser}
}
```

---

## Acknowledgments

This project builds upon incredible work from:

- **Meta AI** - [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2)
- **OpenAI** - [CLIP](https://github.com/openai/CLIP)
- **Stability AI** - [Stable Diffusion XL](https://stability.ai/stable-diffusion)
- **HuggingFace** - [Diffusers Library](https://github.com/huggingface/diffusers)
- **Gradio** - [Gradio Interface](https://gradio.app/)

Special thanks to the COMPSCI372 teaching staff at Duke University for guidance and support.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Xuting Zhang

---

## Contact & Contributing

**Author:** Xuting Zhang

**Issues:** Please report bugs or feature requests via [GitHub Issues](https://github.com/lxzcpro/TextEraser/issues)

**Contributing:** Contributions are welcome! Please feel free to submit a Pull Request.

For detailed technical documentation and development guidelines, see [CLAUDE.md](CLAUDE.md).

---

**Made with ❤️ for COMPSCI372 @ Duke University**
