# TextEraser

**Text-Guided Object Removal using SAM2 + CLIP + Stable Diffusion XL**

Final Project for COMPSCI372: Intro to Applied Machine Learning @ Duke University (Fall 2025)

---

## Overview

TextEraser intelligently removes objects from images using natural language descriptions. Simply type what you want to remove (e.g., "bottle", "person", "car"), and the AI pipeline handles the rest.

### Key Features

- **Natural language control** - Remove objects by describing them in plain text
- **Smart segmentation** - Uses SAM2 to find all objects in the image
- **Intelligent matching** - CLIP identifies which segments match your description
- **Seamless inpainting** - Stable Diffusion XL fills in the removed area naturally
- **Multi-part object handling** - Automatically merges related segments (e.g., cat + tail)
- **Interactive web interface** - Real-time Gradio UI with debug visualization

---

## How It Works

The pipeline has three stages:

1. **Segmentation (SAM2)** - Generates candidate object masks across the image
2. **Matching (CLIP)** - Scores each segment against your text query
3. **Inpainting (SDXL)** - Fills the masked region with contextually appropriate content

---

## Installation

### Requirements

- Python 3.8+
- CUDA GPU with 12GB+ VRAM (recommended)
- ~10GB disk space for models

### Setup

```bash
# Clone repository
git clone https://github.com/lxzcpro/TextEraser.git
cd TextEraser

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

On first run, models will auto-download from HuggingFace (~10GB total).

---

## Usage

### Web Interface

1. Launch the app: `python app.py`
2. Upload an image
3. Enter what to remove (e.g., "bottle", "car", "person")
4. Optionally specify background fill (default: "background")
5. Click "Run Pipeline"
6. Check the debug tab to see what was detected

### Python API

```python
from src.pipeline import ObjectRemovalPipeline
from PIL import Image
import numpy as np

# Initialize pipeline
pipeline = ObjectRemovalPipeline()

# Load and process image
image = np.array(Image.open("photo.jpg"))
result, mask, message = pipeline.process(
    image=image,
    text_query="bottle",
    inpaint_prompt="table surface"
)

# Save result
Image.fromarray(result).save("result.jpg")
```

---

## Project Structure

```
TextEraser/
├── app.py              # Gradio web interface
├── src/
│   ├── pipeline.py     # Main orchestration
│   ├── segmenter.py    # SAM2 segmentation
│   ├── matcher.py      # CLIP matching
│   ├── painter.py      # SDXL inpainting
│   └── utils.py        # Helper functions
└── requirements.txt    # Dependencies
```

---

## Configuration

### Speed vs Quality

Edit `src/painter.py:110` to adjust inference parameters:

```python
# Faster (lower quality)
num_inference_steps=20    # default: 40
guidance_scale=3.5        # default: 4.5-7.5

# Higher quality (slower)
num_inference_steps=80
guidance_scale=9.0
```

### Segmentation Sensitivity

Edit `src/segmenter.py:38-40`:

```python
# More segments (better coverage)
points_per_side=64        # default: 32
pred_iou_thresh=0.70      # default: 0.80

# Fewer segments (faster)
points_per_side=16
pred_iou_thresh=0.90
```

---

## Troubleshooting

**Out of Memory Error:**
- Reduce image size or add `enable_attention_slicing()` in `src/painter.py`

**Wrong Object Selected:**
- Check debug tab to see what was detected
- Use more specific descriptions: "red car" instead of "car"

**Incomplete Removal:**
- Increase mask dilation in `src/pipeline.py:52`
- Increase Gaussian blur in `src/painter.py:78`

---

## Limitations

- Takes 10-30 seconds per image on GPU
- Works best with concrete object names
- Inpainting quality depends on background complexity
- Currently handles one object type per run

---

## Technologies Used

- **SAM2** (Meta AI) - Object segmentation
- **CLIP** (OpenAI) - Text-image matching
- **Stable Diffusion XL** (Stability AI) - Image inpainting
- **Gradio** - Web interface
- **PyTorch + HuggingFace** - Model frameworks

---

## License

MIT License - Copyright (c) 2025 Xuting Zhang

---

## Author

**Xuting Zhang**
Duke University - COMPSCI372 Final Project
