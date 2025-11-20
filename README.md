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
