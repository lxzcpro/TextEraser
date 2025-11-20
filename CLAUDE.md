# CLAUDE.md - TextEraser Project Guide

## Project Overview

**TextEraser** is a text-guided object removal system that combines state-of-the-art computer vision models to intelligently remove objects from images based on natural language descriptions.

- **Project Type**: Machine Learning Application (Academic Project)
- **Purpose**: Final Project for COMPSCI372: Intro to Applied Machine Learning @ Duke (25 Fall)
- **License**: MIT License (Copyright 2025 Xuting Zhang)
- **Tech Stack**: Python, PyTorch, Gradio
- **Primary Models**: SAM2 (Segmentation), CLIP (Matching), Stable Diffusion XL (Inpainting)

## Repository Structure

```
TextEraser/
├── app.py                  # Gradio web interface entry point
├── requirements.txt        # Python dependencies
├── src/                    # Core pipeline modules
│   ├── __init__.py
│   ├── pipeline.py         # Main orchestration pipeline
│   ├── segmenter.py        # YOLOv8 & SAM2 segmentation
│   ├── matcher.py          # CLIP-based text-to-segment matching
│   ├── painter.py          # Stable Diffusion inpainting
│   └── utils.py            # Visualization & helper functions
├── models/                 # Directory for model weights (gitignored)
│   └── README.md
├── .gitignore              # Standard Python gitignore + model exclusions
├── LICENSE                 # MIT License
└── README.md               # Project description
```

## Architecture & Pipeline Flow

### Three-Stage Pipeline

The object removal process follows a sequential three-stage architecture:

1. **Segmentation** (`src/segmenter.py`)
   - Generates candidate object masks using SAM2
   - Filters segments by size (0.5% - 75% of image area)
   - Sorts by area (smallest first) to prefer specific objects over containers

2. **Matching** (`src/matcher.py`)
   - Uses CLIP to score segments against text query
   - Implements Top-K strategy (default k=5) for multi-part objects
   - Applies area-based re-scoring: `weighted_score = clip_score + (area_ratio * 0.2)`

3. **Inpainting** (`src/painter.py`)
   - Merges top candidates if similar scores (>85%) or overlapping
   - Dilates mask (15x15 kernel) + Gaussian blur for smooth edges
   - Fills region using Stable Diffusion XL

### Pipeline Orchestration (`src/pipeline.py:16-58`)

```python
def process(image, text_query, inpaint_prompt):
    segments = segmenter.segment(image)           # Stage 1
    candidates = matcher.get_top_k_segments(...)  # Stage 2
    final_mask = merge_and_dilate(candidates)     # Refinement
    result = inpainter.inpaint(image, mask, ...)  # Stage 3
```

## Key Components

### 1. Segmenter (`src/segmenter.py`)

**SAM2Segmenter** (Currently Active)
- Model: `facebook/sam2.1-hiera-large`
- Configuration:
  - `points_per_side=32`
  - `pred_iou_thresh=0.80`
  - `stability_score_thresh=0.92`
- Filters: Removes segments >75% or <0.5% of image area
- Location: `src/segmenter.py:32-84`

**YOLOSegmenter** (Legacy, not currently used)
- Model: `yolov8x-seg.pt`
- Provides class-aware segmentation
- Location: `src/segmenter.py:8-30`

### 2. Matcher (`src/matcher.py`)

**CLIPMatcher**
- Model: `openai/clip-vit-large-patch14`
- Features:
  - Query cleaning (removes 'remove', 'delete', 'the', etc.)
  - 10px bbox padding for better context
  - Area-weighted scoring to prefer larger objects
- Top-K Strategy: Returns 5 best candidates for merging
- Location: `src/matcher.py:6-72`

### 3. Painter (`src/painter.py`)

**SDXLInpainter** (Currently Active)
- Model: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- Resolution: 1024px (max dimension)
- Inference: 40 steps, dynamic guidance (4.5 for background, 7.5 for custom)
- Mask Processing:
  - 15x15 dilation kernel
  - Gaussian blur (5x5) for smooth transitions
- Prompting Strategy:
  - Default: "clean background, empty space, seamless texture, high quality"
  - Negative: Extensive list to prevent object hallucination
- Location: `src/painter.py:56-122`

**SDInpainter** (Legacy SD 1.5, not currently used)
- Model: `runwayml/stable-diffusion-inpainting`
- Resolution: 512px
- Location: `src/painter.py:6-53`

### 4. Utilities (`src/utils.py`)

- `visualize_mask()`: Red overlay for debugging (50% alpha)
- `combine_masks()`: Logical OR merge
- `resize_image()`: Aspect-preserving resize

### 5. Web Interface (`app.py`)

**Gradio Application**
- Framework: Gradio with `gr.Blocks`
- Features:
  - Progress tracking during inference
  - Tabbed output (Final Result / Debug Mask)
  - Error handling (OOM, NaN/Inf detection)
  - `ensure_uint8()` helper for Gradio compatibility
- Launch: `demo.queue().launch(share=True)`
- Location: `app.py:1-129`

## Development Workflow

### Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Models are auto-downloaded from HuggingFace on first run:
# - facebook/sam2.1-hiera-large (~2.3GB)
# - openai/clip-vit-large-patch14 (~890MB)
# - diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (~6.9GB)
```

### Running the Application

```bash
python app.py
# Loads all models on startup (~10-30s depending on hardware)
# Launches Gradio interface with public share link
```

### Testing Workflow

1. Upload image via Gradio interface
2. Enter text query (e.g., "bottle", "petri dish")
3. Optionally customize inpaint prompt (default: "background")
4. Check "Segmentation Debug" tab to verify mask quality
5. Iterate on query/prompt if needed

## Code Conventions

### General Practices

1. **Module Organization**: One class per file in `src/`
2. **Imports**: Standard library → Third-party → Local (PEP 8)
3. **Type Hints**: Minimal usage; relies on docstrings
4. **Error Handling**: Try-except in `app.py`, assertions in pipeline
5. **Logging**: Print statements (no formal logger)

### Model Management

- **Auto-download**: All models use HuggingFace `from_pretrained()`
- **Caching**: Default HuggingFace cache (`~/.cache/huggingface/`)
- **Device**: Auto-detect CUDA via `torch.cuda.is_available()`
- **Precision**: FP16 on GPU, FP32 on CPU
- **Memory**: CPU offloading enabled for Stable Diffusion

### Key Heuristics & Magic Numbers

| Parameter | Value | Location | Purpose |
|-----------|-------|----------|---------|
| Top-K candidates | 5 | `matcher.py:27` | Multi-part object handling |
| Area weight boost | 0.2 | `matcher.py:61` | Prefer larger objects |
| Merge threshold | 0.85 | `pipeline.py:46` | Score similarity for merging |
| Mask dilation kernel | 15x15 | `painter.py:74,119` | Edge coverage |
| Gaussian blur | 5x5 | `painter.py:78` | Smooth transitions |
| Segment area filter | 0.5%-75% | `segmenter.py:67,71` | Remove noise/background |

### Naming Conventions

- **Variables**: `snake_case` (e.g., `final_mask`, `text_query`)
- **Classes**: `PascalCase` (e.g., `ObjectRemovalPipeline`)
- **Functions**: `snake_case` (e.g., `ensure_uint8()`)
- **Private methods**: Leading underscore (e.g., `_dilate_mask()`)

## Dependencies

### Core ML Libraries

```
torch, torchvision, torchaudio   # PyTorch ecosystem
transformers                      # HuggingFace (CLIP)
diffusers[torch]                  # Stable Diffusion
ultralytics                       # YOLOv8 (legacy)
```

### Vision & Processing

```
Pillow                           # Image I/O
numpy                            # Array operations
opencv (cv2)                     # Mask operations (imported via ultralytics)
```

### Web Interface

```
gradio                           # Web UI
```

### Utilities

```
PyYAML, filelock, sniffio        # Config & async support
```

### Training/Experiment Tools (Not Used in Production)

```
pytorch-lightning, torchmetrics  # Training framework
wandb, tensorboard               # Experiment tracking
pandas, scikit-learn             # Data analysis
ipykernel                        # Jupyter support
```

## Common Tasks & Commands

### Switching Segmentation Model

**From SAM2 to YOLOv8:**
```python
# In src/pipeline.py:11
# Replace:
from .segmenter import SAM2Segmenter
self.segmenter = SAM2Segmenter()

# With:
from .segmenter import YOLOSegmenter
self.segmenter = YOLOSegmenter()
```

### Switching Inpainting Model

**From SDXL to SD 1.5:**
```python
# In src/pipeline.py:13
# Replace:
from .painter import SDXLInpainter
self.inpainter = SDXLInpainter()

# With:
from .painter import SDInpainter
self.inpainter = SDInpainter()
```

### Adjusting Inference Speed vs Quality

**Faster (Lower Quality):**
```python
# In src/painter.py:110
num_inference_steps=20  # Down from 40
guidance_scale=3.5      # Down from 4.5/7.5
```

**Higher Quality (Slower):**
```python
num_inference_steps=80
guidance_scale=9.0
```

### Memory Optimization

**If encountering OOM errors:**
```python
# In src/painter.py, add after model load:
self.pipe.enable_attention_slicing()      # Reduces memory
self.pipe.enable_vae_slicing()            # Further reduction
```

## Debugging & Troubleshooting

### Common Issues

**1. NaN/Inf in Output Images**
- **Cause**: Exploding gradients in diffusion model
- **Location**: `app.py:20-23`
- **Solution**: `ensure_uint8()` replaces NaN with 0, Inf with 255

**2. No Segments Found**
- **Cause**: Image too simple or SAM2 thresholds too strict
- **Solution**: Lower `pred_iou_thresh` or `stability_score_thresh` in `segmenter.py:39-40`

**3. Wrong Object Selected**
- **Cause**: CLIP misinterpreted query or area weight favored wrong segment
- **Debug**: Check "Segmentation Debug" tab in Gradio
- **Solution**: Refine text query or adjust area weight in `matcher.py:61`

**4. Incomplete Object Removal**
- **Cause**: Mask doesn't cover entire object
- **Solution**: Increase dilation kernel size in `pipeline.py:52` or `painter.py:74`

**5. Edge Artifacts**
- **Cause**: Mask boundaries too sharp
- **Solution**: Increase Gaussian blur kernel in `painter.py:78`

### Logging & Inspection Points

```python
# Pipeline prints key decisions:
print(f"Top Match Score: {best_candidate['weighted_score']:.3f}")  # pipeline.py:35
print(f"Merging Rank {i+1}...")                                     # pipeline.py:47

# Enable debug visualization:
mask_viz = visualize_mask(image, mask)  # utils.py:4
```

## Important Notes for AI Assistants

### When Modifying This Codebase:

1. **Model Loading is Expensive**: Avoid recreating pipeline objects. Global initialization in `app.py:9` is intentional.

2. **Mask Coordinate Systems**:
   - SAM2 returns masks in original image resolution
   - SDXL resizes to 1024px then back
   - Always verify mask/image shape consistency

3. **HuggingFace Caching**:
   - First run downloads ~10GB of models
   - Cached in `~/.cache/huggingface/`
   - Don't delete this directory in cleanup scripts

4. **GPU Memory Management**:
   - Pipeline holds 3 models simultaneously (~12GB VRAM)
   - CPU offloading (`enable_model_cpu_offload()`) is critical
   - Queue system in Gradio prevents parallel requests

5. **Top-K Merging is Critical**:
   - Handles "cat tail" problem (query matches parts, not whole)
   - Merging logic in `pipeline.py:38-48`
   - Do not simplify to "take top 1" without understanding trade-offs

6. **Prompt Engineering Matters**:
   - Negative prompts in `painter.py:99-103` prevent object hallucination
   - Default background prompt is carefully tuned
   - User prompts should be context-descriptive, not object-descriptive

7. **No Unit Tests**:
   - Project relies on manual Gradio testing
   - When adding features, test via web interface with multiple images

8. **Academic Project Context**:
   - Code prioritizes experimentation over production robustness
   - Some "magic numbers" are empirically tuned, not theoretically justified
   - Comments explain "why" for key heuristics

### Code Modification Guidelines:

**DO:**
- Add docstrings when creating new functions
- Print intermediate results for debugging
- Keep pipeline stages modular and swappable
- Test mask visualization in Gradio debug tab

**DON'T:**
- Remove print statements (they're the logging system)
- Modify mask dilation/blur without testing edge artifacts
- Change Top-K or merge thresholds without A/B testing
- Add dependencies without updating `requirements.txt`

### Testing Strategy:

Since this is a visual ML application:
1. Use Gradio interface for all tests
2. Test with diverse images (simple backgrounds, complex scenes)
3. Verify mask quality in debug tab before checking final result
4. Test edge cases: small objects, large objects, multiple similar objects
5. Monitor GPU memory usage during inference

### Git Workflow:

- **Branch**: `claude/claude-md-mi6x7pidp76m0ua4-018duzA9S5BTzbVUHrHkGMj5`
- **Commit Style**: Recent commits show descriptive messages
  - "Implement SAM2 and better inpainting"
  - "First Implement"
  - "Add project context to README"
- Follow this pattern: verb + brief description

### Model Files (.gitignore):

```
models/yolov8*   # Excluded from version control
```

All HuggingFace models auto-download; no manual model management needed.

---

**Last Updated**: 2025-11-20
**Repository State**: Functional SAM2 + CLIP + SDXL pipeline with Gradio interface
