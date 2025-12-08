# Setup Instructions

## Prerequisites
- **OS**: Linux, Windows, or macOS
- **Python**: 3.8+ (3.10 recommended)
- **Hardware**: NVIDIA GPU (12GB+ VRAM) recommended for optimal performance.

## Installation

### 1. Environment Setup
We recommend using micromamba to isolate dependencies:

```bash
micromamba create -n texteraser python=3.10
micromamba activate texteraser
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
Launch the Gradio interface. Note: The first run will automatically download required models (~10GB).
```bash
python app.py
```