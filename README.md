# SatQuery AI: Agentic Multimodal Remote-Sensing Assistant

SatQuery AI is an agentic vision-language system engineered for Earth Observation (EO) satellite imagery analysis. It deterministically classifies image modalities, routes analytical tasks to dedicated specialist engines, conducts bi-temporal change detection across georeferenced and non-georeferenced imagery, and performs cross-modal optical-SAR fusion synthesis.

---

## Architecture Summary

```
                       ┌─────────────────────────────────────────┐
                       │           User Natural Language         │
                       │           & Satellite Raster(s)         │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │      Deterministic Modality Detector    │
                       │      (Optical / SAR / Multi-modal)      │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │        Intent Router & Controller       │
                       └─┬──────────────┬───────────────┬───────┬─┘
                         │              │               │       │
       ┌─────────────────┘              │               │       └─────────────────┐
       ▼                                ▼               ▼                         ▼
┌──────────────┐                 ┌──────────────┐ ┌──────────────┐        ┌──────────────┐
│Optical VQA   │                 │SAR VQA       │ │Change-VQA    │        │Optical-SAR   │
│GeoChat-7B    │                 │Qwen2-VL-2B   │ │Bi-Temporal   │        │Fusion Engine │
│Port 8001 Warm│                 │LoRA Adapted  │ │Geo / Visual  │        │Joint Physics │
└──────────────┘                 └──────────────┘ └──────────────┘        └──────────────┘
       │                                │               │                         │
       └─────────────────┬──────────────┴───────────────┴─────────────────────────┘
                         ▼
       ┌─────────────────────────────────────────────────────────┐
       │     Conversational Streamlit GUI & Artifact Viewer      │
       └─────────────────────────────────────────────────────────┘
```

1. **Modality Detection (`tools/gis_preprocess/detect_modality.py`)**:
   - Rule-based evidence accumulation combining sensor metadata tags, band counts, data types, channel variance, and value distributions to reliably distinguish optical multispectral rasters from SAR radar cross-sections.

2. **Router & Controller (`tools/router/router.py`)**:
   - Classifies natural language queries and image counts into specialized analytical tasks (`optical_caption`, `optical_vqa`, `sar_vqa`, `change_vqa`, `optical_sar_fusion`).
   - Dispatches tasks to isolated specialist Python environments with process timeouts, structured logging, and fallback mechanisms.

3. **Optical Specialist (`tools/optical_vqa/`)**:
   - Powered by **GeoChat-7B** running as a warm server on port 8001. Provides rapid high-resolution optical scene description, object identification, and spatial question answering.

4. **SAR Specialist (`tools/sar_vqa/`)**:
   - Powered by **Qwen2-VL-2B-Instruct** running with 4-bit BitsAndBytes quantization for efficient radar backscatter interpretation.
   - Includes a domain-adapted LoRA fine-tuning adapter (`tools/sar_vqa/finetune/adapter/`) trained on remote-sensing SAR imagery.

5. **Bi-Temporal Change-VQA (`tools/change_vqa/`)**:
   - **Geospatial Mode**: For georeferenced GeoTIFFs, applies bilinear re-projection onto a common spatial grid (`rasterio.warp.reproject`), physical BOA surface reflectance normalization (0–10,000 scale), pixel difference mapping, and GeoChat bi-temporal synthesis.
   - **Visual/Image-Space Mode**: For non-georeferenced images (JPG/PNG/screenshots), performs image-space alignment, pixel divergence calculation, and zero-shot VLM comparison without requiring CRS metadata.

6. **Optical-SAR Cross-Modal Fusion (`tools/fusion/`)**:
   - Co-registers optical and SAR rasters over the same geographic footprint, extracts NDVI vegetation indices and calibrated SAR backscatter dB metrics, and produces classification masks with synthesized cross-modal reports.

7. **Conversational GUI (`gui/app.py`)**:
   - Dark-themed ChatGPT/Claude-style Streamlit web interface with persistent active image memory for follow-up questions, multi-image attachment tray, and artifact inspection.

---

## Domain-Specific LoRA Adaptation

As proof of domain-specific adaptation for Synthetic Aperture Radar imagery, this repository includes fine-tuned LoRA weights located at:
`tools/sar_vqa/finetune/adapter/`
- `adapter_config.json`: LoRA configuration (r=8, alpha=16, targeting vision-language projection layers).
- `adapter_model.safetensors`: Quantized adapter checkpoint for radar feature interpretation.

---

## Model Weights Notice

In accordance with best practices for code repositories, heavy foundation model weights (> 15 GB) are **not tracked in git**. Model weights must be downloaded separately from Hugging Face Hub:

| Model | Purpose | Download Link |
| :--- | :--- | :--- |
| **GeoChat-7B** | Optical VQA specialist | [HuggingFace - MBZUAI/geochat-7B](https://huggingface.co/MBZUAI/geochat-7B) |
| **Qwen2-VL-2B-Instruct** | SAR VQA specialist | [HuggingFace - Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |

Place GeoChat weights under `models/geochat-7B` or configure paths in `tools/optical_vqa/`.

---

## Environment Setup (Isolated Virtual Environments)

SatQuery AI uses three isolated Python environments to prevent dependency conflicts between geospatial GIS libraries, PyTorch CUDA builds, and specialist model runtimes.

### 1. Main Environment (`venv`)
Used by the Router controller, GIS preprocessing, Fusion engine, and Streamlit GUI:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-main.txt
```

### 2. Optical Specialist Environment (`tools/optical_vqa/venv`)
Used by GeoChat-7B and Optical VQA:
```bash
python -m venv tools\optical_vqa\venv
tools\optical_vqa\venv\Scripts\activate
pip install -r tools/optical_vqa/requirements.txt
```

### 3. SAR Specialist Environment (`tools/sar_vqa/venv`)
Used by Qwen2-VL-2B and SAR fine-tuning:
```bash
python -m venv tools\sar_vqa\venv
tools\sar_vqa\venv\Scripts\activate
pip install -r tools/sar_vqa/requirements.txt
```

---

## How to Run

### Step 1: Launch the Optical Specialist Server (Port 8001)
Keep GeoChat-7B warm in the background for fast sub-second inference:
```bash
tools\optical_vqa\venv\Scripts\python.exe tools\optical_vqa\server.py --port 8001
```

### Step 2: Launch the Streamlit GUI
In a separate terminal, launch the web application:
```bash
venv\Scripts\streamlit.exe run gui\app.py
```
Open your browser at `http://localhost:8501`.

### Step 3: Run via Command-Line Interface (Optional)
You can also invoke the Router directly from the command line:
```bash
venv\Scripts\python.exe tools\router\router.py --images path/to/image1.tif path/to/image2.tif --query "what changed between these images?"
```

---

## Repository Structure

```
satQai/
├── gui/
│   └── app.py                     # Streamlit web application
├── tools/
│   ├── gis_preprocess/
│   │   └── detect_modality.py     # Deterministic sensor modality detector
│   ├── router/
│   │   └── router.py              # Central agentic controller & task router
│   ├── optical_vqa/
│   │   ├── optical_vqa.py         # GeoChat engine wrapper
│   │   ├── optical_vqa_cli.py     # Isolated CLI specialist interface
│   │   ├── server.py              # Warm FastAPI service (Port 8001)
│   │   └── requirements.txt       # Optical venv dependencies
│   ├── sar_vqa/
│   │   ├── sar_vqa_cli.py         # Qwen2-VL SAR specialist CLI
│   │   ├── finetune/              # Domain-specific LoRA adaptation & adapter weights
│   │   └── requirements.txt       # SAR venv dependencies
│   ├── change_vqa/
│   │   ├── change_vqa.py          # Bi-temporal change detection engine
│   │   └── change_vqa_cli.py      # Modality-aware Change-VQA CLI
│   └── fusion/
│       └── fusion.py              # Cross-modal optical-SAR fusion pipeline
├── requirements-main.txt          # Main controller & GUI dependencies
├── .gitignore                     # Git tracking exclusions
└── README.md                      # Project documentation
```

---

## License

This project is released under the Apache 2.0 License.
