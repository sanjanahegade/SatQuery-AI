# SatQuery AI - Agentic Multimodal Remote-Sensing Assistant

SatQuery AI is an agentic, multimodal remote-sensing analysis framework designed for Earth Observation (EO) intelligence. It unifies high-resolution multispectral optical imagery (Sentinel-2) and Synthetic Aperture Radar (Sentinel-1 SAR) through an automated deterministic modality classifier, an intelligent query router, specialist vision-language models, and a cross-sensor geospatial fusion engine.

---

## Key Features

- **Automated Modality Detection**: Automatically inspects raster metadata, band counts, dynamic ranges, and statistical variance to identify Optical (multispectral) vs. SAR (microwave backscatter) rasters without user annotation.
- **Optical VQA Specialist (GeoChat-7B)**: Employs GeoChat-7B (hosted as a persistent warm FastAPI microservice on port 8001) for detailed land-use classification, infrastructure detection, and conversational question answering.
- **SAR Radar Specialist (Qwen2-VL-2B)**: Interprets microwave surface roughness, structural dielectric properties, and water-land contrast using 4-bit quantized vision-language models.
- **Bi-Temporal Change-VQA**:
  - **Geospatial Mode**: Re-projects GeoTIFF pairs to a common spatial grid (`rasterio.warp`), normalizes surface reflectance, computes normalized difference change maps, and produces natural language change explanations.
  - **Visual/Image-Space Mode**: Aligns non-georeferenced images (JPG/PNG/aerial photography) in pixel coordinate space for zero-shot change detection without requiring CRS or spatial geotransforms.
- **Optical-SAR Cross-Modal Fusion**: Co-registers Sentinel-2 and Sentinel-1 scenes, extracts calibrated radar backscatter alongside optical greenness proxy indices, and synthesizes joint physical reports.
- **Dark-Themed Streamlit Interface**: High-contrast, responsive chat interface with multi-image attachment previews, persistent image memory across conversational turns, and inline artifact inspection.

---

## Architecture Summary

```
                       ┌─────────────────────────────────────────┐
                       │          User Natural Language          │
                       │          & Satellite Raster(s)          │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │     Deterministic Modality Detector     │
                       │      (Optical / SAR / Multi-modal)      │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │       Intent Router & Controller        │
                       └─┬──────────────┬──────────────┬────────┬┘
                         │              │              │        │
       ┌─────────────────┘              │              │        └────────────────┐
       │                                │              │                         │
       ▼                                ▼              ▼                         ▼
┌────────────────┐             ┌────────────────┐┌───────────────┐      ┌────────────────┐
│  Optical VQA   │             │    SAR VQA     ││  Change-VQA   │      │  Optical-SAR   │
│  GeoChat-7B    │             │  Qwen2-VL-2B   ││  Bi-Temporal  │      │ Fusion Engine  │
│ Port 8001 Warm │             │   (zero-shot   ││ Geo / Visual  │      │ Joint Physics  │
│                │             │in production)  ││               │      │                │
└──────┬─────────┘             └────────┬───────┘└───────┬───────┘      └────────┬───────┘
       │                                │                │                       │
       └────────────────────────────────┼────────────────┴───────────────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────────────────────┐
                       │ Conversational Streamlit GUI & Artifacts│
                       └─────────────────────────────────────────┘
```

1. **Modality Detection (`tools/gis_preprocess/detect_modality.py`)**:
   - Rule-based evidence accumulation combining sensor metadata tags, band counts, data types, channel variance, and value distributions to reliably distinguish optical multispectral rasters from SAR radar cross-sections.

2. **Router & Controller (`tools/router/router.py`)**:
   - Classifies natural language queries and image counts into specialized analytical tasks (`optical_caption`, `optical_vqa`, `sar_vqa`, `change_vqa`, `optical_sar_fusion`).
   - Dispatches tasks to isolated specialist Python environments with process timeouts, structured logging, and fallback mechanisms.

3. **Optical Specialist (`tools/optical_vqa/`)**:
   - Powered by **GeoChat-7B** running as a warm server on port 8001. Provides rapid high-resolution optical scene description, object identification, and spatial question answering.

4. **SAR Specialist (`tools/sar_vqa/`)**:
   - Powered by **Qwen2-VL-2B-Instruct** running with 4-bit BitsAndBytes quantization for efficient radar backscatter interpretation (operating zero-shot in the production serving pipeline).
   - Accompanied by a standalone domain-adapted LoRA fine-tuning artifact (`tools/sar_vqa/finetune/adapter/`) demonstrating parameter-efficient adaptation on remote-sensing SAR imagery.

5. **Bi-Temporal Change-VQA (`tools/change_vqa/`)**:
   - **Geospatial Mode**: For georeferenced GeoTIFFs, applies bilinear re-projection onto a common spatial grid (`rasterio.warp.reproject`), physical BOA surface reflectance normalization (0-10,000 scale), pixel difference mapping, and GeoChat bi-temporal synthesis.
   - **Visual/Image-Space Mode**: For non-georeferenced images (JPG/PNG/screenshots), performs image-space alignment, pixel divergence calculation, and zero-shot VLM comparison without requiring CRS metadata.

6. **Optical-SAR Cross-Modal Fusion (`tools/fusion/`)**:
   - Co-registers optical and SAR rasters over the same geographic footprint, extracts a visible green-red normalized difference / greenness proxy index `((G - R) / (G + R))` (since 3-band RGB rasters lack a near-infrared / B08 channel, this serves as a visible-band vegetation proxy rather than true NIR-based NDVI) alongside calibrated SAR backscatter dB metrics, and produces classification masks with synthesized cross-modal reports.

7. **Conversational GUI (`gui/app.py`)**:
   - Dark-themed ChatGPT/Claude-style Streamlit web interface with persistent active image memory for follow-up questions, multi-image attachment tray, and artifact inspection.

---

## Domain-Specific LoRA Adaptation (Proof of Concept)

As proof of domain-specific adaptation capability for Synthetic Aperture Radar imagery, this repository includes fine-tuned LoRA weights located at:
`tools/sar_vqa/finetune/adapter/`
- `adapter_config.json`: LoRA configuration (`r=8`, `alpha=16`, targeting vision-language projection layers).
- `adapter_model.safetensors`: Lightweight PEFT/LoRA adapter checkpoint (~4.17 MB).

> [!NOTE]
> **Scope & Live Pipeline Status**: The live production pipeline (`tools/sar_vqa/sar_vqa_cli.py`) executes base **Qwen2-VL-2B-Instruct zero-shot**. This LoRA adapter is a standalone proof-of-concept (trained on 90 SAR image-question pairs for 1 epoch) demonstrating that parameter-efficient fine-tuning on SAR imagery is feasible within consumer hardware constraints. It is not integrated into the active serving path; further training on larger, diverse multi-sensor SAR datasets (such as full HRSID, SSDD, or RS-VQA) would be required for reliable SAR-domain accuracy.

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
│   ├── fusion/
│       └── fusion.py              # Cross-modal optical-SAR fusion pipeline
│   └── scripts/
│       ├── download_satellite_stac.py # Planetary Computer STAC retrieval utility
│       ├── query_stac.py              # STAC catalog query & search utility
│       └── validate_modality_detector.py # Sensor modality validation test suite
├── requirements-main.txt          # Main controller & GUI dependencies
├── .gitignore                     # Git tracking exclusions
└── README.md                      # Project documentation
```

---

## Licenses & Third-Party Terms

### 1. SatQuery AI Codebase
The original software in this repository (task router, GIS preprocessing, optical-SAR fusion engine, bi-temporal change detection pipeline, and Streamlit web application) is licensed under the **[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)**.

### 2. Foundation Model Weights & Downstream Licenses
Foundation model weights are not hosted in this repository and are governed by their respective upstream creators' licenses:

| Component | Architecture / Lineage | License | Commercial Permissibility |
| :--- | :--- | :--- | :--- |
| **SatQuery AI Code** | Original software | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Permitted |
| **Qwen2-VL-2B-Instruct** | Qwen Team / Alibaba Cloud | [Apache 2.0](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Permitted |
| **GeoChat-7B** | LLaVA-1.5 / Vicuna-1.5 / Meta LLaMA-2 | [LLaMA 2 Community License](https://ai.meta.com/llama/license/) & [Vicuna Research Terms](https://github.com/lm-sys/FastChat#license) | Non-Commercial / Research Only |

> [!IMPORTANT]
> **GeoChat Research Restriction**: GeoChat-7B is built upon LLaVA-v1.5 and Vicuna-1.5, which in turn inherits the Meta LLaMA-2 Community License and incorporates training on ShareGPT conversational data. Consequently, GeoChat weights carry non-commercial research restrictions separate from SatQuery AI's own Apache 2.0 code license. Commercial deployment would require substituting the optical specialist backbone with a fully permissively licensed open-weights vision-language model.
