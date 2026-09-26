#!/usr/bin/env python3
"""SatQueryAI — Modern Conversational Multimodal Satellite Assistant.

Generic chat interface (Claude/ChatGPT style):
- Image persistence: Upload ONCE, ask unlimited follow-up questions on the same image
- High-contrast text readability (crisp white #f1f3f4 on dark theme #131314)
- Sleek purple-accented avatar (no distracting orange Streamlit icon)
- Uploading a new image replaces active image; '+ New Chat' clears session
- Subtle decorative background pattern (6% opacity, rocket/satellite themed)
- Generic file uploader as the sole image input (router determines task)
- Preserves all router, warm GeoChat-7B, and isolated SAR execution logic
"""

import os
import sys
import time
import warnings
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*geotransform.*")
warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")
import json
import uuid
import subprocess
from typing import List, Optional, Dict, Any
import requests
import streamlit as st
from PIL import Image
import numpy as np

# Ensure satQai root and router are on sys.path
PROJECT_ROOT = r"D:\satQai"
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
ROUTER_DIR = os.path.join(TOOLS_DIR, "router")

for p in [PROJECT_ROOT, ROUTER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from router import run_satquery, OPTICAL_SERVER_URL

# ==============================================================================
# STREAMLIT PAGE CONFIG & MODERN DARK THEME CSS
# ==============================================================================
st.set_page_config(
    page_title="SatQueryAI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Typography & Deep Charcoal Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    section.main,
    .main,
    .block-container {
        background-color: #131314 !important;
        background: #131314 !important;
        color: #f1f3f4 !important;
        position: relative;
    }

    /* Subtle Decorative Background (5-7% Opacity Rocket/Satellite Constellation Pattern) */
    [data-testid="stAppViewContainer"]::before,
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 0;
        opacity: 0.06;
        background-image: 
            radial-gradient(circle at 18% 22%, rgba(139, 92, 246, 0.8) 1.5px, transparent 2.5px),
            radial-gradient(circle at 82% 16%, rgba(139, 92, 246, 0.9) 2px, transparent 3.5px),
            radial-gradient(circle at 75% 78%, rgba(139, 92, 246, 0.7) 1.5px, transparent 2.5px),
            radial-gradient(circle at 25% 85%, rgba(139, 92, 246, 0.8) 2px, transparent 3px),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='260' height='260' viewBox='0 0 260 260' fill='none'%3E%3Cpath d='M130 35 L135 48 L148 50 L138 60 L141 73 L130 66 L119 73 L122 60 L112 50 L125 48 Z' stroke='%238b5cf6' stroke-width='1.2' fill='none'/%3E%3Ccircle cx='130' cy='130' r='55' stroke='%238b5cf6' stroke-dasharray='4 6' stroke-width='1'/%3E%3Cpath d='M75 130 C75 100 185 100 185 130 C185 160 75 160 75 130 Z' stroke='%238b5cf6' stroke-width='0.8' stroke-dasharray='3 5'/%3E%3Crect x='122' y='122' width='16' height='16' rx='3' stroke='%238b5cf6' stroke-width='1.2' fill='none'/%3E%3Cline x1='112' y1='130' x2='122' y2='130' stroke='%238b5cf6' stroke-width='1.2'/%3E%3Cline x1='138' y1='130' x2='148' y2='130' stroke='%238b5cf6' stroke-width='1.2'/%3E%3Crect x='102' y='125' width='10' height='10' stroke='%238b5cf6' stroke-width='0.8'/%3E%3Crect x='148' y='125' width='10' height='10' stroke='%238b5cf6' stroke-width='0.8'/%3E%3Cpath d='M205 60 L212 48 M212 48 L220 56 M212 48 L202 38' stroke='%238b5cf6' stroke-width='1'/%3E%3C/svg%3E");
        background-repeat: repeat;
        background-size: auto, auto, auto, auto, 260px 260px;
    }

    /* Hide Streamlit default header decoration */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    footer {
        display: none !important;
    }

    /* Sidebar Styling (ChatGPT/Claude style) */
    [data-testid="stSidebar"] {
        background-color: #1e1f20;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
        padding-top: 1rem;
        z-index: 10;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        margin-bottom: 12px;
    }
    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #c4b5fd 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Status indicator pill (Single Purple Accent) */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        background: rgba(139, 92, 246, 0.14);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }

    /* Chat container max-width */
    .main .block-container {
        max-width: 860px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
        position: relative;
        z-index: 1;
    }

    /* Hero Starter Section */
    .hero-container {
        text-align: center;
        padding: 2.8rem 1rem 1.8rem 1rem;
    }
    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.6rem;
        color: #8b5cf6;
        filter: drop-shadow(0 0 18px rgba(139, 92, 246, 0.5));
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.8px;
        margin-bottom: 0.6rem;
        background: linear-gradient(135deg, #ffffff 0%, #c4c7c5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c4c7c5;
        max-width: 640px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }

    /* Universal Button Styling: Dark Elevated Surfaces (#1e1f20) */
    div[data-testid="stButton"] > button,
    button[data-testid="baseButton-secondary"],
    button[kind="secondary"],
    .stButton > button {
        background-color: #1e1f20 !important;
        color: #f1f3f4 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
        padding: 0.45rem 0.85rem !important;
        box-shadow: none !important;
        transition: all 0.15s ease-in-out !important;
    }

    div[data-testid="stButton"] > button:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[kind="secondary"]:hover,
    .stButton > button:hover {
        background-color: #282a2c !important;
        border-color: #8b5cf6 !important;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.25) !important;
    }

    /* Primary Button (+ New Chat): Single Accent Purple */
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"],
    div[data-testid="stButton"] > button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        background: #8b5cf6 !important;
        color: #ffffff !important;
        border: 1px solid #8b5cf6 !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3) !important;
    }

    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover {
        background: #7c3aed !important;
        border-color: #7c3aed !important;
        box-shadow: 0 0 14px rgba(139, 92, 246, 0.45) !important;
    }

    /* Sidebar Button Alignment */
    [data-testid="stSidebar"] div[data-testid="stButton"] > button {
        text-align: left !important;
        justify-content: flex-start !important;
    }

    /* ========================================================= */
    /* CHAT MESSAGE & TEXT VISIBILITY ENHANCEMENTS               */
    /* ========================================================= */
    .stChatMessage,
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] * {
        color: #f1f3f4 !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] p,
    .stChatMessage [data-testid="stMarkdownContainer"] li,
    .stChatMessage [data-testid="stMarkdownContainer"] span,
    .stChatMessage [data-testid="stMarkdownContainer"] div {
        color: #f1f3f4 !important;
        font-size: 0.98rem !important;
        line-height: 1.65 !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] strong {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] h1,
    .stChatMessage [data-testid="stMarkdownContainer"] h2,
    .stChatMessage [data-testid="stMarkdownContainer"] h3 {
        color: #ffffff !important;
    }
    .stChatMessage code {
        color: #e9d5ff !important;
        background: rgba(139, 92, 246, 0.18) !important;
        border: 1px solid rgba(139, 92, 246, 0.35) !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
    }

    /* Captions, Timestamps, and Small Notes */
    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stImageCaption"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] span {
        color: #c4c7c5 !important;
        font-size: 0.84rem !important;
    }

    /* Assistant Message Container: Sleek Dark Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #1a1a1c !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.85rem !important;
    }

    /* User Message Container: Subtle Purple Elevated Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: rgba(139, 92, 246, 0.09) !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        border-radius: 12px !important;
        padding: 0.85rem 1rem !important;
        margin-bottom: 0.85rem !important;
    }

    /* REMOVE DISTRACTING ORANGE AVATAR - Style with Purple Satellite Accent */
    [data-testid="stChatMessageAvatarAssistant"],
    div[data-testid="stChatMessageAvatarAssistant"],
    div[data-testid*="AvatarAssistant"] {
        background-color: rgba(139, 92, 246, 0.22) !important;
        border: 1.5px solid #8b5cf6 !important;
        border-radius: 50% !important;
        color: #c4b5fd !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] svg,
    div[data-testid*="AvatarAssistant"] svg {
        display: none !important;
    }

    [data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: #282a2c !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 50% !important;
        color: #f1f3f4 !important;
    }

    /* Technical Details Accordion */
    .stExpander {
        background-color: #1e1f20 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        margin-top: 10px !important;
    }
    .stExpander:hover {
        border-color: rgba(139, 92, 246, 0.35) !important;
    }
    .stExpander summary,
    .stExpander summary span,
    .stExpander summary p {
        color: #f1f3f4 !important;
    }

    /* Active Satellite Image Persistence Card */
    .active-image-card {
        background: #1e1f20;
        border: 1.5px solid #8b5cf6;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 0 16px rgba(139, 92, 246, 0.18);
    }

    /* Compact Thumbnail Styling with Rounded Border (Claude / ChatGPT / Gemini style: 120-160px) */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stImage"] img,
    div[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[data-testid="stImage"] img,
    div[data-testid="stImage"] img {
        width: 140px !important;
        max-height: 140px !important;
        border-radius: 10px !important;
        border: 1.5px solid rgba(139, 92, 246, 0.45) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
        object-fit: cover !important;
        transition: transform 0.15s ease, border-color 0.15s ease !important;
    }
    div[data-testid="stImage"] img:hover {
        border-color: #8b5cf6 !important;
        transform: scale(1.02);
    }
    div[data-testid="stImageCaption"] {
        font-size: 0.74rem !important;
        color: #c4c7c5 !important;
        max-width: 140px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        padding-top: 2px !important;
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        background-color: #1e1f20 !important;
        border: 1.5px dashed rgba(139, 92, 246, 0.45) !important;
        border-radius: 12px !important;
        padding: 0.6rem 0.9rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 14px rgba(139, 92, 246, 0.2) !important;
    }
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="stFileUploader"] section {
        background-color: #1e1f20 !important;
        background: #1e1f20 !important;
        color: #f1f3f4 !important;
    }
    div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] p {
        color: #c4c7c5 !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #282a2c !important;
        color: #f1f3f4 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        border-color: #8b5cf6 !important;
    }

    /* Bottom Dock / Container */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    .stChatInputContainer,
    div[data-testid="stChatInputContainer"] {
        background-color: #131314 !important;
        background: #131314 !important;
        border: none !important;
    }

    /* Chat Input Box: Solid Purple */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-baseweb="textarea"],
    div[data-testid="stChatInput"] textarea {
        background-color: #8b5cf6 !important;
        background: #8b5cf6 !important;
        color: #ffffff !important;
        border-color: transparent !important;
    }

    div[data-testid="stChatInput"] {
        border-radius: 26px !important;
        border: 1.5px solid #a78bfa !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.45) !important;
        overflow: hidden !important;
    }

    div[data-testid="stChatInput"]:focus-within,
    div[data-testid="stChatInput"]:focus-within > div,
    div[data-testid="stChatInput"]:focus-within [data-baseweb="base-input"],
    div[data-testid="stChatInput"]:focus-within [data-baseweb="textarea"],
    div[data-testid="stChatInput"]:focus-within textarea {
        background-color: #7c3aed !important;
        background: #7c3aed !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 24px rgba(139, 92, 246, 0.6) !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        caret-color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder,
    div[data-testid="stChatInput"] [data-baseweb="base-input"] input::placeholder {
        color: rgba(255, 255, 255, 0.85) !important;
        -webkit-text-fill-color: rgba(255, 255, 255, 0.85) !important;
    }

    /* Send Button */
    div[data-testid="stChatInput"] button {
        background-color: #ffffff !important;
        color: #8b5cf6 !important;
        border: none !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background-color: #f1f3f4 !important;
        transform: scale(1.05);
    }
    div[data-testid="stChatInput"] button svg {
        fill: #8b5cf6 !important;
        stroke: #8b5cf6 !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #131314;
    }
    ::-webkit-scrollbar-thumb {
        background: #2a2b2d;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #8b5cf6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# HELPER FUNCTIONS: RASTER PREVIEWS & HEALTH CHECKS
# ==============================================================================
def check_optical_server_health() -> bool:
    """Check if the optical server (GeoChat-7B) is warm on port 8001."""
    try:
        r = requests.get(f"{OPTICAL_SERVER_URL}/health", timeout=0.8)
        if r.status_code == 200:
            return r.json().get("model_loaded", False)
        return False
    except Exception:
        return False


def render_image_preview(image_path: str, max_size: int = 140) -> Optional[Image.Image]:
    """
    Load and render high-quality RGB preview for GeoTIFF or standard image.
    Uses exact specialist preprocessing:
    - Optical: 3-band RGB with 2-98% percentile stretch (same as optical_vqa / caption_image)
    - SAR: Single-band dB calibration (20*log10(DN) - 83) + 2-98% percentile stretch to 0-255 (mean ~105)
    Capped at 120-160px (default 140px) maintaining aspect ratio with max-height cap (Claude/ChatGPT style).
    """
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".tif", ".tiff"]:
            import rasterio
            with rasterio.open(image_path) as src:
                if src.count >= 3:
                    r = src.read(1).astype(np.float32)
                    g = src.read(2).astype(np.float32)
                    b = src.read(3).astype(np.float32)
                    arr = np.stack([r, g, b], axis=-1)
                    
                    nz = (r > 0) | (g > 0) | (b > 0)
                    if np.any(nz):
                        p2, p98 = np.percentile(arr[nz], (2.0, 98.0))
                    else:
                        p2, p98 = np.percentile(arr, (2.0, 98.0))
                        
                    if p98 > p2:
                        arr_norm = np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        arr_norm = np.clip(arr, 0, 255).astype(np.uint8)
                        
                    if np.any(nz):
                        arr_norm[~nz] = 0
                        
                    img = Image.fromarray(arr_norm, mode="RGB")
                else:
                    raw = src.read(1).astype(np.float32)
                    nz = raw > 0
                    if np.any(nz):
                        dn_safe = np.where(nz, raw, 1e-6)
                        db = 20.0 * np.log10(dn_safe) - 83.0
                        p2, p98 = np.percentile(db[nz], (2.0, 98.0))
                        if p98 > p2:
                            norm = np.clip((db - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                        else:
                            norm = np.clip(db, 0, 255).astype(np.uint8)
                        norm[~nz] = 0
                    else:
                        norm = np.zeros_like(raw, dtype=np.uint8)
                    img = Image.fromarray(norm, mode="L").convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")

        # Compact thumbnail sizing (140px) maintaining aspect ratio with max-height cap
        if max(img.width, img.height) > max_size:
            aspect = img.height / img.width
            if img.width >= img.height:
                new_w = max_size
                new_h = max(1, int(max_size * aspect))
            else:
                new_h = max_size
                new_w = max(1, int(max_size / aspect))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return img
    except Exception as exc:
        print(f"Error generating preview for {image_path}: {exc}")
        return None


# ==============================================================================
# SESSION STATE MANAGEMENT
# ==============================================================================
if "sessions" not in st.session_state:
    st.session_state.sessions = {
        "default": {
            "title": "New Conversation",
            "messages": [],
            "active_images": [],  # PERSISTENT IMAGERY PER THREAD
        }
    }
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "default"

if "uploader_key_version" not in st.session_state:
    st.session_state.uploader_key_version = 0

current_session = st.session_state.sessions[st.session_state.current_session_id]
if "active_images" not in current_session:
    current_session["active_images"] = []

# Optional query parameter for automated testing/demo: ?preload=sar or ?preload=optical
if "preload" in st.query_params and not current_session.get("active_images"):
    p_val = st.query_params.get("preload")
    if p_val == "sar":
        current_session["active_images"] = ["D:/satQai/data/real_test/sentinel1_mysuru_registered_vv.tif"]
    elif p_val == "optical":
        current_session["active_images"] = ["D:/satQai/data/real_test/sentinel2_mysuru.tif"]
    elif p_val == "change":
        current_session["active_images"] = ["D:/satQai/data/real_test/sentinel2_mysuru_earlier.tif", "D:/satQai/data/real_test/sentinel2_mysuru.tif"]


# ==============================================================================
# SIDEBAR: CHAT HISTORY & PERSISTENCE (NO PRESETS / NO SAMPLE SCENARIOS)
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span style="color: #8b5cf6; font-size: 1.3rem;">🛰️</span>
            <span class="sidebar-brand-title">SatQueryAI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # New Chat Button - CLEARS IMAGES AND CONVERSATION
    if st.button("+  New Chat", key="new_chat_btn", type="primary", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = {
            "title": "New Conversation",
            "messages": [],
            "active_images": [],  # CLEARED FRESH
        }
        st.session_state.current_session_id = new_id
        st.session_state.uploader_key_version += 1
        st.query_params.clear()
        st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.caption("RECENT CONVERSATIONS")

    # List of conversation threads
    for s_id, s_data in list(st.session_state.sessions.items()):
        is_active = s_id == st.session_state.current_session_id
        title = s_data.get("title", "Conversation")[:24]

        col_a, col_b = st.columns([5, 1])
        with col_a:
            label = f"◈  {title}" if is_active else f"◇  {title}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"session_btn_{s_id}", type=btn_type, use_container_width=True):
                st.session_state.current_session_id = s_id
                st.session_state.uploader_key_version += 1
                st.rerun()
        with col_b:
            if len(st.session_state.sessions) > 1:
                if st.button("×", key=f"del_{s_id}", help="Delete chat"):
                    del st.session_state.sessions[s_id]
                    st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
                    st.session_state.uploader_key_version += 1
                    st.rerun()

    # ARCHITECTURE & HARDWARE STATUS removed per user specification


# ==============================================================================
# MAIN CONVERSATION VIEW
# ==============================================================================

# 1. EMPTY STATE HERO (if no messages in current session)
if not current_session["messages"]:
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-icon">🛰️</div>
            <div class="hero-title">What would you like to analyze today?</div>
            <div class="hero-subtitle">
                SatQueryAI is an agentic satellite assistant. Attach any optical or SAR raster(s) below once, 
                and ask unlimited follow-up questions — the router detects modalities and retains your imagery.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# 2. RENDER CHAT HISTORY (Bright, High-Contrast Text with Sleek Satellite Avatar)
for msg in current_session["messages"]:
    role = msg["role"]
    avatar = "👤" if role == "user" else "🛰️"

    with st.chat_message(role, avatar=avatar):
        # If user attached images specifically in this message, render compact thumbnail (120-160px width)
        if role == "user" and msg.get("images"):
            cols = st.columns([1, 1, 4]) if len(msg["images"]) > 1 else st.columns([1, 4])
            for idx, img_path in enumerate(msg["images"][:2]):
                with cols[idx]:
                    preview = render_image_preview(img_path, max_size=140)
                    if preview:
                        st.image(preview, caption=os.path.basename(img_path), width=140)

        # Message Text (Bright, crisp #f1f3f4)
        st.markdown(msg["content"])

        # If assistant generated visual artifacts (e.g. Fusion map, difference map)
        if role == "assistant" and msg.get("artifacts"):
            for art_path in msg["artifacts"]:
                if isinstance(art_path, str) and art_path.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(art_path):
                    st.markdown(f"**Visual Deliverable:** `{os.path.basename(art_path)}`")
                    st.image(art_path, use_container_width=True)

        # Collapsible Technical Details (Clean ChatGPT/Claude style)
        if role == "assistant" and msg.get("diagnostics"):
            diag = msg["diagnostics"]
            with st.expander("Technical Insights & Modality Audit", expanded=False):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown(f"**Task Classification:** `{diag.get('task', '').upper()}`")
                    st.markdown(f"**Specialist Engine:** `{', '.join(diag.get('specialists_used', []))}`")
                    meta = diag.get("metadata", {})
                    routing_type = meta.get("routing_path") or ("HTTP warm server (" + meta.get("server_url", "") + ")" if meta.get("server_url") else "Subprocess")
                    st.markdown(f"**Routing Architecture:** `{routing_type}`")
                with col_d2:
                    timing = diag.get("timing", {})
                    meta = diag.get("metadata", {})
                    load_t = meta.get("load_time_s", 0.0)
                    infer_t = meta.get("inference_time_s", 0.0)
                    total_t = timing.get("router_total_time_s", 0.0)
                    st.markdown(f"**Model Load Time:** `{load_t}s` | **Inference:** `{infer_t}s`")
                    st.markdown(f"**Total Execution Turnaround:** `{total_t}s`")

                modalities = diag.get("modalities", [])
                if modalities:
                    st.markdown("**Modality Detection Evidence:**")
                    for m in modalities:
                        st.markdown(f"- `{os.path.basename(m.get('path', ''))}`: **{m.get('modality', '').upper()}** (confidence: `{m.get('confidence', 0):.3f}`)")

                with st.expander("Full JSON Response Payload"):
                    st.json(diag)


# ==============================================================================
# IMAGE PERSISTENCE & COMPOSER AREA
# ==============================================================================

# A. Active Persistent Image Tray (Persists across ALL follow-up questions)
active_images = current_session.get("active_images", [])
if active_images:
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    count_str = f"{len(active_images)} attached — persistent for follow-up questions"
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <div style="font-size: 0.88rem; font-weight: 600; color: #f1f3f4;">
                <span style="color: #4ade80; margin-right: 6px;">●</span> Active Satellite Imagery ({count_str}):
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    tray_cols = st.columns([1, 1, 4]) if len(active_images) > 1 else st.columns([1, 4])
    for idx, p in enumerate(active_images[:2]):
        with tray_cols[idx]:
            prev = render_image_preview(p, max_size=140)
            if prev:
                st.image(prev, caption=os.path.basename(p), width=140)

    col_btn_rm, _ = st.columns([1, 4])
    with col_btn_rm:
        clear_label = "🗑️ Remove Active Images" if len(active_images) > 1 else "🗑️ Remove Active Image"
        if st.button(clear_label, key="clear_active_btn"):
            current_session["active_images"] = []
            st.session_state.uploader_key_version += 1
            st.query_params.clear()
            st.rerun()

# B. File Uploader: "The only way to add an image should be the generic file uploader"
if len(active_images) >= 2:
    uploader_label = "Replace Active Satellite Images (Upload 1 or 2 new files):"
elif len(active_images) == 1:
    uploader_label = "Attach 2nd Image for Temporal Change / Fusion (or upload 2 files to replace):"
else:
    uploader_label = "Attach Satellite Imagery (GeoTIFF, TIFF, PNG, JPG) — 1 or 2 files:"

uploaded_files = st.file_uploader(
    uploader_label,
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"uploader_v{st.session_state.uploader_key_version}",
    help="Upload 1 or 2 satellite rasters. The images stay active for all follow-up questions until replaced.",
)

if uploaded_files:
    upload_dir = os.path.join(PROJECT_ROOT, "outputs", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    new_paths = []
    for uf in uploaded_files[:2]:
        save_path = os.path.join(upload_dir, uf.name)
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        new_paths.append(save_path)
    
    if len(new_paths) >= 2:
        # User dropped 2 files at once -> set both as active
        current_session["active_images"] = new_paths[:2]
    elif len(new_paths) == 1:
        existing = current_session.get("active_images", [])
        # If 1 image already active and new image is different -> combine into bi-temporal pair!
        if len(existing) == 1 and existing[0] != new_paths[0]:
            current_session["active_images"] = [existing[0], new_paths[0]]
        else:
            current_session["active_images"] = new_paths

    st.session_state.uploader_key_version += 1
    st.rerun()

# C. Generic Chat Input (Always available for free-text questions)
chat_input_placeholder = (
    "Ask any question about the active satellite imagery..."
    if active_images
    else "Attach a satellite image above and ask any question..."
)
user_query = st.chat_input(chat_input_placeholder)

if user_query:
    active_imgs = list(current_session.get("active_images", []))
    
    # If user sent without images, prompt user to attach
    if not active_imgs:
        st.warning("Please attach at least one satellite raster using the file picker above.")
    else:
        # 1. Append user message to history
        current_session["messages"].append({
            "role": "user",
            "content": user_query,
            "images": active_imgs if len(current_session["messages"]) == 0 else [], # Only show preview thumbnail on first turn
        })

        # Update title if it was a new chat
        if current_session["title"] == "New Conversation":
            current_session["title"] = user_query[:28] + ("..." if len(user_query) > 28 else "")

        # 2. Render user message in UI immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        # 3. Stream/execute assistant response with router
        with st.chat_message("assistant", avatar="🛰️"):
            with st.status("SatQueryAI is analyzing satellite raster with specialist...", expanded=True) as status_box:
                t_start = time.time()
                result = run_satquery(active_imgs, user_query)
                t_dur = round(time.time() - t_start, 2)
                task_name = result.get("task", "analysis")
                status_box.update(label=f"Completed {task_name.upper()} in {t_dur}s", state="complete", expanded=False)

            # Assistant text (Bright, readable)
            answer_text = result.get("answer", "")
            if "error" in result:
                raw_err = str(result['error'].get('message', 'Specialist failure'))
                clean_err = re.sub(r'[\w\\/]+[\\/]rasterio[\\/].*?NotGeoreferencedWarning.*?\n?', '', raw_err)
                clean_err = re.sub(r'NotGeoreferencedWarning.*?\n?', '', clean_err)
                st.error(f"**Error:** {clean_err.strip()}")
            else:
                st.markdown(answer_text)

            # Visual deliverables
            artifacts = result.get("artifacts", [])
            valid_arts = [
                a for a in artifacts
                if isinstance(a, str) and a.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(a)
            ]
            for art_path in valid_arts:
                st.markdown(f"**Visual Deliverable:** `{os.path.basename(art_path)}`")
                st.image(art_path, use_container_width=True)

            # Append assistant message to history
            current_session["messages"].append({
                "role": "assistant",
                "content": answer_text,
                "artifacts": valid_arts,
                "diagnostics": result,
            })

            # CRITICAL: DO NOT CLEAR active_images!
            # It persists in current_session["active_images"] for unlimited follow-ups!
            st.rerun()
