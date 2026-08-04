"""
config.py
=========
Central configuration for AI Video Studio.

This module is responsible for:
    - Detecting the available hardware (CPU / T4 / L4 / A100 / other CUDA GPUs).
    - Exposing GPU-aware quality presets (resolution, steps, frames) so the
      rest of the application can automatically scale generation quality to
      whatever hardware it is running on (Colab Free T4, Colab Pro L4/A100,
      RunPod, or a local workstation).
    - Declaring the registry of supported video diffusion models. Swapping the
      active model is a one-line change (`ACTIVE_MODEL`), which keeps the
      pipeline architecture modular as new open models are released.

Nothing in this file performs heavy imports (no torch/diffusers) so it can be
imported cheaply from anywhere, including Gradio startup, without slowing
down app boot time.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"
CHARACTERS_DIR = ASSETS_DIR / "characters"
HISTORY_FILE = OUTPUTS_DIR / "prompt_history.json"

for _dir in (ASSETS_DIR, OUTPUTS_DIR, MODELS_DIR, CHARACTERS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry (modular: swap ACTIVE_MODEL to change the backbone)
# --------------------------------------------------------------------------- #

MODEL_REGISTRY = {
    "wan2.1-t2v-1.3b": {
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "type": "text-to-video",
        "min_vram_gb": 8,
        "description": "Wan 2.1 1.3B - lightweight, runs on a free Colab T4.",
    },
    "wan2.1-t2v-14b": {
        "repo_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "type": "text-to-video",
        "min_vram_gb": 40,
        "description": "Wan 2.1 14B - high quality, requires an A100/H100.",
    },
    "cogvideox-2b": {
        "repo_id": "THUDM/CogVideoX-2b",
        "type": "text-to-video",
        "min_vram_gb": 10,
        "description": "CogVideoX 2B - good quality/VRAM tradeoff.",
    },
    "cogvideox-5b": {
        "repo_id": "THUDM/CogVideoX-5b",
        "type": "text-to-video",
        "min_vram_gb": 24,
        "description": "CogVideoX 5B - higher fidelity, needs L4/A100.",
    },
    "ltx-video": {
        "repo_id": "Lightricks/LTX-Video",
        "type": "text-to-video+image-to-video",
        "min_vram_gb": 12,
        "description": "LTX-Video - fast real-time-ish generation.",
    },
    "hunyuanvideo": {
        "repo_id": "tencent/HunyuanVideo",
        "type": "text-to-video",
        "min_vram_gb": 45,
        "description": "HunyuanVideo - state of the art, needs A100 80GB.",
    },
    "cogvideox-5b-i2v": {
        "repo_id": "THUDM/CogVideoX-5b-I2V",
        "type": "image-to-video",
        "min_vram_gb": 24,
        "description": "CogVideoX 5B Image-to-Video variant.",
    },
}

# Change this single value to switch the backbone used across the whole app.
ACTIVE_TEXT2VIDEO_MODEL = os.environ.get("AIVS_T2V_MODEL", "wan2.1-t2v-1.3b")
ACTIVE_IMAGE2VIDEO_MODEL = os.environ.get("AIVS_I2V_MODEL", "cogvideox-5b-i2v")


# --------------------------------------------------------------------------- #
# GPU detection
# --------------------------------------------------------------------------- #

@dataclass
class GPUProfile:
    name: str = "cpu"
    vram_gb: float = 0.0
    tier: str = "cpu"          # one of: cpu, t4, l4, a10, a100, other
    supports_bf16: bool = False
    supports_xformers: bool = True


def _nvidia_smi_name() -> Optional[str]:
    """Fallback GPU name detection via nvidia-smi, used if torch isn't
    importable yet (e.g. before dependencies are installed)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5,
        )
        return out.decode().strip().splitlines()[0]
    except Exception:
        return None


def detect_gpu() -> GPUProfile:
    """Detect the current GPU and classify it into a known tier so that
    generation quality (resolution/steps/frames) can be auto-scaled.
    """
    try:
        import torch  # local import: keep config.py import-light

        if not torch.cuda.is_available():
            return GPUProfile()

        props = torch.cuda.get_device_properties(0)
        name = props.name
        vram_gb = round(props.total_memory / (1024 ** 3), 1)
        supports_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        name = _nvidia_smi_name() or "unknown"
        vram_gb = 0.0
        supports_bf16 = False

    lname = name.lower()
    if "t4" in lname:
        tier = "t4"
    elif "l4" in lname:
        tier = "l4"
    elif "a100" in lname:
        tier = "a100"
    elif "a10" in lname:
        tier = "a10"
    elif "v100" in lname:
        tier = "v100"
    elif "h100" in lname:
        tier = "h100"
    elif "cpu" in lname or name == "unknown":
        tier = "cpu" if vram_gb == 0 else "other"
    else:
        tier = "other"

    return GPUProfile(
        name=name,
        vram_gb=vram_gb,
        tier=tier,
        supports_bf16=supports_bf16,
        supports_xformers=tier != "cpu",
    )


# --------------------------------------------------------------------------- #
# Quality presets, auto-scaled per detected GPU tier
# --------------------------------------------------------------------------- #

@dataclass
class QualityPreset:
    max_resolution: tuple = (512, 512)
    max_frames: int = 49          # ~2s @ 24fps
    default_steps: int = 25
    max_steps: int = 30
    dtype: str = "float16"
    enable_cpu_offload: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_attention_slicing: bool = True
    enable_xformers: bool = True


QUALITY_PRESETS = {
    "cpu": QualityPreset(
        max_resolution=(256, 256), max_frames=17, default_steps=12,
        max_steps=15, dtype="float32", enable_xformers=False,
    ),
    "t4": QualityPreset(
        max_resolution=(512, 512), max_frames=49, default_steps=25,
        max_steps=30, dtype="float16",
    ),
    "v100": QualityPreset(
        max_resolution=(576, 576), max_frames=65, default_steps=30,
        max_steps=35, dtype="float16",
    ),
    "l4": QualityPreset(
        max_resolution=(768, 768), max_frames=81, default_steps=35,
        max_steps=40, dtype="bfloat16",
    ),
    "a10": QualityPreset(
        max_resolution=(768, 768), max_frames=81, default_steps=35,
        max_steps=40, dtype="bfloat16",
    ),
    "a100": QualityPreset(
        max_resolution=(1024, 1024), max_frames=121, default_steps=50,
        max_steps=60, dtype="bfloat16", enable_cpu_offload=False,
    ),
    "h100": QualityPreset(
        max_resolution=(1280, 1280), max_frames=161, default_steps=50,
        max_steps=60, dtype="bfloat16", enable_cpu_offload=False,
    ),
    "other": QualityPreset(),
}


def get_quality_preset(gpu: Optional[GPUProfile] = None) -> QualityPreset:
    gpu = gpu or detect_gpu()
    return QUALITY_PRESETS.get(gpu.tier, QUALITY_PRESETS["other"])


# --------------------------------------------------------------------------- #
# Misc app settings
# --------------------------------------------------------------------------- #

APP_TITLE = "AI Video Studio"
APP_VERSION = "1.0.0"
DEFAULT_FPS = 24
DEFAULT_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3"]
DEFAULT_CAMERA_MOTIONS = [
    "static", "slow pan left", "slow pan right", "zoom in", "zoom out",
    "dolly forward", "dolly backward", "orbit", "handheld",
]
DEFAULT_LIGHTING = [
    "soft natural light", "golden hour", "studio lighting", "neon night",
    "overcast diffuse light", "dramatic low-key", "volumetric sunbeams",
]

GOOGLE_DRIVE_FOLDER = "/content/drive/MyDrive/AI-Video-Studio"
