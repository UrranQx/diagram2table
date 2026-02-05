"""Configuration management for Diagram2Table."""

import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class DeploymentMode(str, Enum):
    """Deployment mode for the application.
    
    Modes:
    - VLM_ONLY: VLM without OCR (default, FP16 precision)
    - VLM_QUANTIZED: 4-bit quantized VLM via bitsandbytes (uses ~1.5GB VRAM)
    - MOCK: Mock VLM for testing without GPU
    """
    VLM_ONLY = "vlm_only"      # VLM without OCR (default)
    VLM_QUANTIZED = "vlm_quantized"  # VLM 4-bit quantized via bitsandbytes
    MOCK = "mock"              # Mock for testing


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Deployment mode
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.VLM_ONLY,
        description="Deployment mode: vlm_only (default), vlm_quantized, or mock"
    )

    # Model settings
    # Default: 3B model for 4-5GB VRAM (fits most modern GPUs)
    # For weaker GPUs, use 2B variants; for powerful GPUs, use 7B via environment variables
    model_name: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="HuggingFace model name or local path (default: 3B for 4-5GB VRAM)",
    )
    model_name_quantized: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Model for 4-bit quantization via bitsandbytes - uses ~2-2.5GB VRAM",
    )
    model_name_full: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Full precision model name (FP16) - uses ~5-6GB VRAM",
    )
    model_device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto", description="Device for model inference"
    )
    model_dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto", description="Model precision"
    )
    max_image_size: int = Field(default=1280, description="Maximum image dimension")
    min_image_size: int = Field(default=100, description="Minimum image dimension")

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")

    # Gradio settings
    gradio_host: str = Field(default="0.0.0.0", description="Gradio host")
    gradio_port: int = Field(default=7860, description="Gradio port")
    gradio_share: bool = Field(default=False, description="Enable Gradio sharing")
    
    # Gradio API client mode (use FastAPI backend instead of loading model)
    gradio_use_api: bool = Field(
        default=True,
        description="If True, Gradio calls FastAPI instead of loading model directly"
    )
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for FastAPI backend (used when gradio_use_api=True)"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Limits
    max_file_size_mb: int = Field(default=10, description="Maximum file size in MB")

    # Pipeline settings (for colleague's implementation)
    qwen_model_id: Optional[str] = Field(
        default=None,
        description="Override Qwen model ID (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct')"
    )
    tile_size: int = Field(default=1024, description="Tile size for large image processing")
    tile_overlap: int = Field(default=128, description="Overlap between tiles")
    max_new_tokens: int = Field(default=128, description="Max tokens for VLM generation")
    batch_size: int = Field(default=2, description="Batch size for tile processing")

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    def get_device(self) -> str:
        """Get the actual device to use for inference."""
        if self.model_device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.model_device

    def is_quantized(self) -> bool:
        """Check if quantized model should be used."""
        return self.deployment_mode == DeploymentMode.VLM_QUANTIZED

    def get_model_name(self) -> str:
        """Get the appropriate model name based on deployment mode."""
        if self.deployment_mode == DeploymentMode.VLM_QUANTIZED:
            return self.model_name_quantized
        elif self.deployment_mode == DeploymentMode.VLM_ONLY:
            return self.model_name_full
        return self.model_name


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def init_settings(**kwargs) -> Settings:
    """Initialize settings with custom values."""
    global _settings
    _settings = Settings(**kwargs)
    return _settings
