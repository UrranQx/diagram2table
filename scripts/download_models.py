"""Download models for Diagram2Table ahead of time.

This script pre-downloads model weights to HuggingFace cache to avoid
waiting during service startup. Works for both local development and
Docker builds.

The script downloads only the files (no GPU/CUDA required), making it
suitable for Docker build stage where GPU is not available.

Usage:
    # Download default model (Qwen2-VL-2B-Instruct)
    python scripts/download_models.py
    
    # Download specific model
    python scripts/download_models.py --model qwen2-vl-2b
    python scripts/download_models.py --model qwen2-vl-7b
    
    # Download all supported models
    python scripts/download_models.py --all
    
    # List available models
    python scripts/download_models.py --list
"""

import argparse
import logging
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Registry of supported models
# Key: short name, Value: (HuggingFace repo, description, size estimate)
MODELS = {
    "qwen2-vl-2b": (
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen2-VL 2B - Recommended for 6GB VRAM GPUs (GTX 1660 Ti)",
        "~4.4 GB"
    ),
    "qwen2.5-vl-3b": (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL 3B - Recommended for 8GB VRAM GPUs (RTX 3060)",
        "~6.5 GB"
    ),
    "qwen2-vl-7b": (
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen2-VL 7B - Better quality, needs 16GB+ VRAM",
        "~15 GB"
    ),
    # Future quantized models can be added here:
    # "qwen2-vl-2b-awq": (
    #     "Qwen/Qwen2-VL-2B-Instruct-AWQ",
    #     "Qwen2-VL 2B AWQ quantized - For limited VRAM",
    #     "~2 GB"
    # ),
}

DEFAULT_MODEL = "qwen2.5-vl-3b"


def download_model(model_key: str) -> bool:
    """Download a model's weights to HuggingFace cache.
    
    Uses snapshot_download() which only downloads files without loading
    the model into memory. This is fast, memory-efficient, and works
    without GPU/CUDA (perfect for Docker builds).
    
    Args:
        model_key: Short name from MODELS registry.
        
    Returns:
        True if successful, False otherwise.
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        logger.error(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    repo_id, description, size = MODELS[model_key]
    
    logger.info("=" * 60)
    logger.info(f"Downloading: {description}")
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Estimated size: {size}")
    logger.info("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download all model files to cache
        # This does NOT load the model into memory - just downloads files!
        logger.info("Downloading model files...")
        cache_dir = snapshot_download(
            repo_id=repo_id,
            resume_download=True,  # Resume if interrupted
            local_files_only=False,
        )
        logger.info(f"✓ Model files downloaded to: {cache_dir}")
        
        # Also download processor/tokenizer files
        logger.info("Downloading processor files...")
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                repo_id,
                trust_remote_code=True,
            )
            del processor  # Free memory
            logger.info("✓ Processor files downloaded")
        except Exception as e:
            # Non-critical - processor might already be included
            logger.warning(f"Processor download note: {e}")
        
        logger.info(f"✓ Successfully downloaded {model_key}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {model_key}: {e}")
        return False


def list_models() -> None:
    """Print available models."""
    print("\nAvailable models:")
    print("-" * 70)
    for key, (repo_id, description, size) in MODELS.items():
        default = " (default)" if key == DEFAULT_MODEL else ""
        print(f"  {key}{default}")
        print(f"    Repository: {repo_id}")
        print(f"    {description}")
        print(f"    Size: {size}")
        print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Diagram2Table models ahead of time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default model (recommended)
  python scripts/download_models.py
  
  # Download specific model
  python scripts/download_models.py --model qwen2-vl-7b
  
  # Download all models
  python scripts/download_models.py --all
  
  # See available models
  python scripts/download_models.py --list
        """,
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to download (default: {DEFAULT_MODEL})",
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all available models",
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models and exit",
    )
    
    # Legacy flag for backward compatibility (does nothing now)
    parser.add_argument(
        "--simple",
        action="store_true",
        help=argparse.SUPPRESS,  # Hidden, for backward compatibility
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        list_models()
        return 0
    
    # Determine which models to download
    if args.all:
        to_download = list(MODELS.keys())
        logger.info(f"Downloading ALL models: {', '.join(to_download)}")
    else:
        to_download = [args.model]
        logger.info(f"Downloading model: {args.model}")
    
    # Show cache location
    import os
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    logger.info(f"Cache directory: {hf_home}")
    logger.info("")
    
    # Download models
    results = {}
    for model_key in to_download:
        success = download_model(model_key)
        results[model_key] = success
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    
    for model_key, success in results.items():
        repo_id, description, _ = MODELS[model_key]
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {status}: {model_key} ({repo_id})")
    
    all_success = all(results.values())
    
    if all_success:
        logger.info("")
        logger.info("✓ All models downloaded successfully!")
        logger.info("You can now start the service:")
        logger.info("  python -m src.main both")
        return 0
    else:
        logger.error("")
        logger.error("✗ Some models failed to download.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
