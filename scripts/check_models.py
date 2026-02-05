"""Check if models are already downloaded.

Quick utility to verify model availability before starting the service.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_model_cached(model_name: str) -> tuple[bool, str]:
    """Check if a model is already cached locally.
    
    Args:
        model_name: HuggingFace model identifier.
        
    Returns:
        Tuple of (is_cached, cache_path)
    """
    try:
        from huggingface_hub import scan_cache_dir
        
        cache_info = scan_cache_dir()
        
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                size_gb = repo.size_on_disk / (1024**3)
                return True, f"{size_gb:.2f} GB"
        
        return False, "Not cached"
        
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Check all model variants."""
    models = {
        "Qwen/Qwen2-VL-2B-Instruct": "2B Model (FP16 or 4-bit bitsandbytes)",
    }
    
    logger.info("=" * 80)
    logger.info("MODEL CACHE STATUS")
    logger.info("=" * 80)
    
    all_cached = True
    
    for model_name, description in models.items():
        is_cached, info = check_model_cached(model_name)
        
        if is_cached:
            status = f"✓ Cached ({info})"
        else:
            status = "✗ Not downloaded"
            all_cached = False
        
        logger.info(f"{description:30s}: {status}")
    
    logger.info("=" * 80)
    
    if all_cached:
        logger.info("✓ All models are cached and ready to use!")
    else:
        logger.info("✗ Some models need to be downloaded. Run:")
        logger.info("  python scripts/download_models.py")
    
    return 0 if all_cached else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
