"""Main entry point for Diagram2Table application."""

import argparse
import logging
import os
import sys
from typing import Optional

from src.config import get_settings, init_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_api(host: str, port: int, workers: int = 1) -> None:
    """Run the FastAPI server.

    Args:
        host: Host to bind to.
        port: Port to use.
        workers: Number of worker processes.
    """
    import uvicorn

    logger.info(f"Starting API server on {host}:{port}")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


def run_ui(host: str, port: int, share: bool = False) -> None:
    """Run the Gradio UI.

    Args:
        host: Host to bind to.
        port: Port to use.
        share: Create public link.
    """
    from src.ui.gradio_app import launch_ui

    logger.info(f"Starting Gradio UI on {host}:{port}")
    launch_ui(host=host, port=port, share=share)


def run_both(api_host: str, api_port: int, ui_host: str, ui_port: int) -> None:
    """Run both API and UI (for development/production).
    
    API loads the model, Gradio calls API via HTTP (no duplicate model loading).

    Args:
        api_host: API host.
        api_port: API port.
        ui_host: UI host.
        ui_port: UI port.
    """
    import signal
    import threading
    import time
    from src.ui.api_client import APIClient
    from src.config import get_settings

    settings = get_settings()
    
    # Configure API URL for Gradio to use
    api_url = f"http://localhost:{api_port}"
    
    # Flag to track shutdown
    shutdown_event = threading.Event()

    # Start API in background thread
    def start_api():
        import uvicorn
        config = uvicorn.Config(
            "src.api.app:app",
            host=api_host,
            port=api_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        
        # Store server instance for graceful shutdown
        start_api.server = server
        
        server.run()

    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, cleaning up...")
        shutdown_event.set()
        
        # Stop API server if it's running
        if hasattr(start_api, 'server'):
            logger.info("Shutting down API server...")
            start_api.server.should_exit = True
        
        # Clear CUDA cache immediately
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared CUDA cache on shutdown")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA on shutdown: {e}")
        
        # Force exit
        import sys
        logger.info("Forcing exit...")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    # Wait for API to be ready (increased timeout for model loading)
    # Model loading can take 15-30 seconds on GPU, 30-60 seconds on CPU
    logger.info(f"Waiting for API to start on {api_url}...")
    client = APIClient(base_url=api_url)
    
    # Increased from 60 to 180 retries (3 minutes) with 1 second delay = 3 minutes total
    max_retries = 180
    if not client.wait_for_api(max_retries=max_retries, delay=1.0):
        logger.error(f"API failed to start within {max_retries} seconds")
        raise RuntimeError("API failed to start")
    
    logger.info(f"API running on http://{api_host}:{api_port}")
    logger.info(f"UI running on http://{ui_host}:{ui_port}")
    logger.info("Gradio will use API client mode (single model instance)")

    # Run UI in main thread - use API client mode
    from src.ui.gradio_app import launch_ui
    launch_ui(
        analyzer=client,  # Pass the API client directly
        host=ui_host, 
        port=ui_port,
        use_api=True,  # Explicitly use API mode
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diagram2Table - Automatic diagram recognition and extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run API server
  python -m src.main api

  # Run Gradio UI
  python -m src.main ui

  # Run both API and UI (default)
  python -m src.main both

  # Run with 4-bit quantization (recommended for GPUs with 4-8GB VRAM)
  python -m src.main both --deployment-mode vlm_quantized

  # Run with full precision (requires 6GB+ VRAM)
  python -m src.main both --deployment-mode vlm_only

  # Run with mock VLM for testing (no GPU required)
  python -m src.main both --deployment-mode mock

  # Run with custom ports
  python -m src.main api --port 9000
  python -m src.main ui --port 7861 --share
        """,
    )

    parser.add_argument(
        "mode",
        choices=["api", "ui", "both"],
        default="both",
        nargs="?",
        help="Run mode: api, ui, or both (default: both)",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    parser.add_argument( # TODO: Этот кусок кода нам вообще нужен? Если мы на для API и UI юзаем разные порты
        "--port",
        type=int,
        default=None,
        help="Port to use (default: 8000 for API, 7860 for UI)",
    )

    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API port when running both (default: 8000)",
    )

    parser.add_argument(
        "--ui-port",
        type=int,
        default=7860,
        help="UI port when running both (default: 7860)",
    )

    parser.add_argument(#TODO: К сожалению, будь у нас больше вычислительных ресурсов, мы могли бы юзать несколько воркеров
        "--workers",
        type=int,
        default=1,
        help="Number of API workers (default: 1)",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link",
    )

    parser.add_argument(
        "--deployment-mode",
        choices=["vlm_only", "vlm_quantized", "mock"],
        default=None,
        help="Deployment mode: vlm_only (FP16, ~5-6GB VRAM), vlm_quantized (4-bit, ~2-2.5GB VRAM), mock (testing)",
    )

    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock VLM for testing (deprecated, use --deployment-mode mock)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Set deployment mode from command line arguments
    if args.deployment_mode:
        os.environ["DEPLOYMENT_MODE"] = args.deployment_mode
        logger.info(f"Deployment mode set to: {args.deployment_mode}")
    elif args.use_mock:
        os.environ["DEPLOYMENT_MODE"] = "mock"
        logger.warning("--use-mock is deprecated, use --deployment-mode mock instead")

    # Initialize settings (will read DEPLOYMENT_MODE from environment)
    settings = get_settings()

    logger.info("=" * 50)
    logger.info("Diagram2Table v1.0.0")
    logger.info("=" * 50)
    logger.info(f"Deployment mode: {settings.deployment_mode.value}")
    logger.info(f"Model: {settings.get_model_name()}")
    logger.info(f"Device: {settings.model_device}")
    logger.info("=" * 50)

    try:
        if args.mode == "api":
            port = args.port or 8000
            run_api(args.host, port, args.workers)

        elif args.mode == "ui":
            port = args.port or 7860
            run_ui(args.host, port, args.share)

        elif args.mode == "both":
            run_both(args.host, args.api_port, args.host, args.ui_port)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
