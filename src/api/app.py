"""FastAPI application."""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.services.diagram_service import DiagramService

logger = logging.getLogger(__name__)

# Global service instance with thread safety
_diagram_service: Optional[DiagramService] = None
_service_lock = threading.Lock()

# GPU processing queue - only one request at a time
_gpu_semaphore: Optional[asyncio.Semaphore] = None


def get_diagram_service() -> DiagramService:
    """Get the global diagram service instance (thread-safe singleton)."""
    global _diagram_service
    
    # Double-checked locking pattern
    if _diagram_service is None:
        with _service_lock:
            if _diagram_service is None:
                logger.info("Creating new DiagramService instance")
                _diagram_service = DiagramService()
    else:
        logger.debug(f"Returning existing service (ready={_diagram_service.is_ready})")
    return _diagram_service


# Legacy alias for backward compatibility
def get_analyzer_service() -> DiagramService:
    """Legacy alias for get_diagram_service."""
    return get_diagram_service()


def get_gpu_semaphore() -> asyncio.Semaphore:
    """Get GPU processing semaphore (max 1 concurrent request)."""
    global _gpu_semaphore
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(1)
        logger.info("Created GPU semaphore with capacity=1")
    return _gpu_semaphore


def set_diagram_service(service: DiagramService) -> None:
    """Set the global diagram service instance."""
    global _diagram_service
    _diagram_service = service


# Legacy alias
def set_analyzer_service(service: DiagramService) -> None:
    """Legacy alias for set_diagram_service."""
    set_diagram_service(service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Diagram2Table API...")
    service = get_diagram_service()
    logger.info(f"Lifespan got service with id={id(service)}, ready={service.is_ready}")

    try:
        service.load()
        logger.info(f"Service loaded successfully - id={id(service)}, ready={service.is_ready}")
    except Exception as e:
        logger.error(f"Failed to load service: {e}")
        # Continue with degraded service

    yield

    # Shutdown
    logger.info("Shutting down Diagram2Table API...")
    
    # Clear CUDA cache to release GPU memory immediately
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            logger.info("Cleared CUDA cache on shutdown")
    except Exception as e:
        logger.warning(f"Failed to clear CUDA cache on shutdown: {e}")
    
    # Unload service
    if service.is_ready:
        service.unload()
    
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title="Diagram2Table API",
        description="API для распознавания диаграмм и извлечения структурированных описаний алгоритмов",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from src.api.routes import health
    from src.api.routes import diagram

    app.include_router(health.router, prefix="/api/v1", tags=["System"])
    app.include_router(diagram.router, prefix="/api/v1", tags=["Diagram Analysis"])

    return app


# Default app instance - uses deployment_mode from settings
app = create_app()
