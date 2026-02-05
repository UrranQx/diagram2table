"""Health check endpoints."""

import time
from typing import Optional

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.api.app import get_analyzer_service

router = APIRouter()

# Track startup time
_start_time = time.time()


@router.get("/health")
async def health_check():
    """Basic health check endpoint.

    Returns service status and basic information.
    """
    service = get_analyzer_service()

    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    return {
        "status": "healthy" if service.is_ready else "degraded",
        "version": "1.0.0",
        "model_loaded": service.is_ready,
        "gpu_available": gpu_available,
        "uptime_seconds": int(time.time() - _start_time),
    }


@router.get("/health/ready")
async def readiness_probe():
    """Readiness probe for Kubernetes/Docker health checks.

    Returns 200 if service is ready to accept requests, 503 otherwise.
    """
    import logging
    logger = logging.getLogger(__name__)
    service = get_analyzer_service()
    logger.debug(f"Health check got service id={id(service)}, ready={service.is_ready}")

    if service.is_ready:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready"},
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "message": "Model is loading"},
        )


@router.get("/health/live")
async def liveness_probe():
    """Liveness probe for Kubernetes/Docker health checks.

    Always returns 200 if the server is running.
    """
    return {"status": "alive"}
