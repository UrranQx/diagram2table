"""Diagram analysis endpoints."""

import asyncio
import logging
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel

from src.api.app import get_analyzer_service
from src.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalyzeResponse(BaseModel):
    """Response model for diagram analysis."""

    task_id: str
    status: Literal["completed", "processing", "failed"]
    ir: Optional[str] = None  # Changed from dict to str - raw text extracted
    raw_text: Optional[str] = None  # Alias for ir for compatibility
    table: Optional[str] = None
    lines: Optional[list] = None
    meta: Optional[dict] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""

    code: str
    message: str
    details: Optional[dict] = None


class JudgeResponse(BaseModel):
    """Response model for quality evaluation."""

    text_coverage: int
    text_accuracy: int
    order_quality: int
    homoglyphs: int
    overall: int
    missing_examples: list[str]
    hallucinated_examples: list[str]
    notes: str
    processing_time_ms: float


@router.post(
    "/diagram/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def analyze_diagram(
    request: Request,
    image: UploadFile = File(..., description="Diagram image (PNG, JPG, WEBP, SVG)"),
):
    """Analyze a diagram image and extract structured information.

    - **image**: Diagram image file (PNG, JPG, WEBP, SVG supported)
    
    The service automatically:
    - Detects optimal tiling strategy based on image size
    - Extracts text from diagram blocks
    - Returns both structured IR and markdown table
    """
    settings = get_settings()
    task_id = str(uuid.uuid4())

    # Validate content type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_IMAGE_FORMAT",
                "message": f"Invalid content type: {image.content_type}. Expected image/*",
            },
        )

    # Read image
    try:
        image_bytes = await image.read()
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "IMAGE_READ_ERROR",
                "message": "Failed to read image file",
            },
        )

    # Validate file size (max 10 MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"Image size {len(image_bytes)} bytes exceeds maximum {MAX_FILE_SIZE} bytes (10 MB)",
            },
        )

    # Get services
    service = get_analyzer_service()
    from src.api.app import get_gpu_semaphore
    gpu_semaphore = get_gpu_semaphore()
    
    gpu_acquired = False

    # Acquire GPU lock - only one inference at a time
    try:
        await gpu_semaphore.acquire()
        gpu_acquired = True
        logger.info(f"Acquired GPU lock for task {task_id}")
        
        try:
            # Check if client disconnected before starting
            if await request.is_disconnected():
                logger.warning(f"Client disconnected before analysis for task {task_id}")
                raise HTTPException(
                    status_code=499,  # Client Closed Request (non-standard but widely used)
                    detail={"code": "CLIENT_DISCONNECTED", "message": "Client disconnected"},
                )
            
            # Create a task for monitoring client connection
            async def run_analysis_with_monitoring():
                """Run analysis with periodic client disconnect checks."""
                # Start the analysis in thread pool
                analysis_task = asyncio.create_task(
                    asyncio.to_thread(
                        service.analyze_from_bytes,
                        image_bytes=image_bytes,
                    )
                )
                
                # Monitor client connection
                check_interval = 2.0  # Check every 2 seconds
                while not analysis_task.done():
                    await asyncio.sleep(check_interval)
                    if await request.is_disconnected():
                        logger.warning(f"Client disconnected during analysis for task {task_id}")
                        # Cancel the async wrapper (but thread will continue)
                        analysis_task.cancel()
                        try:
                            await analysis_task
                        except asyncio.CancelledError:
                            pass
                        
                        # Clear GPU cache to free memory
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                logger.info(f"Cleared CUDA cache for cancelled task {task_id}")
                        except Exception as e:
                            logger.error(f"Failed to clear CUDA cache: {e}")
                        
                        raise HTTPException(
                            status_code=499,  # Client Closed Request
                            detail={"code": "CLIENT_DISCONNECTED", "message": "Client disconnected during processing"},
                        )
                
                return await analysis_task
            
            result = await run_analysis_with_monitoring()

            return AnalyzeResponse(
                task_id=task_id,
                status="completed",
                ir=result.get("ir"),
                raw_text=result.get("raw_text"),
                table=result.get("table"),
                lines=result.get("lines"),
                meta=result.get("meta"),
                processing_time_ms=result.get("processing_time_ms"),
            )
        except asyncio.CancelledError:
            logger.warning(f"Task {task_id} was cancelled")
            # Clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"Cleared CUDA cache for cancelled task {task_id}")
            except Exception:
                pass
            raise HTTPException(
                status_code=499,  # Client Closed Request
                detail={"code": "TASK_CANCELLED", "message": "Task cancelled due to client disconnect"},
            )
        finally:
            # Always release GPU lock
            if gpu_acquired:
                gpu_semaphore.release()
                logger.info(f"Released GPU lock for task {task_id}")
                gpu_acquired = False

    except ValueError as e:
        # Validation errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "MODEL_ERROR",
                "message": "Failed to analyze diagram",
                "details": {"error": str(e)},
            },
        )


@router.post(
    "/diagram/judge",
    response_model=JudgeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def judge_diagram(
    request: Request,
    image: UploadFile = File(..., description="Original diagram image"),
    raw_text: str = Form(..., description="Recognized text to evaluate"),
):
    """Evaluate the quality of diagram text recognition.

    - **image**: Original diagram image
    - **raw_text**: Recognized text to evaluate
    
    Returns quality metrics:
    - text_coverage: How much text was captured (0-100)
    - text_accuracy: How accurate the recognition was (0-100)
    - order_quality: How well the order was preserved (0-100)
    - homoglyphs: Homoglyph detection score (0-100)
    - overall: Overall quality score (0-100)
    """
    settings = get_settings()
    
    # Validate content type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_IMAGE_FORMAT",
                "message": f"Invalid content type: {image.content_type}. Expected image/*",
            },
        )
    
    # Read image
    try:
        image_bytes = await image.read()
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "IMAGE_READ_ERROR",
                "message": "Failed to read image file",
            },
        )
    
    # Validate file size (max 10 MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"Image size {len(image_bytes)} bytes exceeds maximum {MAX_FILE_SIZE} bytes (10 MB)",
            },
        )
    
    # Get service
    service = get_analyzer_service()
    from src.api.app import get_gpu_semaphore
    gpu_semaphore = get_gpu_semaphore()
    
    gpu_acquired = False
    
    try:
        await gpu_semaphore.acquire()
        gpu_acquired = True
        logger.info(f"Acquired GPU lock for judge evaluation")
        
        try:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.warning(f"Client disconnected before judge evaluation")
                raise HTTPException(
                    status_code=499,
                    detail={"code": "CLIENT_DISCONNECTED", "message": "Client disconnected"},
                )
            
            # Use service for image conversion (handles SVG)
            result = await asyncio.to_thread(
                service.judge,
                image=image_bytes,
                raw_text=raw_text,
            )
            
            return JudgeResponse(
                text_coverage=result.get("text_coverage", 0),
                text_accuracy=result.get("text_accuracy", 0),
                order_quality=result.get("order_quality", 0),
                homoglyphs=result.get("homoglyphs", 0),
                overall=result.get("overall", 0),
                missing_examples=result.get("missing_examples", []),
                hallucinated_examples=result.get("hallucinated_examples", []),
                notes=result.get("notes", ""),
                processing_time_ms=result.get("processing_time_ms", 0),
            )
            
        finally:
            if gpu_acquired:
                gpu_semaphore.release()
                logger.info(f"Released GPU lock for judge evaluation")
                gpu_acquired = False
                
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.exception(f"Judge evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "MODEL_ERROR",
                "message": "Failed to evaluate diagram quality",
                "details": {"error": str(e)},
            },
        )


@router.post(
    "/diagram/render",
    responses={
        200: {"content": {"image/png": {}}, "description": "Rendered flowchart PNG"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def render_diagram(
    request: Request,
    algorithm_text: str = Form(..., description="Algorithm text (one line per step)"),
    dpi: int = Form(default=180, description="Output image DPI"),
):
    """Render algorithm text as a flowchart PNG diagram.

    - **algorithm_text**: Algorithm description (one action per line)
    - **dpi**: DPI for output image (default 180)
    
    Returns PNG image of the flowchart.
    """
    try:
        # Validate input
        if not algorithm_text or not algorithm_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "Algorithm text cannot be empty",
                },
            )
        
        if dpi < 72 or dpi > 600:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "DPI must be between 72 and 600",
                },
            )
        
        # Check if client disconnected
        if await request.is_disconnected():
            logger.warning("Client disconnected before render")
            raise HTTPException(
                status_code=499,
                detail={"code": "CLIENT_DISCONNECTED", "message": "Client disconnected"},
            )
        
        # Get service
        service = get_analyzer_service()
        
        # Render in thread pool
        png_bytes = await asyncio.to_thread(
            service.render,
            algorithm_text=algorithm_text,
            dpi=dpi,
        )
        
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=flowchart.png"},
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.exception(f"Render failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "RENDER_ERROR",
                "message": "Failed to render diagram",
                "details": {"error": str(e)},
            },
        )


@router.post(
    "/diagram/render-mermaid",
    responses={
        200: {"content": {"image/png": {}}, "description": "Rendered flowchart PNG"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def render_mermaid_diagram(
    request: Request,
    mermaid_code: str = Form(..., description="Mermaid flowchart code"),
    dpi: int = Form(default=180, description="Output image DPI"),
):
    """Render Mermaid flowchart code as PNG diagram.

    - **mermaid_code**: Mermaid flowchart syntax
    - **dpi**: DPI for output image (default 180)
    
    Returns PNG image of the flowchart.
    """
    try:
        # Validate input
        if not mermaid_code or not mermaid_code.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "Mermaid code cannot be empty",
                },
            )
        
        if dpi < 72 or dpi > 600:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "DPI must be between 72 and 600",
                },
            )
        
        # Check if client disconnected
        if await request.is_disconnected():
            logger.warning("Client disconnected before mermaid render")
            raise HTTPException(
                status_code=499,
                detail={"code": "CLIENT_DISCONNECTED", "message": "Client disconnected"},
            )
        
        # Get service
        service = get_analyzer_service()
        
        # Render in thread pool
        png_bytes = await asyncio.to_thread(
            service.render_from_mermaid,
            mermaid_code=mermaid_code,
            dpi=dpi,
        )
        
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=flowchart.png"},
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.exception(f"Mermaid render failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "RENDER_ERROR",
                "message": "Failed to render mermaid diagram",
                "details": {"error": str(e)},
            },
        )


@router.get(
    "/diagram/{task_id}",
    response_model=AnalyzeResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
    },
)
async def get_task_result(task_id: str):
    """Get the result of an async analysis task.

    Currently, all analyses are synchronous, so this endpoint
    returns 404 for any task_id (for future async support).
    """
    # For MVP, all tasks are synchronous
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "TASK_NOT_FOUND",
            "message": f"Task {task_id} not found",
        },
    )
