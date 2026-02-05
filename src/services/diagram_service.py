"""Main diagram service using the new ML pipeline.

This service wraps the colleague's pipeline from src/pipeline/ and provides
a clean interface for the API and UI layers.
"""

import asyncio
import io
import logging
import tempfile
import time
from typing import Any, Dict, Optional, Union

from PIL import Image

from src.config import get_settings

logger = logging.getLogger(__name__)


class DiagramService:
    """Main service for diagram analysis, evaluation, and rendering.

    Uses the ML pipeline from src/pipeline/ for:
    - Image -> text extraction with tiling and early stopping
    - Quality evaluation (judge) with VLM
    - Text -> diagram rendering with matplotlib
    """

    def __init__(self):
        """Initialize the service (models not loaded yet)."""
        self._model = None
        self._processor = None
        self._ready = False
        self._model_id: Optional[str] = None
        
        # GPU semaphore for sequential inference
        self._gpu_semaphore: Optional[asyncio.Semaphore] = None

    def load(self) -> None:
        """Load the VLM model and processor."""
        if self._ready:
            logger.info("DiagramService already loaded")
            return

        logger.info("Loading DiagramService...")
        
        from src.pipeline.vlm_engine import load_qwen25_vl, MODEL_ID
        
        settings = get_settings()
        
        # Check if we should use mock mode
        if settings.deployment_mode.value == "mock":
            logger.info("Mock mode enabled - skipping model loading")
            self._model = None
            self._processor = None
            self._model_id = "mock"
            self._ready = True
            return
        
        # Load actual model
        model_id = settings.qwen_model_id or MODEL_ID
        logger.info(f"Loading model: {model_id}")
        
        self._model, self._processor = load_qwen25_vl(model_id)
        self._model_id = model_id
        self._ready = True
        
        # Initialize GPU semaphore for async operations
        self._gpu_semaphore = asyncio.Semaphore(1)
        
        logger.info("DiagramService ready")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._ready = False
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("DiagramService unloaded")

    @property
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self._ready

    def _ensure_ready(self) -> None:
        """Ensure service is loaded before use."""
        if not self._ready:
            self.load()

    def analyze(
        self,
        image: Union[Image.Image, str, bytes],
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze a diagram image and extract text.

        Args:
            image: PIL Image, file path, or bytes
            **kwargs: Additional parameters passed to analyze_diagram

        Returns:
            Dictionary with:
                - raw_text: Extracted text (one line per block)
                - table: Markdown table
                - meta: Processing metadata
                - processing_time_ms: Total processing time
        """
        self._ensure_ready()
        
        start_time = time.time()
        
        # Handle different input types
        pil_image = self._to_pil_image(image)
        
        # Check for mock mode
        if self._model_id == "mock":
            result = self._mock_analyze(pil_image)
        else:
            result = self._real_analyze(pil_image, **kwargs)
        
        # Calculate processing time
        elapsed_ms = (time.time() - start_time) * 1000
        result["processing_time_ms"] = round(elapsed_ms, 1)
        
        return result

    def _to_pil_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """Convert various input types to PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _real_analyze(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run actual VLM analysis."""
        from src.pipeline.diagram_pipeline import (
            analyze_diagram,
            raw_text_to_md_table,
            choose_top_k_tiles,
        )
        
        # Determine optimal tile count
        top_k = kwargs.pop("top_k_tiles", None)
        if top_k is None:
            top_k = choose_top_k_tiles(image)
        
        # Run analysis
        result = analyze_diagram(
            image,
            self._model,
            self._processor,
            top_k_tiles=top_k,
            **kwargs,
        )
        
        # Generate markdown table
        raw_text = result.get("raw_text", "")
        table = raw_text_to_md_table(raw_text)
        
        return {
            "raw_text": raw_text,
            "lines": result.get("lines", []),
            "table": table,
            "meta": result.get("meta", {}),
        }

    def _mock_analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Return mock results for testing."""
        mock_text = "Получить запрос\nПроверить данные\nОбработать запрос\nОтправить ответ"
        from src.pipeline.diagram_pipeline import raw_text_to_md_table
        
        return {
            "raw_text": mock_text,
            "lines": mock_text.split("\n"),
            "table": raw_text_to_md_table(mock_text),
            "meta": {
                "mode": "mock",
                "image_size": image.size,
            },
        }

    def analyze_from_bytes(
        self,
        image_bytes: bytes,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze diagram from bytes.
        
        Args:
            image_bytes: Image data as bytes
            **kwargs: Additional parameters passed to analyze
        
        Returns:
            Dictionary with analysis results compatible with API response
        """
        result = self.analyze(image=image_bytes, **kwargs)
        
        # Return full result with both 'ir' and 'raw_text' for compatibility
        raw_text = result.get("raw_text", "")
        return {
            "ir": raw_text,  # API expects 'ir'
            "raw_text": raw_text,  # Client expects 'raw_text'
            "table": result.get("table", ""),
            "lines": result.get("lines", []),
            "meta": result.get("meta", {}),
            "processing_time_ms": result.get("processing_time_ms", 0),
        }

    def judge(
        self,
        image: Union[Image.Image, str, bytes],
        raw_text: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate recognition quality using VLM as judge.

        Args:
            image: Original diagram image
            raw_text: Recognized text to evaluate
            **kwargs: Additional parameters

        Returns:
            Dictionary with quality metrics:
                - text_coverage: 0-100
                - text_accuracy: 0-100
                - order_quality: 0-100
                - homoglyphs: 0-100
                - overall: 0-100
                - missing_examples: List of missing text examples
                - hallucinated_examples: List of hallucinated examples
                - notes: Judge's notes
        """
        self._ensure_ready()
        
        start_time = time.time()
        pil_image = self._to_pil_image(image)
        
        if self._model_id == "mock":
            result = self._mock_judge(raw_text)
        else:
            from src.pipeline.diagram_pipeline import judge_recognition_quality
            result = judge_recognition_quality(
                pil_image,
                raw_text,
                self._model,
                self._processor,
                **kwargs,
            )
        
        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)
        return result

    def _mock_judge(self, raw_text: str) -> Dict[str, Any]:
        """Return mock judge results for testing."""
        from src.pipeline.diagram_pipeline import homoglyphs_score
        
        return {
            "text_coverage": 85,
            "text_accuracy": 90,
            "order_quality": 80,
            "homoglyphs": homoglyphs_score(raw_text),
            "overall": 85,
            "missing_examples": [],
            "hallucinated_examples": [],
            "notes": "Mock evaluation",
        }

    def render(
        self,
        algorithm_text: str,
        use_llm: bool = False,
        dpi: int = 180,
    ) -> bytes:
        """Render algorithm text as a flowchart PNG.

        Args:
            algorithm_text: Text description of algorithm (one step per line)
            use_llm: Whether to use LLM for Mermaid generation (not implemented yet)
            dpi: DPI for output image

        Returns:
            PNG image as bytes
        """
        from src.pipeline.flowchart_render import algorithm_text_to_diagram_png_no_llm
        
        if not algorithm_text or not algorithm_text.strip():
            raise ValueError("Текст алгоритма не может быть пустым")
        
        # Currently only no-LLM mode is supported
        return algorithm_text_to_diagram_png_no_llm(algorithm_text, dpi=dpi)

    def render_from_mermaid(self, mermaid_code: str, dpi: int = 180) -> bytes:
        """Render Mermaid flowchart code to PNG.

        Args:
            mermaid_code: Mermaid flowchart code
            dpi: DPI for output image

        Returns:
            PNG image as bytes
        """
        from src.pipeline.flowchart_render import render_flowchart_to_png
        
        if not mermaid_code or not mermaid_code.strip():
            raise ValueError("Mermaid код не может быть пустым")
        
        return render_flowchart_to_png(mermaid_code, dpi=dpi)

    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        import torch
        
        info = {
            "service": "DiagramService",
            "ready": self._ready,
            "model_id": self._model_id,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        
        return info

    async def analyze_async(
        self,
        image: Union[Image.Image, str, bytes],
        **kwargs,
    ) -> Dict[str, Any]:
        """Async version of analyze with GPU semaphore."""
        self._ensure_ready()
        
        if self._gpu_semaphore is None:
            self._gpu_semaphore = asyncio.Semaphore(1)
        
        async with self._gpu_semaphore:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.analyze(image, **kwargs)
            )

    async def judge_async(
        self,
        image: Union[Image.Image, str, bytes],
        raw_text: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Async version of judge with GPU semaphore."""
        self._ensure_ready()
        
        if self._gpu_semaphore is None:
            self._gpu_semaphore = asyncio.Semaphore(1)
        
        async with self._gpu_semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.judge(image, raw_text, **kwargs)
            )
