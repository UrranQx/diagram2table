"""HTTP client for Gradio to communicate with FastAPI backend."""

import io
import logging
from typing import Any, Dict, Optional

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class APIClient:
    """HTTP client for FastAPI backend.
    
    Used by Gradio UI to call the FastAPI backend instead of loading
    the ML model directly. This allows running Gradio as a lightweight
    frontend while the heavy model runs in a single FastAPI instance.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 600.0):
        """Initialize API client.
        
        Args:
            base_url: Base URL of the FastAPI backend.
            timeout: Request timeout in seconds (default 600s = 10 minutes).
        """
        self.base_url = base_url.replace("localhost", "127.0.0.1").rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                trust_env=False  # Don't use system proxy settings
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        """Destructor to ensure client is closed."""
        self.close()

    def health_check(self) -> bool:
        """Check if the API is ready."""
        try:
            response = self._get_client().get(f"{self.base_url}/api/v1/health/ready")
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def wait_for_api(self, max_retries: int = 30, delay: float = 1.0) -> bool:
        """Wait for API to become ready."""
        import time
        
        for i in range(max_retries):
            if self.health_check():
                logger.info(f"API ready after {i + 1} attempts")
                return True
            logger.debug(f"Waiting for API... attempt {i + 1}/{max_retries}")
            time.sleep(delay)
        
        logger.error(f"API not ready after {max_retries} attempts")
        return False

    def _image_to_bytes(self, image: Image.Image) -> tuple:
        """Convert PIL image to bytes for uploading."""
        img_buffer = io.BytesIO()
        image_format = image.format or "PNG"
        image.save(img_buffer, format=image_format)
        img_buffer.seek(0)
        return (
            f"image.{image_format.lower()}",
            img_buffer,
            f"image/{image_format.lower()}"
        )

    def analyze(
        self,
        image: Image.Image,
    ) -> Dict[str, Any]:
        """Analyze a diagram image via API.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            Analysis result dict with 'raw_text', 'table', 'lines', 'meta', 'processing_time_ms'.
        """
        files = {"image": self._image_to_bytes(image)}
        
        try:
            response = self._get_client().post(
                f"{self.base_url}/api/v1/diagram/analyze",
                files=files,
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "raw_text": result.get("raw_text", ""),
                "table": result.get("table", ""),
                "lines": result.get("lines", []),
                "meta": result.get("meta", {}),
                "processing_time_ms": result.get("processing_time_ms", 0),
            }
            
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error(e)
            logger.error(f"API error: {error_detail}")
            raise ValueError(f"API error: {error_detail}") from e
            
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}")
            raise ValueError(f"Failed to connect to API: {e}") from e

    def judge(
        self,
        image: Image.Image,
        raw_text: str,
    ) -> Dict[str, Any]:
        """Evaluate recognition quality via API.
        
        Args:
            image: Original diagram image.
            raw_text: Recognized text to evaluate.
            
        Returns:
            Quality metrics dict.
        """
        files = {"image": self._image_to_bytes(image)}
        data = {"raw_text": raw_text}
        
        try:
            response = self._get_client().post(
                f"{self.base_url}/api/v1/diagram/judge",
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error(e)
            logger.error(f"Judge API error: {error_detail}")
            raise ValueError(f"Judge API error: {error_detail}") from e
            
        except httpx.RequestError as e:
            logger.error(f"Judge request failed: {e}")
            raise ValueError(f"Failed to connect to API: {e}") from e

    def render(
        self,
        algorithm_text: str,
        dpi: int = 180,
    ) -> bytes:
        """Render algorithm text as flowchart PNG.
        
        Args:
            algorithm_text: Algorithm text (one step per line).
            dpi: Output image DPI.
            
        Returns:
            PNG image bytes.
        """
        data = {"algorithm_text": algorithm_text, "dpi": dpi}
        
        try:
            response = self._get_client().post(
                f"{self.base_url}/api/v1/diagram/render",
                data=data,
            )
            response.raise_for_status()
            return response.content
            
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error(e)
            logger.error(f"Render API error: {error_detail}")
            raise ValueError(f"Render API error: {error_detail}") from e
            
        except httpx.RequestError as e:
            logger.error(f"Render request failed: {e}")
            raise ValueError(f"Failed to connect to API: {e}") from e

    def render_mermaid(
        self,
        mermaid_code: str,
        dpi: int = 180,
    ) -> bytes:
        """Render Mermaid code as PNG.
        
        Args:
            mermaid_code: Mermaid flowchart code.
            dpi: Output image DPI.
            
        Returns:
            PNG image bytes.
        """
        data = {"mermaid_code": mermaid_code, "dpi": dpi}
        
        try:
            response = self._get_client().post(
                f"{self.base_url}/api/v1/diagram/render-mermaid",
                data=data,
            )
            response.raise_for_status()
            return response.content
            
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error(e)
            logger.error(f"Mermaid render API error: {error_detail}")
            raise ValueError(f"Mermaid render API error: {error_detail}") from e
            
        except httpx.RequestError as e:
            logger.error(f"Mermaid render request failed: {e}")
            raise ValueError(f"Failed to connect to API: {e}") from e

    def get_info(self) -> Dict[str, Any]:
        """Get service information from API."""
        try:
            response = self._get_client().get(f"{self.base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.warning(f"Failed to get service info: {e}")
            return {
                "status": "unknown",
                "mode": "api_client",
                "api_url": self.base_url,
                "error": str(e),
            }

    def _extract_error(self, e: httpx.HTTPStatusError) -> str:
        """Extract error message from HTTP error response."""
        try:
            error_json = e.response.json()
            detail = error_json.get("detail", {})
            if isinstance(detail, dict):
                return detail.get("message", str(e))
            return str(detail)
        except Exception:
            return e.response.text or str(e)
