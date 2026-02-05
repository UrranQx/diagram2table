"""
Test script for DeepSeek-OCR-2 model evaluation.

This script tests the DeepSeek-OCR-2 model for diagram analysis capabilities.
It evaluates:
- Model loading and VRAM usage
- Inference time on sample images
- Output format and quality for diagram understanding
- Comparison with Qwen2-VL-2B-Instruct

IMPORTANT: This model is primarily designed for OCR and document markdown conversion,
NOT specifically for diagram understanding. Use cautiously.

Usage:
    python scripts/test_deepseek_ocr.py --image path/to/diagram.jpg
    python scripts/test_deepseek_ocr.py --test-vram-only
"""

"""
Краткие выводы:
❌ DeepSeek-OCR-2 НЕ подходит для вашего проекта
Основные проблемы:

🔴 VRAM - Требует ~8-9 ГБ, у вас только 6 ГБ (GTX 1660 Ti)
🔴 Неправильный тип модели - Это OCR для документов, а не анализатор диаграмм
🔴 Выход - Markdown текст, а вам нужен структурированный DiagramIR
🔴 Конфликты - Требует Flash Attention 2 и transformers 4.46.3
Сравнение:

Qwen2-VL-2B: 4.43 ГБ, работает на вашем GPU, понимает диаграммы ✅
DeepSeek-OCR-2: 6.78 ГБ, НЕ влезет в память, только извлекает текст ❌
✅ Рекомендация: Оставить Qwen2-VL-2B
Если хотите улучшить качество:

Включите INT4 квантизацию (освободит 3-4 ГБ VRAM)
Оптимизируйте промпты для лучшего структурированного вывода
Рассмотрите Qwen2-VL-7B-AWQ (может влезть в 6 ГБ)
Тестовый скрипт создан в test_deepseek_ocr.py, но запустить его не получится из-за нехватки VRAM.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepSeekOCR2Tester:
    """Test harness for DeepSeek-OCR-2 model."""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR-2"):
        """Initialize tester.
        
        Args:
            model_name: HuggingFace model name or local path.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.torch = None
        
    def check_requirements(self) -> Dict[str, Any]:
        """Check system requirements and dependencies.
        
        Returns:
            Dict with system info and compatibility checks.
        """
        logger.info("Checking system requirements...")
        results = {
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "gpu_memory_gb": None,
            "transformers_available": False,
            "transformers_version": None,
            "flash_attn_available": False,
        }
        
        # Check PyTorch
        try:
            import torch
            self.torch = torch
            results["torch_available"] = True
            results["torch_version"] = torch.__version__
            results["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                results["cuda_version"] = torch.version.cuda
                results["gpu_name"] = torch.cuda.get_device_name(0)
                results["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {results['gpu_name']} ({results['gpu_memory_gb']:.2f} GB)")
        except ImportError:
            logger.error("PyTorch not installed!")
            
        # Check Transformers
        try:
            import transformers
            results["transformers_available"] = True
            results["transformers_version"] = transformers.__version__
            logger.info(f"Transformers version: {transformers.__version__}")
        except ImportError:
            logger.error("Transformers not installed!")
            
        # Check Flash Attention (required for DeepSeek-OCR-2)
        try:
            import flash_attn
            results["flash_attn_available"] = True
            logger.info("Flash Attention 2 is available")
        except ImportError:
            logger.warning("Flash Attention 2 not installed - required for DeepSeek-OCR-2!")
            logger.warning("Install with: pip install flash-attn==2.7.3 --no-build-isolation")
            
        return results
        
    def load_model(self, use_fp16: bool = True) -> Dict[str, Any]:
        """Load DeepSeek-OCR-2 model and measure VRAM usage.
        
        Args:
            use_fp16: Whether to use FP16 (if False, uses bfloat16).
            
        Returns:
            Dict with loading stats (time, VRAM, success).
        """
        logger.info(f"Loading DeepSeek-OCR-2 model: {self.model_name}")
        
        if not self.torch or not self.torch.cuda.is_available():
            logger.error("CUDA not available - model requires GPU!")
            return {"success": False, "error": "CUDA not available"}
            
        stats = {
            "success": False,
            "load_time_sec": None,
            "vram_before_mb": None,
            "vram_after_mb": None,
            "vram_used_mb": None,
            "error": None,
        }
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Measure initial VRAM
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats()
            stats["vram_before_mb"] = self.torch.cuda.memory_allocated() / 1e6
            
            start_time = time.time()
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with Flash Attention 2
            logger.info("Loading model (this may take a few minutes)...")
            dtype = self.torch.float16 if use_fp16 else self.torch.bfloat16
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                _attn_implementation='flash_attention_2',  # Required for DeepSeek-OCR-2
                trust_remote_code=True,
                use_safetensors=True,
            )
            
            # Move to GPU and set dtype
            self.model = self.model.eval().cuda().to(dtype)
            
            end_time = time.time()
            
            # Measure final VRAM
            self.torch.cuda.synchronize()
            stats["vram_after_mb"] = self.torch.cuda.memory_allocated() / 1e6
            stats["vram_used_mb"] = stats["vram_after_mb"] - stats["vram_before_mb"]
            stats["load_time_sec"] = end_time - start_time
            stats["success"] = True
            
            logger.info(f"Model loaded successfully in {stats['load_time_sec']:.2f}s")
            logger.info(f"VRAM used: {stats['vram_used_mb']:.2f} MB ({stats['vram_used_mb']/1024:.2f} GB)")
            
        except Exception as e:
            stats["error"] = str(e)
            logger.error(f"Failed to load model: {e}")
            
        return stats
        
    def infer_diagram(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference on a diagram image.
        
        Args:
            image_path: Path to input image.
            prompt: Custom prompt (default: markdown conversion).
            output_path: Directory to save results.
            
        Returns:
            Dict with inference results (output, time, VRAM).
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded! Call load_model() first.")
            return {"success": False, "error": "Model not loaded"}
            
        logger.info(f"Running inference on: {image_path}")
        
        # Default prompt for document/diagram conversion
        if prompt is None:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
            
        stats = {
            "success": False,
            "inference_time_sec": None,
            "output_text": None,
            "output_length": None,
            "vram_peak_mb": None,
            "error": None,
        }
        
        try:
            # Reset memory stats
            self.torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            # Run inference using model's built-in infer method
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_path or "./output",
                base_size=1024,    # Base resolution
                image_size=768,    # Tile resolution
                crop_mode=True,    # Enable dynamic tiling
                save_results=True if output_path else False,
            )
            
            end_time = time.time()
            
            # Measure peak VRAM
            stats["vram_peak_mb"] = self.torch.cuda.max_memory_allocated() / 1e6
            stats["inference_time_sec"] = end_time - start_time
            stats["output_text"] = result
            stats["output_length"] = len(result) if result else 0
            stats["success"] = True
            
            logger.info(f"Inference completed in {stats['inference_time_sec']:.2f}s")
            logger.info(f"Output length: {stats['output_length']} chars")
            logger.info(f"Peak VRAM: {stats['vram_peak_mb']:.2f} MB ({stats['vram_peak_mb']/1024:.2f} GB)")
            
        except Exception as e:
            stats["error"] = str(e)
            logger.error(f"Inference failed: {e}")
            
        return stats
        
    def analyze_output_for_diagrams(self, output_text: str) -> Dict[str, Any]:
        """Analyze if output is suitable for diagram understanding.
        
        DeepSeek-OCR-2 is designed for OCR/markdown conversion, not structured
        diagram analysis. This method checks if output contains:
        - Structured elements (nodes, edges)
        - Diagram-specific terminology
        - Relationships between elements
        
        Args:
            output_text: Model output text.
            
        Returns:
            Dict with analysis results.
        """
        analysis = {
            "is_markdown": False,
            "has_structure": False,
            "has_relationships": False,
            "has_diagram_terms": False,
            "suitable_for_diagrams": False,
            "notes": [],
        }
        
        if not output_text:
            analysis["notes"].append("Empty output")
            return analysis
            
        # Check for markdown
        if any(marker in output_text for marker in ["#", "**", "-", "*", "|"]):
            analysis["is_markdown"] = True
            analysis["notes"].append("Output is in Markdown format")
            
        # Check for structured elements
        structure_keywords = ["node", "element", "shape", "box", "rectangle", "circle"]
        if any(kw in output_text.lower() for kw in structure_keywords):
            analysis["has_structure"] = True
            analysis["notes"].append("Contains structural elements")
            
        # Check for relationships
        relationship_keywords = ["connect", "link", "arrow", "flow", "point", "->", "→"]
        if any(kw in output_text.lower() for kw in relationship_keywords):
            analysis["has_relationships"] = True
            analysis["notes"].append("Contains relationship indicators")
            
        # Check for diagram terminology
        diagram_keywords = ["diagram", "flowchart", "bpmn", "process", "workflow", "activity"]
        if any(kw in output_text.lower() for kw in diagram_keywords):
            analysis["has_diagram_terms"] = True
            analysis["notes"].append("Contains diagram terminology")
            
        # Overall assessment
        if analysis["has_structure"] and analysis["has_relationships"]:
            analysis["suitable_for_diagrams"] = True
            analysis["notes"].append("✓ May be suitable for diagram analysis")
        else:
            analysis["suitable_for_diagrams"] = False
            analysis["notes"].append("✗ Likely just OCR text, not diagram understanding")
            
        return analysis
        
    def cleanup(self) -> None:
        """Clean up model and free VRAM."""
        logger.info("Cleaning up...")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if self.torch and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            logger.info("VRAM cleared")


def main():
    """Main test routine."""
    parser = argparse.ArgumentParser(description="Test DeepSeek-OCR-2 for diagram analysis")
    parser.add_argument("--image", type=str, help="Path to diagram image")
    parser.add_argument("--test-vram-only", action="store_true", help="Only test VRAM usage")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--prompt", type=str, help="Custom prompt")
    
    args = parser.parse_args()
    
    tester = DeepSeekOCR2Tester()
    
    # Step 1: Check requirements
    print("\n" + "="*80)
    print("STEP 1: Checking Requirements")
    print("="*80)
    req_results = tester.check_requirements()
    
    for key, value in req_results.items():
        print(f"  {key}: {value}")
        
    if not req_results["torch_available"] or not req_results["cuda_available"]:
        logger.error("CUDA not available - cannot proceed!")
        return
        
    if not req_results["flash_attn_available"]:
        logger.error("Flash Attention 2 not available - required for DeepSeek-OCR-2!")
        logger.error("Install with: pip install flash-attn==2.7.3 --no-build-isolation")
        return
        
    # Check VRAM
    gpu_memory_gb = req_results["gpu_memory_gb"]
    if gpu_memory_gb and gpu_memory_gb < 6.0:
        logger.warning(f"GPU has only {gpu_memory_gb:.2f} GB VRAM - model requires ~6-8 GB!")
        
    # Step 2: Load model
    print("\n" + "="*80)
    print("STEP 2: Loading Model")
    print("="*80)
    load_stats = tester.load_model(use_fp16=True)
    
    for key, value in load_stats.items():
        print(f"  {key}: {value}")
        
    if not load_stats["success"]:
        logger.error("Failed to load model!")
        return
        
    # Check if VRAM usage exceeds available memory
    if load_stats["vram_used_mb"] and gpu_memory_gb:
        vram_used_gb = load_stats["vram_used_mb"] / 1024
        if vram_used_gb > gpu_memory_gb * 0.9:
            logger.warning(f"VRAM usage ({vram_used_gb:.2f} GB) near limit ({gpu_memory_gb:.2f} GB)!")
            
    if args.test_vram_only:
        print("\n" + "="*80)
        print("VRAM TEST COMPLETE")
        print("="*80)
        tester.cleanup()
        return
        
    # Step 3: Run inference (if image provided)
    if args.image:
        print("\n" + "="*80)
        print("STEP 3: Running Inference")
        print("="*80)
        
        infer_stats = tester.infer_diagram(
            image_path=args.image,
            prompt=args.prompt,
            output_path=args.output,
        )
        
        for key, value in infer_stats.items():
            if key != "output_text":  # Don't print full output
                print(f"  {key}: {value}")
                
        if infer_stats["success"] and infer_stats["output_text"]:
            # Print first 500 chars of output
            print("\n--- OUTPUT (first 500 chars) ---")
            print(infer_stats["output_text"][:500])
            if len(infer_stats["output_text"]) > 500:
                print("... (truncated)")
                
            # Analyze output for diagram suitability
            print("\n" + "="*80)
            print("STEP 4: Analyzing Output for Diagram Suitability")
            print("="*80)
            analysis = tester.analyze_output_for_diagrams(infer_stats["output_text"])
            
            for key, value in analysis.items():
                print(f"  {key}: {value}")
    else:
        print("\nNo image provided - skipping inference test")
        print("Usage: python scripts/test_deepseek_ocr.py --image path/to/diagram.jpg")
        
    # Cleanup
    tester.cleanup()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
