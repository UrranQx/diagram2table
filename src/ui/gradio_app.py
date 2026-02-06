"""Gradio web interface for Diagram2Table with metrics and rendering."""

import io
import json as json_lib
import logging
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
from PIL import Image

from src.config import get_settings

logger = logging.getLogger(__name__)

# Type alias for backend
AnalyzerBackend = Any


def create_ui(
    analyzer: Optional[AnalyzerBackend] = None,
    share: bool = False,
    use_api: Optional[bool] = None,
) -> gr.Blocks:
    """Create Gradio interface for Diagram2Table.

    Args:
        analyzer: Backend instance (DiagramService or APIClient). 
                  If None, creates based on settings.
        share: Whether to create a public link.
        use_api: If True, use API client mode. If None, uses settings.gradio_use_api.

    Returns:
        Gradio Blocks application.
    """
    settings = get_settings()

    # Determine mode: API client or direct model
    if use_api is None:
        use_api = settings.gradio_use_api

    if analyzer is None:
        if use_api:
            from src.ui.api_client import APIClient
            analyzer = APIClient(base_url=settings.api_base_url)
            logger.info(f"Gradio using API client mode (backend: {settings.api_base_url})")
        else:
            from src.services.diagram_service import DiagramService
            analyzer = DiagramService()
            analyzer.load()
            logger.info("Gradio using direct model mode")

    is_api_mode = hasattr(analyzer, 'health_check')

    # ============== Analysis Functions ==============

    def analyze_image(
        image: Optional[Union[Image.Image, str]],
        preview_png: Optional[bytes],
        md_filename: str,
        txt_filename: str,
    ) -> Tuple[str, str, str, float, tuple, tuple]:
        """Analyze diagram and return results plus downloadable contents.

        - `image`: path or PIL image (from file upload)
        - `preview_png`: PNG bytes generated for preview (if any)
        - `md_filename` / `txt_filename`: filenames the user wants for downloads

        Returns:
            table, raw_text, time_str, time_ms, md_download_tuple, txt_download_tuple
        """
        if image is None:
            return "Ошибка: Пожалуйста, загрузите изображение диаграммы", "", "", 0.0, None, None

        try:
            # Convert for API client mode if needed
            if is_api_mode:
                if isinstance(image, str) and image.lower().endswith('.svg'):
                    if preview_png:
                        pil_image = Image.open(io.BytesIO(preview_png))
                    else:
                        # Fallback conversion
                        try:
                            from wand.image import Image as WandImage
                        except Exception as e:
                            logger.exception("Wand not available to convert SVG for API mode: %s", e)
                            raise RuntimeError("SVG support requires Wand and ImageMagick installed in the UI environment") from e

                        with open(image, 'rb') as f:
                            svg_bytes = f.read()
                        with WandImage(blob=svg_bytes, resolution=100) as wimg:
                            wimg.format = 'png'
                            png_bytes = wimg.make_blob('png')
                        pil_image = Image.open(io.BytesIO(png_bytes))
                elif isinstance(image, str):
                    pil_image = Image.open(image)
                else:
                    pil_image = image

                result = analyzer.analyze(image=pil_image)

            else:
                # Direct DiagramService: pass path or PIL - it will handle SVG conversion server-side
                result = analyzer.analyze(image=image)

            table = result.get("table", "")
            raw_text = result.get("raw_text", "")
            time_ms = result.get("processing_time_ms", 0)

            time_str = f"{time_ms:.1f} мс"

            # Prepare download payloads as file paths with user-specified names (safe)
            import os

            safe_md = os.path.basename(md_filename) or "diagram.md"
            if not safe_md.lower().endswith(".md"):
                safe_md = f"{safe_md}.md"

            safe_txt = os.path.basename(txt_filename) or "diagram.txt"
            if not safe_txt.lower().endswith(".txt"):
                safe_txt = f"{safe_txt}.txt"

            tmp_dir = tempfile.mkdtemp(prefix="d2t-")
            md_path = os.path.join(tmp_dir, safe_md)
            txt_path = os.path.join(tmp_dir, safe_txt)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(table)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw_text)

            md_download = md_path
            txt_download = txt_path

            return table, raw_text, time_str, time_ms, md_download, txt_download

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            return f"Ошибка анализа: {e}", "", "", 0.0, None, None

    def evaluate_quality(
        image: Optional[Union[Image.Image, str]],
        raw_text: str,
    ) -> Tuple[int, int, int, int, int, str]:
        """Evaluate recognition quality using VLM judge."""
        if image is None:
            return 0, 0, 0, 0, 0, "Загрузите изображение"
        if not raw_text or not raw_text.strip():
            return 0, 0, 0, 0, 0, "Сначала выполните анализ"

        try:
            # API client expects a PIL Image
            if is_api_mode:
                if isinstance(image, str):
                    pil = Image.open(image)
                else:
                    pil = image
                result = analyzer.judge(image=pil, raw_text=raw_text)
            else:
                # Local service accepts path/PIL and handles SVG conversion
                result = analyzer.judge(image=image, raw_text=raw_text)

            coverage = result.get("text_coverage", 0)
            accuracy = result.get("text_accuracy", 0)
            order = result.get("order_quality", 0)
            homoglyphs = result.get("homoglyphs", 100)
            overall = result.get("overall", 0)
            notes = result.get("notes", "")

            return coverage, accuracy, order, homoglyphs, overall, notes

        except Exception as e:
            logger.exception(f"Judge failed: {e}")
            return 0, 0, 0, 0, 0, f"Ошибка: {e}"

    def render_diagram(
        algorithm_text: str,
        dpi: int,
    ) -> Optional[Image.Image]:
        """Render algorithm text as flowchart."""
        if not algorithm_text or not algorithm_text.strip():
            return None

        try:
            png_bytes = analyzer.render(algorithm_text=algorithm_text, dpi=dpi)
            return Image.open(io.BytesIO(png_bytes))
        except Exception as e:
            logger.exception(f"Render failed: {e}")
            return None

    def get_system_info() -> Dict[str, Any]:
        """Get system information."""
        if is_api_mode:
            info = analyzer.get_info()
            info["ui_mode"] = "api_client"
            info["api_url"] = settings.api_base_url
        else:
            info = analyzer.get_info()
            info["ui_mode"] = "direct_model"

        try:
            import torch
            info["gpu_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["gpu_available"] = False

        return info

    # ============== Build UI ==============

    app = gr.Blocks(title="Diagram2Table")

    with app:
        gr.Markdown("# Diagram2Table", elem_classes=["main-title"])
        gr.Markdown(
            "Распознавание диаграмм и извлечение описаний алгоритмов",
            elem_classes=["subtitle"],
        )

        with gr.Tabs():
            # ============== Tab 1: Analysis ==============
            with gr.TabItem("Анализ диаграммы"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.File(
                            label="Загрузите диаграмму (PNG, JPG, WEBP, SVG)",
                            file_count="single",
                            file_types=[".svg", ".png", ".jpg", ".jpeg", ".webp"],
                        )

                        # Preview image (shows SVG converted to PNG for preview)
                        preview_image = gr.Image(label="Предпросмотр", interactive=False)
                        preview_png_state = gr.State(value=None)

                        analyze_btn = gr.Button(
                            "Анализировать",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        table_output = gr.Markdown(
                            label="Результат (таблица)",
                            value="Результат появится здесь...",
                        )

                        with gr.Accordion("Распознанный текст", open=False):
                            raw_text_output = gr.Textbox(
                                label="Текст (построчно)",
                                lines=8,
                                interactive=False,
                            )

                        with gr.Row():
                            time_output = gr.Textbox(
                                label="Время обработки",
                                interactive=False,
                            )
                            time_ms_hidden = gr.Number(visible=False)

                        # Quality evaluation section (inline)
                        gr.Markdown("---")
                        gr.Markdown("**Оценка качества распознавания**")
                        
                        judge_btn = gr.Button("Оценить качество", variant="secondary")
                        
                        with gr.Row():
                            coverage_score = gr.Number(label="Coverage (0-100)", interactive=False)
                            accuracy_score = gr.Number(label="Accuracy (0-100)", interactive=False)
                        with gr.Row():
                            order_score = gr.Number(label="Order (0-100)", interactive=False)
                            homoglyphs_score = gr.Number(label="Homoglyphs (0-100)", interactive=False)
                        overall_score = gr.Number(label="Overall (0-100)", interactive=False)
                        judge_notes = gr.Textbox(label="Примечания", interactive=False, lines=2)

                        # File name inputs for downloads
                        md_filename = gr.Textbox(label="Имя файла MD", value="diagram.md")
                        txt_filename = gr.Textbox(label="Имя файла TXT", value="diagram.txt")

                        with gr.Row():
                            md_download = gr.DownloadButton(label="Скачать MD", variant="secondary")
                            txt_download = gr.DownloadButton(label="Скачать TXT", variant="secondary")

                # Connect file change to preview generator
                def _generate_preview(file_path: Optional[str]):
                    """Return (PIL.Image or None, png_bytes or None) for preview and state."""
                    if not file_path:
                        return None, None
                    try:
                        from PIL import Image as PILImage
                        import io
                        p = file_path
                        if isinstance(p, str) and p.lower().endswith('.svg'):
                            try:
                                from wand.image import Image as WandImage
                            except Exception:
                                raise RuntimeError("SVG preview requires Wand and ImageMagick in the UI environment")
                            with open(p, 'rb') as f:
                                svg_bytes = f.read()
                            with WandImage(blob=svg_bytes, resolution=100) as wimg:
                                wimg.format = 'png'
                                png = wimg.make_blob('png')
                            return PILImage.open(io.BytesIO(png)), png
                        else:
                            img = PILImage.open(p)
                            buf = io.BytesIO()
                            img.convert('RGB').save(buf, format='PNG')
                            return img, buf.getvalue()
                    except Exception as e:
                        logger.exception("Preview generation failed: %s", e)
                        return None, None

                image_input.change(
                    fn=_generate_preview,
                    inputs=[image_input],
                    outputs=[preview_image, preview_png_state],
                )

                # Connect analyze button
                analyze_btn.click(
                    fn=analyze_image,
                    inputs=[image_input, preview_png_state, md_filename, txt_filename],
                    outputs=[table_output, raw_text_output, time_output, time_ms_hidden, md_download, txt_download],
                )

                # Connect judge button (uses image_input and raw_text_output from analysis)
                judge_btn.click(
                    fn=evaluate_quality,
                    inputs=[image_input, raw_text_output],
                    outputs=[coverage_score, accuracy_score, order_score, homoglyphs_score, overall_score, judge_notes],
                )

            # ============== Tab 2: Render Diagram ==============
            with gr.TabItem("Генерация диаграммы"):
                gr.Markdown(
                    """
                    ### Генерация диаграммы из текста
                    
                    Введите алгоритм (по одному шагу на строку) и получите блок-схему.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        render_text = gr.Textbox(
                            label="Алгоритм (один шаг на строку)",
                            lines=10,
                            placeholder="Получить запрос\nПроверить данные\nОбработать запрос\nОтправить ответ",
                        )
                        render_dpi = gr.Slider(
                            minimum=72,
                            maximum=300,
                            value=180,
                            step=10,
                            label="DPI (качество изображения)",
                        )
                        render_btn = gr.Button("Сгенерировать диаграмму", variant="primary")

                    with gr.Column(scale=1):
                        render_output = gr.Image(
                            label="Сгенерированная диаграмма",
                            type="pil",
                        )

                # Connect render button
                render_btn.click(
                    fn=render_diagram,
                    inputs=[render_text, render_dpi],
                    outputs=[render_output],
                )

        # Footer
        gr.Markdown(
            """
            ---
            **Diagram2Table** v0.1.0 
            """
        )

    return app


def launch_ui(
    analyzer: Optional[AnalyzerBackend] = None,
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    use_api: Optional[bool] = None,
) -> None:
    """Launch the Gradio UI.

    Args:
        analyzer: Backend instance.
        host: Host to bind to.
        port: Port to use.
        share: Create public link.
        use_api: If True, use API client mode.
    """
    app = create_ui(analyzer, share, use_api=use_api)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
    )


if __name__ == "__main__":
    launch_ui()
