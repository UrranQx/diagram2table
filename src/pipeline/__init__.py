"""Pipeline module: VLM-based diagram analysis with tiling, judge, and rendering.

This module contains the core ML pipeline extracted from the research notebook.
"""

from .vlm_engine import (
    load_qwen25_vl,
    vlm_generate_text_batch,
    clean_vlm_output,
    build_messages_for_image,
    MODEL_ID,
)

from .diagram_pipeline import (
    analyze_diagram,
    judge_recognition_quality,
    raw_text_to_md_table,
    score_to_markdown,
    homoglyphs_score,
    choose_top_k_tiles,
    TEXT_ONLY_PROMPT_RU,
)

from .flowchart_render import (
    Node,
    Edge,
    render_graph_to_png,
    render_flowchart_to_png,
    algorithm_text_to_graph_simple,
    algorithm_text_to_diagram_png_no_llm,
    parse_mermaid_flowchart,
    extract_mermaid_flowchart,
)

__all__ = [
    # VLM Engine
    "load_qwen25_vl",
    "vlm_generate_text_batch",
    "clean_vlm_output",
    "build_messages_for_image",
    "MODEL_ID",
    # Pipeline
    "analyze_diagram",
    "judge_recognition_quality",
    "raw_text_to_md_table",
    "score_to_markdown",
    "homoglyphs_score",
    "choose_top_k_tiles",
    "TEXT_ONLY_PROMPT_RU",
    # Rendering
    "Node",
    "Edge",
    "render_graph_to_png",
    "render_flowchart_to_png",
    "algorithm_text_to_graph_simple",
    "algorithm_text_to_diagram_png_no_llm",
    "parse_mermaid_flowchart",
    "extract_mermaid_flowchart",
]
