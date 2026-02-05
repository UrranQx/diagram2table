"""Algorithm text -> graph/mermaid -> PNG renderer.

Contains:
- Mermaid flowchart parser + renderer (matplotlib)
- No-LLM fallback: build a simple linear graph from steps text

Extracted from the notebook `best_mvp_ipynb_.ipynb`.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from PIL import Image as PILImage

# -------- Mermaid helpers (optional) --------

PROMPT_TEXT2DIAGRAM_MERMAID = """Ты — генератор диаграмм по текстовому алгоритму.

Вход: текст алгоритма (шаги на русском, могут быть условия, ветвления, циклы).
Выход: ТОЛЬКО код Mermaid.

Требования:
- Верни только mermaid-код без пояснений.
- Формат: flowchart TD
- Каждый шаг — отдельный узел.
- Если есть условие (если/иначе/проверка/вопрос) — используй ромб { } и подпиши выходы "Да"/"Нет".
- Не выдумывай шагов: используй только то, что есть во входе.
- Узлы подписывай внутри [] или {} (для условия).
""".strip()


def extract_mermaid_flowchart(text: str) -> str:
    m = re.search(r"```(?:mermaid)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    m2 = re.search(r"(flowchart\s+TD[\s\S]*)", text, flags=re.IGNORECASE)
    if m2:
        text = m2.group(1).strip()
    parts = re.split(r"(?=flowchart\s+TD)", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        for p in parts:
            p = p.strip()
            if p.lower().startswith("flowchart td"):
                text = p
                break
    return text.strip()


# -------- Graph dataclasses --------

@dataclass
class Node:
    id: str
    label: str
    shape: str  # "rect" or "diamond"


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""


# -------- Mermaid parsing --------

_NODE_RE = re.compile(
    r"""
    (?P<id>[A-Za-z][A-Za-z0-9_]*)
    (?:
        \[(?P<rect>[^\]]+)\]     |    # A[Text]
        \{(?P<diam>[^\}]+)\}     |    # A{Cond}
        \((?P<round>[^\)]+)\)         # A(Text)
    )?
    """,
    re.VERBOSE,
)

_EDGE_RE = re.compile(
    r"""
    ^
    \s*(?P<lhs>.+?)\s*
    (?P<arrow>-->|---|-.+?->|==>|-->)
    \s*(?P<rhs>.+?)\s*$
    """,
    re.VERBOSE,
)

_EDGE_LABEL_RE = re.compile(r"--\s*(?P<label>[^-]+?)\s*-->")  # A -- Да --> B


def _parse_node(token: str) -> Tuple[str, Optional[str], str]:
    token = token.strip()
    m = _NODE_RE.match(token)
    if not m:
        return token, None, "rect"
    nid = m.group("id")
    if m.group("rect") is not None:
        return nid, m.group("rect").strip(), "rect"
    if m.group("diam") is not None:
        return nid, m.group("diam").strip(), "diamond"
    if m.group("round") is not None:
        return nid, m.group("round").strip(), "rect"
    return nid, None, "rect"


def parse_mermaid_flowchart(mermaid_code: str) -> Tuple[Dict[str, Node], List[Edge]]:
    lines = [ln.strip() for ln in (mermaid_code or "").splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("flowchart"):
        lines = lines[1:]

    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    for ln in lines:
        if ln.startswith("%%"):
            continue
        m = _EDGE_RE.match(ln)
        if not m:
            continue
        lhs_raw = m.group("lhs").strip()
        rhs_raw = m.group("rhs").strip()

        edge_label = ""
        labm = _EDGE_LABEL_RE.search(ln)
        if labm:
            edge_label = labm.group("label").strip()

        src_id, src_label, src_shape = _parse_node(lhs_raw)
        dst_id, dst_label, dst_shape = _parse_node(rhs_raw)

        if src_id not in nodes:
            nodes[src_id] = Node(src_id, src_label or src_id, src_shape)
        else:
            if src_label and nodes[src_id].label == src_id:
                nodes[src_id].label = src_label
            if src_shape == "diamond":
                nodes[src_id].shape = "diamond"

        if dst_id not in nodes:
            nodes[dst_id] = Node(dst_id, dst_label or dst_id, dst_shape)
        else:
            if dst_label and nodes[dst_id].label == dst_id:
                nodes[dst_id].label = dst_label
            if dst_shape == "diamond":
                nodes[dst_id].shape = "diamond"

        edges.append(Edge(src_id, dst_id, edge_label))

    return nodes, edges


# -------- Layout + render --------

def _topo_layout(nodes: Dict[str, Node], edges: List[Edge]) -> Dict[str, Tuple[float, float]]:
    adj = {nid: [] for nid in nodes}
    indeg = {nid: 0 for nid in nodes}
    for e in edges:
        adj.setdefault(e.src, []).append(e.dst)
        indeg[e.dst] = indeg.get(e.dst, 0) + 1
        indeg.setdefault(e.src, indeg.get(e.src, 0))

    queue = [nid for nid, d in indeg.items() if d == 0]
    queue.sort()
    topo = []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
        queue.sort()

    if len(topo) != len(nodes):
        topo = list(nodes.keys())

    rank = {nid: 0 for nid in nodes}
    for u in topo:
        for v in adj.get(u, []):
            rank[v] = max(rank[v], rank[u] + 1)

    by_rank: Dict[int, List[str]] = {}
    for nid, r in rank.items():
        by_rank.setdefault(r, []).append(nid)
    for r in by_rank:
        by_rank[r].sort()

    pos = {}
    max_r = max(by_rank.keys()) if by_rank else 0
    for r in range(max_r + 1):
        layer = by_rank.get(r, [])
        for i, nid in enumerate(layer):
            pos[nid] = (i, -r)
    return pos


def render_graph_to_png(
    nodes: Dict[str, Node],
    edges: List[Edge],
    out_path: Optional[str] = None,
    dpi: int = 180,
) -> bytes:
    """Render nodes/edges directly (no Mermaid).
    
    Args:
        nodes: Dictionary of node_id -> Node
        edges: List of Edge objects
        out_path: Optional path to save the PNG file
        dpi: DPI for the output image
        
    Returns:
        PNG image as bytes
    """
    if not nodes:
        raise ValueError("nodes пустой — нечего рисовать.")

    pos = _topo_layout(nodes, edges)

    layers = {}
    for nid, (x, y) in pos.items():
        layers.setdefault(y, []).append(x)

    max_cols = max((len(sorted(set(xs))) for xs in layers.values()), default=1)
    max_rows = len(layers) if layers else 1

    fig_w = max(8, 2.2 * max_cols)
    fig_h = max(6, 1.6 * max_rows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()

    box_w = 2.8
    box_h = 0.9
    x_gap = 0.8
    y_gap = 1.2

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def to_plot(nid: str) -> Tuple[float, float]:
        x, y = pos[nid]
        px = (x - min_x) * (box_w + x_gap)
        py = (y - max_y) * (box_h + y_gap)
        return px, py

    # edges
    for e in edges:
        if e.src not in nodes or e.dst not in nodes:
            continue
        if e.src not in pos or e.dst not in pos:
            continue
        x1, y1 = to_plot(e.src)
        x2, y2 = to_plot(e.dst)

        start = (x1 + box_w / 2, y1 - box_h / 2)
        end = (x2 + box_w / 2, y2 - box_h / 2)

        arr = FancyArrowPatch(
            start, end,
            arrowstyle='->',
            mutation_scale=14,
            linewidth=1.5,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arr)

        if e.label:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2
            ax.text(mx, my + 0.12, e.label, ha="center", va="bottom", fontsize=10)

    # nodes
    for nid, node in nodes.items():
        if nid not in pos:
            continue
        x, y = to_plot(nid)
        label = (node.label or nid).strip()
        label = re.sub(r"\s+", " ", label)

        if len(label) > 26:
            words = label.split()
            lines, cur, cur_len = [], [], 0
            for w in words:
                add = len(w) + (1 if cur else 0)
                if cur_len + add <= 26:
                    cur.append(w)
                    cur_len += add
                else:
                    lines.append(" ".join(cur))
                    cur = [w]
                    cur_len = len(w)
            if cur:
                lines.append(" ".join(cur))
            label = "\n".join(lines)

        shape = node.shape
        if shape == "diamond":
            patch = FancyBboxPatch(
                (x, y - box_h), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=1.8,
                facecolor="white",
            )
            ax.add_patch(patch)
            ax.text(x + box_w/2, y - box_h/2, label, ha="center", va="center", fontsize=10, fontweight="bold")
        else:
            patch = FancyBboxPatch(
                (x, y - box_h), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.6,
                facecolor="white",
            )
            ax.add_patch(patch)
            ax.text(x + box_w/2, y - box_h/2, label, ha="center", va="center", fontsize=10)

    all_px = [to_plot(n)[0] for n in nodes if n in pos]
    all_py = [to_plot(n)[1] for n in nodes if n in pos]
    ax.set_xlim(min(all_px) - 1, max(all_px) + box_w + 1)
    ax.set_ylim(min(all_py) - box_h - 1, max(all_py) + 1)

    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()
    
    # Optionally save to file
    if out_path:
        with open(out_path, 'wb') as f:
            f.write(png_bytes)
    
    return png_bytes


def render_flowchart_to_png(
    mermaid_code: str,
    out_path: Optional[str] = None,
    dpi: int = 180
) -> bytes:
    """Parse Mermaid code and render to PNG bytes."""
    nodes, edges = parse_mermaid_flowchart(mermaid_code)
    if not nodes:
        raise ValueError("Не удалось распарсить Mermaid-код (нет узлов).")

    return render_graph_to_png(nodes, edges, out_path=out_path, dpi=dpi)


# -------- No-LLM: steps -> graph --------

def algorithm_text_to_graph_simple(algorithm_text: str) -> Tuple[Dict[str, Node], List[Edge]]:
    """Convert algorithm text (one step per line) to a simple linear graph."""
    lines = [ln.strip() for ln in (algorithm_text or "").splitlines() if ln.strip()]
    nodes = {f"n{i+1}": Node(id=f"n{i+1}", label=lines[i], shape="rect") for i in range(len(lines))}
    edges = [Edge(src=f"n{i+1}", dst=f"n{i+2}") for i in range(len(lines)-1)]
    return nodes, edges


def algorithm_text_to_diagram_png_no_llm(
    algorithm_text: str,
    out_path: Optional[str] = None,
    dpi: int = 180
) -> bytes:
    """Convert algorithm text to PNG diagram without using LLM.
    
    Args:
        algorithm_text: Text with one step per line
        out_path: Optional path to save PNG file
        dpi: DPI for output image
        
    Returns:
        PNG image as bytes
    """
    nodes, edges = algorithm_text_to_graph_simple(algorithm_text)
    if not nodes:
        raise ValueError("Текст алгоритма пустой — нечего рисовать.")
    return render_graph_to_png(nodes, edges, out_path=out_path, dpi=dpi)
