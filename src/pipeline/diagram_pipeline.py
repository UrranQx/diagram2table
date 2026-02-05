"""Diagram -> steps pipeline (tiling + top-K + post-processing + judge + markdown).

Extracted from the notebook `best_mvp_ipynb_.ipynb`.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch

from .vlm_engine import _resize_for_vlm, vlm_generate_text_batch, clean_vlm_output

# ----- Prompts -----

TEXT_ONLY_PROMPT_RU = """На изображении диаграмма (BPMN или блок-схема).

Выведи ТОЛЬКО текст, который ВИЗУАЛЬНО написан внутри блоков (прямоугольники/квадраты/ромбы/овалы/треугольники), по одной фразе на строку.
Правила:
- Не добавляй ничего от себя.
- Не выводи system/user/assistant и не повторяй инструкции.
- Если текста нет — верни пустую строку.
""".strip()

JUDGE_PROMPT_FMT_RU = """Ты — строгий судья качества распознавания текста на диаграммах.

Оцени по изображению и распознанному результату.

Игнорируй: надписи на стрелках, условия "Да/Нет", "Старт/Конец", заголовки, роли/дорожки, служебный текст.

Верни ответ СТРОГО в формате 9 строк:

COVERAGE: <0-100>
ACCURACY: <0-100>
ORDER: <0-100>
HOMOGLYPHS: 0           (НЕ оценивай это поле, оставь 0)
OVERALL: <0-100>
MISSING1: <пример текста, который есть на изображении, но отсутствует в выводе или пусто>
MISSING2: <... или пусто>
HALLUC1: <пример строки из вывода, которой нет на изображении или пусто>
NOTES: <1 короткое предложение>

Никаких других строк. Никаких списков. Никаких JSON.
""".strip()

HOMOGLYPHS = {
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
    'X': 'Х', 'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о',
    'p': 'р', 'x': 'х', 'y': 'у', '3': 'З', '0': 'О',
}
_ALNUM_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")


def homoglyphs_score(text: str) -> int:
    """100 = нет гомоглифов, 0 = все символы (из учитываемых) — гомоглифы."""
    if not text:
        return 100
    total = 0
    bad = 0
    for ch in text:
        if not _ALNUM_RE.match(ch):
            continue
        total += 1
        if ch in HOMOGLYPHS:
            bad += 1
    if total == 0:
        return 100
    score = round(100 * (1 - bad / total))
    return max(0, min(100, int(score)))


# ----- Tiling utilities -----

def crop_pil(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box).convert("RGB")


def content_score(tile_img: Image.Image) -> float:
    """Cheap content metric: mean abs gradients (higher => more text/lines)."""
    arr = np.asarray(tile_img.convert("L"), dtype=np.float32) / 255.0
    if arr.shape[0] > 256 or arr.shape[1] > 256:
        arr = arr[::2, ::2]
    gx = np.abs(np.diff(arr, axis=1)).mean() if arr.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    return float(gx + gy)


def make_tiles(
    img: Image.Image,
    *,
    tile_w: int = 1024,
    tile_h: int = 1024,
    overlap: int = 128,
) -> List[Tuple[int, int, int, int]]:
    w, h = img.size
    step_x = max(1, tile_w - overlap)
    step_y = max(1, tile_h - overlap)

    boxes: List[Tuple[int, int, int, int]] = []
    y = 0
    while y < h:
        x = 0
        y2 = min(h, y + tile_h)
        y1 = max(0, y2 - tile_h)
        while x < w:
            x2 = min(w, x + tile_w)
            x1 = max(0, x2 - tile_w)
            boxes.append((x1, y1, x2, y2))
            x += step_x
        y += step_y
    return boxes


def select_topk_tiles(
    tiles: Sequence[Tuple[int, int, int, int]],
    img: Image.Image,
    *,
    top_k_tiles: int = 4,
    min_score: float = 0.01,
) -> List[Tuple[int, int, int, int]]:
    scored = []
    for box in tiles:
        sc = content_score(crop_pil(img, box))
        if sc >= min_score:
            scored.append((sc, box))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in scored[:top_k_tiles]]


def choose_top_k_tiles(
    image: Union[str, Image.Image],
    *,
    base_side: int = 1400,
    base_k: int = 4,
    k_min: int = 2,
    k_max: int = 12,
) -> int:
    """Heuristic: top_k grows with image area and extreme aspect ratios."""
    if isinstance(image, str):
        w, h = Image.open(image).size
    else:
        w, h = image.size
    area = w * h
    base_area = base_side * base_side
    scale = (area / base_area) ** 0.5
    k = int(round(base_k * scale))
    aspect = max(w / h, h / w)
    if aspect >= 2.0:
        k += 2
    if aspect >= 3.0:
        k += 2
    return max(k_min, min(k_max, k))


# ----- Post-processing -----

def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def filter_tasks(lines: List[str]) -> List[str]:
    bad_exact = {"start", "end", "старт", "конец"}
    seen = set()
    out = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        low = s.lower()
        if low in bad_exact:
            continue
        if re.fullmatch(r"(да|нет)", low):
            continue
        key = _norm_key(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


_ORDER_HINTS = [
    "начало", "старт", "получить", "ввести", "проверить", "проверка",
    "создать", "сформировать", "отправить", "сохранить", "успешно", "завершить", "конец",
]


def sort_tasks_semantic(lines: List[str]) -> List[str]:
    def hint_index(s: str) -> int:
        low = s.lower()
        for i, h in enumerate(_ORDER_HINTS):
            if h in low:
                return i
        return 10_000
    return sorted(lines, key=hint_index)


def score(lines_before: List[str], lines_after: List[str]) -> Dict[str, Any]:
    """Small helper for notebook diagnostics: how many lines kept/removed."""
    return {
        "before": len(lines_before),
        "after": len(lines_after),
        "removed": max(0, len(lines_before) - len(lines_after)),
    }


# ----- Main analysis -----
def analyze_diagram(
    image: Union[str, Image.Image],
    model,
    processor,
    *,
    top_k_tiles: int = 4,
    max_new_tokens: int = 128,
    batch_size: int = 2,
    max_side: int = 768,
    prompt: str = TEXT_ONLY_PROMPT_RU,
    early_stop_patience: int = 2,
    early_stop_min_new: int = 1,
) -> Dict[str, Any]:
    """Image -> raw_text (one step per line) + meta.
    
    Args:
        image: PIL Image or file path
        model: VLM model
        processor: VLM processor
        top_k_tiles: Number of top tiles to process
        max_new_tokens: Maximum tokens for generation
        batch_size: Batch size for processing
        max_side: Maximum side for resizing
        prompt: Prompt template
        early_stop_patience: Early stopping patience
        early_stop_min_new: Minimum new lines for early stopping
    
    Returns:
        Dict with raw_text, lines, and meta
    """
    t0 = time.time()
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB") if image.mode != "RGB" else image
    w, h = img.size

    do_tiling = (max(w, h) > 1400) or (w > 1.6 * h) or (h > 1.6 * w)
    mode = "full"

    lines: List[str] = []
    if do_tiling:
        mode = "tiling"
        tiles = make_tiles(img)
        tiles = select_topk_tiles(tiles, img, top_k_tiles=top_k_tiles)
        crops = [crop_pil(img, box) for box in tiles]

        seen = set()
        no_gain = 0

        for i in range(0, len(crops), batch_size):
            batch = crops[i : i + batch_size]
            texts = vlm_generate_text_batch(
                model, processor, batch, prompt,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                max_side=max_side,
                clean=True,
            )
            if isinstance(texts, str):
                texts = [texts]

            added = 0
            for txt in texts:
                for ln in (txt or "").splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    key = ln.lower()
                    if key not in seen:
                        seen.add(key)
                        lines.append(ln)
                        added += 1

            if added < early_stop_min_new:
                no_gain += 1
            else:
                no_gain = 0

            if no_gain >= early_stop_patience:
                break
    else:
        txt = vlm_generate_text_batch(
            model, processor, img, prompt,
            max_new_tokens=max_new_tokens,
            batch_size=1,
            max_side=max_side,
            clean=True,
        )
        lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]

    lines_before = list(lines)
    lines = filter_tasks(lines)
    lines = sort_tasks_semantic(lines)

    meta = {
        "image_size": (w, h),
        "mode": mode,
        "elapsed_sec": round(time.time() - t0, 3),
        "top_k_tiles": top_k_tiles,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "postprocess": score(lines_before, lines),
    }
    raw_text = "\n".join(lines).strip()
    return {"raw_text": raw_text, "lines": lines, "meta": meta}


# ----- Markdown output -----

def action_to_md_rows_keep_linebreaks(text: str) -> str:
    rows = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(f"| {ln.replace('|', '\\|')} |")
    return "\n".join(rows)


def raw_text_to_md_table(raw_text: str) -> str:
    header = "| Действие |\n|---|"
    rows = action_to_md_rows_keep_linebreaks(raw_text)
    return f"{header}\n{rows}".strip()


# ----- Judge -----

@torch.inference_mode()
def judge_recognition_quality(
    image: Union[str, Image.Image],
    raw_text: str,
    model,
    processor,
    max_new_tokens: int = 200
) -> dict:
    """Use VLM to evaluate recognition quality."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    try:
        image_j = _resize_for_vlm(image, max_side=1024)
    except Exception:
        image_j = image

    prompt_text = (
        JUDGE_PROMPT_FMT_RU
        + "\n\nРаспознанный результат (каждая строка — блок):\n"
        + (raw_text or "").strip()
    )

    messages = [
        {"role": "system", "content": "Ты отвечаешь строго по формату."},
        {"role": "user", "content": [
            {"type": "image", "image": image_j},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image_j], padding=True, return_tensors="pt")

    if torch.cuda.is_available():
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    out = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _get_int(key: str, default=0):
        m = re.search(rf"^{key}:\s*(\d{{1,3}})\s*$", out, flags=re.MULTILINE)
        if not m:
            return default
        v = int(m.group(1))
        return max(0, min(100, v))

    def _get_str(key: str):
        m = re.search(rf"^{key}:\s*(.*)\s*$", out, flags=re.MULTILINE)
        return (m.group(1).strip() if m else "")

    result = {
        "text_coverage": _get_int("COVERAGE", 0),
        "text_accuracy": _get_int("ACCURACY", 0),
        "order_quality": _get_int("ORDER", 0),
        "homoglyphs": homoglyphs_score(raw_text or ""),
        "overall": _get_int("OVERALL", 0),
        "missing_examples": [x for x in [_get_str("MISSING1"), _get_str("MISSING2")] if x and x.lower() != "пусто"],
        "hallucinated_examples": [x for x in [_get_str("HALLUC1")] if x and x.lower() != "пусто"],
        "notes": _get_str("NOTES"),
        "raw_judge_output": out,
    }
    return result


def score_to_markdown(score_dict: Dict[str, Any]) -> str:
    if "error" in score_dict:
        return f"**Judge error:** {score_dict.get('error')}\n\n```\n{score_dict.get('raw','')}\n```"

    cov = score_dict.get("text_coverage", "?")
    acc = score_dict.get("text_accuracy", "?")
    order = score_dict.get("order_quality", "?")
    hg = score_dict.get("homoglyphs", "?")
    overall = score_dict.get("overall", "?")
    return (
        f"**Оценка качества**\n\n"
        f"- Coverage: **{cov}/100**\n"
        f"- Accuracy: **{acc}/100**\n"
        f"- Order: **{order}/100**\n"
        f"- Homoglyphs: **{hg}/100**\n"
        f"- Overall: **{overall}/100**"
    )
