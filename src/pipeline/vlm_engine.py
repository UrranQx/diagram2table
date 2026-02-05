"""VLM engine: model loading + generation helpers + text cleanup.

Extracted from the notebook `best_mvp_ipynb_.ipynb`.
"""

from __future__ import annotations

import os
import re
from typing import List, Sequence, Union, Optional, Tuple, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID: str = os.environ.get("QWEN25_VL_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")


def load_qwen25_vl(model_id: str = MODEL_ID) -> Tuple[Any, Any]:
    """Load processor+model (GPU if available, otherwise CPU).

    Tries to enable 4-bit quantization on CUDA (bitsandbytes) to fit <8GB VRAM.
    Returns: (model, processor)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    common_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ),
            )
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, quantization_config=bnb, device_map="auto", **common_kwargs
            )
            model.eval()
            return model, processor
        except Exception:
            pass

        model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", **common_kwargs)
        model.eval()
        return model, processor

    model = AutoModelForVision2Seq.from_pretrained(model_id, **common_kwargs)
    model.eval()
    return model, processor


_RE_ROLE_PREFIX = re.compile(r"^(system|assistant|user)\s*[:\-]\s*", flags=re.IGNORECASE)
_RE_WS = re.compile(r"\s+")
_RE_YESNO = re.compile(r"^(да|нет)$", flags=re.IGNORECASE)
_BAD_LINE_PATTERNS = [
    re.compile(r"^на изображении\b.*", flags=re.IGNORECASE),
    re.compile(r"^выведи\b.*", flags=re.IGNORECASE),
    re.compile(r"^вывести\b.*", flags=re.IGNORECASE),
    re.compile(r"^правила\b.*", flags=re.IGNORECASE),
    re.compile(r"^-+\s*не\b.*", flags=re.IGNORECASE),
    re.compile(r"^-+\s*выводи\b.*", flags=re.IGNORECASE),
    re.compile(r"^-+\s*кажд\w*\b.*", flags=re.IGNORECASE),
    re.compile(r"^-+\s*не добавляй\b.*", flags=re.IGNORECASE),
    re.compile(r"^\.\s*$"),
]


def clean_vlm_output(model_text: str, prompt: str = "", drop_questions: bool = False) -> str:
    t = (model_text or "").strip()
    if not t:
        return ""

    if "assistant" in t:
        t = t.split("assistant")[-1].strip()

    p = (prompt or "").strip()
    if p and p in t:
        t = t.replace(p, "").strip()

    raw_lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    out_lines, seen = [], set()

    for ln in raw_lines:
        low = ln.lower().strip()
        if low in ("system", "user", "assistant"):
            continue

        ln = _RE_ROLE_PREFIX.sub("", ln).strip()
        ln = ln.replace("\u200b", "").strip()
        if not ln:
            continue

        low = ln.lower().strip()
        if any(rgx.match(low) for rgx in _BAD_LINE_PATTERNS):
            continue
        if _RE_YESNO.fullmatch(low):
            continue
        if drop_questions and ln.endswith("?"):
            continue

        ln = _RE_WS.sub(" ", ln).strip()
        key = ln.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out_lines.append(ln)

    return "\n".join(out_lines).strip()


def build_messages_for_image(image: Image.Image, prompt: str):
    """Build Qwen-VL chat messages with image + prompt text."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _resize_for_vlm(img: Image.Image, max_side: int = 768) -> Image.Image:
    """Resize image keeping aspect ratio so max side <= max_side."""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((max(1, nw), max(1, nh)))


@torch.inference_mode()
def vlm_generate_text_batch(
    model,
    processor,
    images: Union[Image.Image, Sequence[Image.Image]],
    prompt: str,
    *,
    max_new_tokens: int = 256,
    batch_size: int = 2,
    max_side: Optional[int] = 768,
    clean: bool = True,
) -> Union[str, List[str]]:
    """Generate text for one or many images using model.generate()."""
    single = isinstance(images, Image.Image)
    imgs = [images] if single else list(images)

    if max_side:
        imgs = [_resize_for_vlm(im, max_side=max_side) for im in imgs]

    outs: List[str] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        msgs = [build_messages_for_image(im, prompt) for im in batch]
        chats = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in msgs
        ]
        inputs = processor(text=chats, images=batch, padding=True, return_tensors="pt")
        if device == "cuda":
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
        decoded = processor.batch_decode(gen, skip_special_tokens=True)
        for d in decoded:
            outs.append(clean_vlm_output(d, prompt=prompt) if clean else (d or "").strip())

    return outs[0] if single else outs
