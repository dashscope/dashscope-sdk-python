# -*- coding: utf-8 -*-
"""Built-in vision tool — let the agent analyze local images itself.

Previously the only multimodal entry point was user-input ``@path``
expansion, so after ``scrape_web_screenshot`` / ``camera_capture`` the
model could not see the saved image and sometimes echoed the
user-facing ``@path`` hint verbatim, ending the turn prematurely.
``view_image`` closes that gap by routing the image to a
vision-capable model on the active provider.

Very tall/wide images (e.g. a long announcement table) get downscaled
or truncated by vision-model backends until text is illegible, so
``view_image`` auto-slices them with Pillow (when installed) and runs
one vision call per slice. Without Pillow it falls back to sending the
whole image as-is.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
from typing import Callable

from dashscope.acli.config import Config, is_vision_model
from dashscope.acli.tools.registry import (
    PermissionLevel,
    ToolDefinition,
    registry,
)

_FALLBACK_VISION_MODELS = {
    "tongyi": "qwen-vl-max",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}

# Slicing triggers: beyond these the model backend downscales or
# truncates the image and fine text becomes unreadable.
_SLICE_MAX_LONG_SIDE = 8000
_SLICE_MAX_RATIO = 4.0
_SLICE_STEP = 2000
_SLICE_OVERLAP = 100
_SLICE_MAX_COUNT = 32
_SLICE_JPEG_QUALITY = 88


def pick_vision_model(config: Config) -> str:
    """Choose a vision-capable model on the active provider.

    Preference: the active model when it accepts images, then the first
    vision model declared on the provider's extension block, then a
    per-provider built-in fallback.
    """
    if is_vision_model(config.model):
        return config.model

    from dashscope.acli.extensions import find_provider

    ext = find_provider(config.provider)
    if ext and ext.vision_models:
        return ext.vision_models[0]
    return _FALLBACK_VISION_MODELS.get(config.provider, "qwen-vl-max")


def plan_slices(width: int, height: int):
    """Return crop boxes for an oversized image, or None if it fits.

    Slices run along the long axis with a small overlap so rows cut at
    a box boundary still appear complete in one of the two neighbors.
    """
    long_side = max(width, height)
    short_side = max(min(width, height), 1)
    if (
        long_side <= _SLICE_MAX_LONG_SIDE
        and long_side / short_side <= _SLICE_MAX_RATIO
    ):
        return None

    step = _SLICE_STEP
    overlap = _SLICE_OVERLAP
    count = math.ceil((long_side - overlap) / (step - overlap))
    if count > _SLICE_MAX_COUNT:
        # Re-derive a coarser step; drop the overlap so the cap holds.
        step = math.ceil(long_side / _SLICE_MAX_COUNT)
        overlap = 0

    boxes = []
    if height >= width:
        y = 0
        while y < height:
            y2 = min(y + step, height)
            boxes.append((0, y, width, y2))
            if y2 == height:
                break
            y = y2 - overlap
    else:
        x = 0
        while x < width:
            x2 = min(x + step, width)
            boxes.append((x, 0, x2, height))
            if x2 == width:
                break
            x = x2 - overlap
    return boxes


def _slice_data_urls(image, boxes) -> list:
    """Crop ``boxes`` out of a PIL image, JPEG-encode as data URLs."""
    urls = []
    for box in boxes:
        crop = image.crop(box)
        if crop.mode not in ("RGB", "L"):
            crop = crop.convert("RGB")
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=_SLICE_JPEG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode()
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


async def _ask_vision(provider, data_url: str, text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    response = await provider.chat(messages)
    return response.content or "(vision model returned no content)"


def register_vision_tools(config: Config, *, get_provider_fn: Callable):
    """Register ``view_image``; called once after agent construction."""

    async def view_image(
        image_path: str,
        question: str = (
            "Describe this image in detail. If it contains text or "
            "tables, transcribe them verbatim."
        ),
        slice_mode: str = "auto",
    ) -> str:
        from dashscope.acli.utils.paths import validate_path

        try:
            safe_path = validate_path(image_path)
        except (ValueError, OSError) as e:
            return f"Error: failed to read image: {e}"

        model = pick_vision_model(config)
        provider = get_provider_fn(
            config.provider,
            model,
            config.api_key,
            base_url=config.base_url or None,
        )

        boxes = None
        image = None
        if slice_mode != "never":
            try:
                from PIL import Image

                image = Image.open(safe_path)
                image.load()
                boxes = plan_slices(*image.size)
            except Exception:  # pylint: disable=broad-except
                # No Pillow, or not a decodable image: fall back to the
                # whole-image path and let it report the real error.
                boxes = None
                image = None

        if boxes:
            axis = (
                "top-to-bottom"
                if image.height >= image.width
                else ("left-to-right")
            )
            urls = _slice_data_urls(image, boxes)
            image.close()
            total = len(urls)
            tasks = [
                _ask_vision(
                    provider,
                    url,
                    (
                        f"Slice {i + 1}/{total} ({axis}) of one large "
                        "image. " + question
                    ),
                )
                for i, url in enumerate(urls)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parts = [
                f"(Image auto-split into {total} slices {axis}; "
                "results per slice below.)",
            ]
            for i, r in enumerate(results, start=1):
                if isinstance(r, BaseException):
                    parts.append(f"--- slice {i}/{total} ---\nError: {r}")
                else:
                    parts.append(f"--- slice {i}/{total} ---\n{r}")
            return "\n\n".join(parts)

        try:
            from dashscope.acli.utils.images import image_to_data_url

            data_url = image_to_data_url(safe_path)
        except (ValueError, OSError) as e:
            return f"Error: failed to read image: {e}"
        try:
            return await _ask_vision(provider, data_url, question)
        except Exception as e:  # pylint: disable=broad-except
            return f"Error: vision call failed: {e}"

    registry.register(
        ToolDefinition(
            name="view_image",
            description=(
                "Analyze a local image file (screenshot, photo, chart) "
                "with a vision model: OCR/transcribe text, describe "
                "layout or content. Very tall/wide images are split "
                "automatically. Use after scrape_web_screenshot or "
                "camera_capture, or whenever you need to read an image."
            ),
            permission=PermissionLevel.AUTO,
            func=view_image,
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": (
                            "Path to the image file (relative to the "
                            "workspace or absolute)"
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "What to extract from the image; defaults "
                            "to a full description + verbatim OCR"
                        ),
                    },
                    "slice_mode": {
                        "type": "string",
                        "enum": ["auto", "never"],
                        "description": (
                            "auto (default): split oversized images "
                            "into slices; never: always send the whole "
                            "image in one call"
                        ),
                    },
                },
                "required": ["image_path"],
            },
        ),
    )
