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

``auto_perceive`` is the read_file hook: nine bench runs (071c–071i)
showed the model reads an image as text exactly once, receives the
"call view_image / tesseract" redirect, and goes back to pixel
scripting within seconds — advice is ignorable, delivered perception
is not. When read_file hits a binary image it calls ``auto_perceive``,
which runs the same vision pipeline inline and injects the result,
falling back to the text redirect whenever perception is unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import threading
from typing import Callable

from dashscope.acli.config import Config, is_vision_model
from dashscope.acli.tools.registry import (
    PermissionLevel,
    ToolDefinition,
    registry,
)

_FALLBACK_VISION_MODELS = {
    # qwen-vl-max is retired on the public DashScope endpoint (404 "Model
    # not exist"; qwen-vl-max-latest exists but is 403-gated on accounts
    # without access). qwen3-vl-plus is the current generally-available VL
    # model — verified against the live endpoint 2026-10-13 after the
    # stale fallback silently broke every view_image call.
    "tongyi": "qwen3-vl-plus",
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

# Region crops (the ``region`` parameter): small crops are enlarged
# before the vision call so dense content stays legible.
_REGION_MIN_SHORT_SIDE = 768
_REGION_MAX_UPSCALE = 4.0

# Independent samples per region read. Region reads exist for
# exact-character verification, where one sample is a coin flip at the
# recognizer's noise floor — gcode-to-text 071r got i/ï/1 for the same
# glyph across identical crops and 071s committed after single reads of
# each region, never seeing the disagreements. Sampling surfaces the
# noise in the result text so it can be adjudicated instead of trusted.
# Runs concurrently: wall-clock stays ~one read. Env-overridable.
_REGION_SAMPLES = int(os.environ.get("ACLI_REGION_SAMPLES", "3"))

# Adopted-once signal for the run_command vision tip: after the first
# view_image call the tool-choice nudges stop (see shell._vision_tip).
_view_image_calls = 0

# Hard cap on one vision call. gcode-to-text (071n) ended with a
# view_image await that produced no output for 700+s — the spinner
# froze, the turn never returned, and the harness watchdog reaped the
# container before the in-container cap could. A vision read that
# takes minutes is indistinguishable from a hang; fail it and let the
# model retry or narrow the ask. Env-overridable for slow endpoints.
_VISION_CALL_TIMEOUT = int(os.environ.get("ACLI_VISION_TIMEOUT", "300"))


def view_image_call_count() -> int:
    return _view_image_calls


def _reset_view_image_calls() -> None:
    """Test hook: the counter is per-process, suites need a clean slate."""
    global _view_image_calls
    _view_image_calls = 0


# Auto-perception state. _PERCEPTION_CTX is stored by
# register_vision_tools so the read_file hook can reach the active
# provider/config without threading them through the filesystem layer.
_PERCEPTION_CTX = None
_auto_perceive_attempts = 0
_auto_perceived_paths: set = set()
_AUTO_PERCEIVE_MAX = 5
# A sliced oversized image runs its calls concurrently; 240s bounds the
# tail without cutting realistic reads.
_AUTO_PERCEIVE_TIMEOUT = 240


def auto_perceive_count() -> int:
    return _auto_perceive_attempts


def _reset_auto_perceive() -> None:
    """Test hook: attempts are per-process, suites need a clean slate."""
    global _auto_perceive_attempts
    _auto_perceive_attempts = 0
    _auto_perceived_paths.clear()


def _auto_perceive_budget() -> int:
    try:
        return int(os.environ.get("ACLI_AUTO_VISION_MAX", _AUTO_PERCEIVE_MAX))
    except ValueError:
        return _AUTO_PERCEIVE_MAX


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
    return _FALLBACK_VISION_MODELS.get(config.provider, "qwen3-vl-plus")


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
    return [_encode_jpeg(image.crop(box)) for box in boxes]


def _encode_jpeg(crop) -> str:
    if crop.mode not in ("RGB", "L"):
        crop = crop.convert("RGB")
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=_SLICE_JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def parse_region(region: str, width: int, height: int) -> tuple:
    """Parse ``x0,y0,x1,y1`` fractions into a pixel crop box.

    Fractions keep the parameter resolution-independent: the model can
    ask for "the top-left ninth" as 0,0,0.333,0.333 without knowing the
    image size. Pixels would invite off-by-one hallucinations.
    """
    parts = [p.strip() for p in region.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f'expected "x0,y0,x1,y1" (4 comma-separated fractions), '
            f"got {region!r}",
        )
    try:
        x0, y0, x1, y1 = (float(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"region values must be numbers, got {region!r}",
        ) from None
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(
            f"region must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1, got {region!r}",
        )
    return (
        int(x0 * width),
        int(y0 * height),
        max(int(x1 * width), int(x0 * width) + 1),
        max(int(y1 * height), int(y0 * height) + 1),
    )


def _upscale_for_vision(crop):
    """Enlarge small crops: dense content (digit grids, small print)
    reads far more accurately once the short side is ~768px — this is
    what makes a real crop beat asking about a region of the full image.
    """
    w, h = crop.size
    short = max(min(w, h), 1)
    if short >= _REGION_MIN_SHORT_SIDE:
        return crop
    factor = min(_REGION_MAX_UPSCALE, _REGION_MIN_SHORT_SIDE / short)
    from PIL import Image

    return crop.resize(
        (max(1, round(w * factor)), max(1, round(h * factor))),
        Image.LANCZOS,
    )


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
    try:
        response = await asyncio.wait_for(
            provider.chat(messages),
            timeout=_VISION_CALL_TIMEOUT,
        )
    except (TimeoutError, asyncio.TimeoutError):
        # asyncio.TimeoutError is distinct from TimeoutError on 3.10.
        return (
            f"Error: vision call timed out after {_VISION_CALL_TIMEOUT}s; "
            "retry, or narrow the region/question."
        )
    return response.content or "(vision model returned no content)"


def _aggregate_region_answers(answers: list[str]) -> str:
    """Merge the independent samples of one region read.

    A unanimous vote returns the single answer, indistinguishable from a
    one-shot read. Any split shows every sample with its vote count so
    the disagreement is visible in the result — hiding it would be the
    single-read coin flip all over again (gcode-to-text 071s).
    """
    if not answers:
        return "(vision model returned no content)"
    if len(set(answers)) == 1:
        return answers[0]
    samples = "\n\n".join(
        f"[sample {i + 1}/{len(answers)}]\n{a}" for i, a in enumerate(answers)
    )
    return (
        samples
        + "\n\n[(these independent reads of the same crop disagree — do "
        "not just pick one: re-read a tighter crop of only the contested "
        "glyph with a feature question (dot, serif, enclosed counters "
        "present or absent), and prefer the candidate that makes the "
        "assembled string read as coherent text)]"
    )


async def _perceive_region(
    config: Config,
    get_provider_fn: Callable,
    provider,
    safe_path: str,
    question: str,
    slice_mode: str,
    region: str,
) -> str:
    """Region read: real pixel crop, upscaled, sampled _REGION_SAMPLES×."""
    # A real pixel crop beats asking about a region of the full
    # image: solve-sudoku 071l did everything procedurally right
    # (whole read, band re-reads, box re-reads) but the crops were
    # attention-simulated by the VL and two cells came back wrong —
    # the model then "resolved" the inconsistency by guessing and
    # solved a different puzzle.
    try:
        from PIL import Image
    except ImportError:
        # Pillow is optional and the offline bench bundle is
        # pure-python (gcode-to-text 071o: three region calls
        # errored in a row). Degrade to the whole-image read with
        # working alternatives instead of failing the verification
        # loop outright.
        whole = await _perceive_image(
            config,
            get_provider_fn,
            safe_path,
            question,
            slice_mode,
        )
        return (
            # Warning FIRST: appended after the read, the note let
            # gcode-to-text 071u mistake two whole-image reads for
            # two independent region verifications of its answer.
            "[REGION READ FAILED — Pillow is not installed in this "
            "environment. What follows is a WHOLE-IMAGE read, NOT "
            "the requested crop: it is not independent verification "
            "of the region, and it must not be treated as "
            "confirmation of an earlier whole-image read. "
            "`pip install pillow` enables real region crops; or "
            "zoom the region into its own image file and read "
            "that.]\n\n" + whole
        )
    try:
        with Image.open(safe_path) as im:
            im.load()
            box = parse_region(region, *im.size)
            crop = _upscale_for_vision(im.crop(box))
            data_url = _encode_jpeg(crop)
    except ValueError as e:
        return f"Error: invalid region: {e}"
    except Exception as e:  # pylint: disable=broad-except
        return f"Error: failed to crop region: {e}"
    try:
        answers = await asyncio.gather(
            *(
                _ask_vision(provider, data_url, question)
                for _ in range(max(_REGION_SAMPLES, 1))
            ),
        )
    except Exception as e:  # pylint: disable=broad-except
        return f"Error: vision call failed: {e}"
    return _aggregate_region_answers(list(answers))


async def _perceive_image(
    config: Config,
    get_provider_fn: Callable,
    safe_path: str,
    question: str,
    slice_mode: str,
    region: str = "",
) -> str:
    """Run one vision read of ``safe_path`` (pre-validated).

    Shared by the view_image tool and the read_file auto-perceive hook;
    errors are returned as text, matching the tool contract.
    """
    model = pick_vision_model(config)
    provider = get_provider_fn(
        config.provider,
        model,
        config.api_key,
        base_url=config.base_url or None,
    )

    if region:
        return await _perceive_region(
            config,
            get_provider_fn,
            provider,
            safe_path,
            question,
            slice_mode,
            region,
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


def perception_available() -> bool:
    """Whether an agent has registered a perception context."""
    return _PERCEPTION_CTX is not None


def auto_perceive(safe_path: str, *, via: str = "read_file") -> str | None:
    """Perceive an image inline and return the read, or None.

    Called by read_file (an image opened as text) and by run_command (a
    command referencing an image). ``safe_path`` must already exist.
    Returns None — caller then falls back to its text hint — when the
    hook is disabled (ACLI_AUTO_VISION=0), unconfigured (no agent
    registered a perception context), this same image was already
    perceived once (the content is in context; repeats just burn
    latency), out of budget (ACLI_AUTO_VISION_MAX successful+failed
    attempts per process), or the pipeline itself raises.

    The agent loop runs sync tools inline on the event-loop thread, so
    the async pipeline runs on a helper thread with its own loop — the
    same blocking shape run_command already has.
    """
    global _auto_perceive_attempts
    if os.environ.get("ACLI_AUTO_VISION", "1") == "0":
        return None
    if _PERCEPTION_CTX is None:
        return None
    real = os.path.realpath(safe_path)
    if real in _auto_perceived_paths:
        return None
    if _auto_perceive_attempts >= _auto_perceive_budget():
        return None
    _auto_perceive_attempts += 1
    config, get_provider_fn = _PERCEPTION_CTX
    question = (
        "Describe this image in detail. If it contains text, digits or "
        "tables, transcribe them verbatim."
    )

    box: list = []

    def _runner():
        try:
            box.append(
                asyncio.run(
                    _perceive_image(
                        config,
                        get_provider_fn,
                        real,
                        question,
                        "auto",
                    ),
                ),
            )
        except BaseException:  # pylint: disable=broad-except
            box.append(None)

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout=_AUTO_PERCEIVE_TIMEOUT)
    if not box or box[0] is None:
        return None
    _auto_perceived_paths.add(real)
    if via == "command":
        prefix = (
            "[this command references an image; its content was "
            "perceived with a vision model (auto-perception):]"
        )
    else:
        prefix = (
            "[read_file cannot decode images as text; this image was "
            "perceived with a vision model (auto-perception):]"
        )
    return (
        f"{prefix}\n"
        f"{box[0]}\n"
        "[for exact digit-level text cross-check with tesseract OCR on "
        "the file, or call view_image with a focused question or a "
        "cropped region.]"
    )


def register_vision_tools(config: Config, *, get_provider_fn: Callable):
    """Register ``view_image``; called once after agent construction."""
    global _PERCEPTION_CTX
    _PERCEPTION_CTX = (config, get_provider_fn)

    async def view_image(
        image_path: str,
        question: str = (
            "Describe this image in detail. If it contains text or "
            "tables, transcribe them verbatim."
        ),
        slice_mode: str = "auto",
        region: str = "",
    ) -> str:
        global _view_image_calls
        _view_image_calls += 1

        from dashscope.acli.utils.paths import validate_path

        try:
            safe_path = validate_path(image_path)
        except (ValueError, OSError) as e:
            return f"Error: failed to read image: {e}"

        return await _perceive_image(
            config,
            get_provider_fn,
            safe_path,
            question,
            slice_mode,
            region,
        )

    registry.register(
        ToolDefinition(
            name="view_image",
            description=(
                "Analyze a local image file (screenshot, photo, chart, "
                "scan, render) with a vision model: OCR/transcribe text "
                "or digits, describe layout or content. Very tall/wide "
                "images are split automatically. Use whenever a task "
                "requires reading an image's content — including images "
                "you rendered yourself from data (plots, geometry, "
                "toolpaths) — instead of approximating the perception "
                "with terminal output or pixel scripting. For grid-like "
                "content (tables, digit grids), read the whole image "
                "first, then re-read it block by block with the region "
                "parameter: region is a real pixel crop (upscaled "
                "automatically), far more accurate than asking about a "
                "region of the full image. When every character matters "
                "(codes, serial numbers, mixed-case text), never trust "
                "one whole-image read of small or stylized text: "
                "verify it with region reads of a few characters each "
                "before committing to an answer, and overlap adjacent "
                "crops so no character sits on a crop boundary. When "
                "re-reads of one glyph keep disagreeing, more identical "
                "re-reads will not converge — the glyph is at the "
                "recognizer's noise floor: adjudicate with a feature "
                "question (a dot, serif, or enclosed counter present or "
                "absent) on a crop of only that glyph, and prefer the "
                "candidate that makes the assembled string read as "
                "coherent text over the one that does not. Overlapping "
                "crops also cross-check each other: characters shared by "
                "two crops must read the same in both, and if they do "
                "not, re-read the contested span before committing. And "
                "when you rendered the image yourself, ambiguous reads "
                "are a rendering problem, not a perception limit: "
                "re-render larger or higher-contrast and read the new "
                "render instead of re-reading the same pixels. Glyphs "
                "from confusable sets (0/O/o, 1/l/i/I, 5/S, 8/B, and "
                "same-shape letter case like cC, kK, pP) need "
                "feature-level confirmation — dot, serif, counter, "
                "relative height — not just repeated transcription: "
                "stylized fonts make wrong glyphs look unambiguous. "
                "Context is what corrupts stylized reads: word-shaped "
                "crops get pattern-completed into expected words, while "
                "a crop of one isolated glyph can only be read as "
                "pixels — settle contested characters glyph by glyph, "
                "and never let an expected word override what isolated "
                "glyphs clearly show."
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
                    "region": {
                        "type": "string",
                        "description": (
                            'Crop before reading: "x0,y0,x1,y1" as '
                            "fractions of width/height (0-1, origin "
                            "top-left), e.g. the middle horizontal "
                            'third is "0,0.333,1,0.667". Overrides '
                            "slicing. For dense grids/tables transcribe "
                            "a few cells per call; for exact-character "
                            "work (codes, labels, stylized or small "
                            "text) crop to a few characters per call. "
                            "When tiling a long string with several "
                            "crops, overlap adjacent crops: a character "
                            "at a crop boundary is exactly the one most "
                            "likely to be dropped or duplicated, so "
                            "every character must appear fully inside "
                            "at least one crop. Each region read samples "
                            "the vision model several times and shows "
                            "every answer when they disagree."
                        ),
                    },
                },
                "required": ["image_path"],
            },
        ),
    )
