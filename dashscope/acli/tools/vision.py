# -*- coding: utf-8 -*-
"""Built-in vision tool — let the agent analyze local images itself.

Previously the only multimodal entry point was user-input ``@path``
expansion, so after ``scrape_web_screenshot`` / ``camera_capture`` the
model could not see the saved image and sometimes echoed the
user-facing ``@path`` hint verbatim, ending the turn prematurely.
``view_image`` closes that gap by routing the image to a
vision-capable model on the active provider.
"""

from __future__ import annotations

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


def register_vision_tools(config: Config, *, get_provider_fn: Callable):
    """Register ``view_image``; called once after agent construction."""

    async def view_image(
        image_path: str,
        question: str = (
            "Describe this image in detail. If it contains text or "
            "tables, transcribe them verbatim."
        ),
    ) -> str:
        from dashscope.acli.utils.images import image_to_data_url
        from dashscope.acli.utils.paths import validate_path

        try:
            safe_path = validate_path(image_path)
            data_url = image_to_data_url(safe_path)
        except (ValueError, OSError) as e:
            return f"Error: failed to read image: {e}"

        model = pick_vision_model(config)
        provider = get_provider_fn(
            config.provider,
            model,
            config.api_key,
            base_url=config.base_url or None,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        try:
            response = await provider.chat(messages)
        except Exception as e:  # pylint: disable=broad-except
            return f"Error: vision call failed: {e}"
        return response.content or "(vision model returned no content)"

    registry.register(
        ToolDefinition(
            name="view_image",
            description=(
                "Analyze a local image file (screenshot, photo, chart) "
                "with a vision model: OCR/transcribe text, describe "
                "layout or content. Use after scrape_web_screenshot or "
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
                },
                "required": ["image_path"],
            },
        ),
    )
