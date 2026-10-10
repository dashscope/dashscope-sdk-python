# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import aiohttp

import dashscope
from dashscope.api_entities.aio_session import (
    get_shared_aio_session,
    send_with_retry_async,
)
from dashscope.client.base_api import CreateMixin, _workspace_header
from dashscope.common.api_key import get_default_api_key
from dashscope.common.base_type import BaseObjectMixin
from dashscope.common.constants import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_KEYWORD,
)
from dashscope.common.env import resolve_base_url
from dashscope.common.error import (
    InputRequired,
    InvalidParameter,
    ModelRequired,
)
from dashscope.common.logging import logger
from dashscope.common.utils import (
    default_headers,
    get_api_module,
    join_url,
)

__all__ = [
    "DecisionModel",
    "AioDecisionModel",
    "DecisionAnswer",
    "DecisionModelResponse",
    "DecisionUsage",
]

SYSTEMONE_PATH = "systemone"


@dataclass(init=False)
class DecisionUsage(BaseObjectMixin):
    input_tokens: int
    """Number of tokens in the request."""

    def __init__(self, **kwargs):  # pylint: disable=useless-parent-delegation
        super().__init__(**kwargs)


@dataclass(init=False)
class DecisionAnswer(BaseObjectMixin):
    type: str
    """Answer type, same as the question type(choice/noul/score)."""

    choice: Optional[str] = None
    """choice type: the selected option name."""

    noul: Optional[float] = None
    """noul type: P(yes), 0 means no and 1 means yes."""

    score: Optional[float] = None
    """score type: probability weighted expectation of the level index,
    may fall between two levels(e.g. 1.43)."""

    probabilities: Optional[Dict[str, float]] = None
    """Probability of each option/level, sums to 1."""

    confidence: Optional[float] = None
    """Confidence of the answer(choice/score types)."""

    legend: Optional[Dict[str, str]] = None
    """score type: level index -> level description."""

    def __init__(self, **kwargs):  # pylint: disable=useless-parent-delegation
        super().__init__(**kwargs)


@dataclass(init=False)
class DecisionModelResponse(BaseObjectMixin):
    model: str = ""
    """The model name echo."""

    request_id: str = ""
    """The request id."""

    status_code: int = HTTPStatus.OK
    """The http status code, 200 means success."""

    answers: Optional[Dict[str, DecisionAnswer]] = None
    """Answers keyed by the question id of the request."""

    usage: Optional[DecisionUsage] = None
    """Token usage."""

    latency_ms: Optional[float] = None
    """Server side latency in milliseconds."""

    code: str = ""
    """Error code when the request failed."""

    message: str = ""
    """Error message when the request failed."""

    def __init__(self, **kwargs):
        answers = kwargs.pop("answers", None)
        usage = kwargs.pop("usage", None)
        super().__init__(**kwargs)
        if answers is not None:
            self.answers = {
                key: DecisionAnswer(**answer)
                if isinstance(answer, dict)
                else answer
                for key, answer in answers.items()
            }
        if usage is not None:
            self.usage = (
                DecisionUsage(**usage) if isinstance(usage, dict) else usage
            )


def _validate_parameters(
    model: str,
    state: Union[str, Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]],
) -> None:
    if model is None or not model:
        raise ModelRequired("Model is required!")
    if state is None:
        raise InputRequired("state is required!")
    if questions is None or not questions:
        raise InputRequired("questions is required!")


class DecisionModel(CreateMixin):
    """API for decision models(TypeSafe System One).

    One forward pass returns classification, scoring and yes/no
    judgement with probability distributions and confidence, without
    generating text.
    """

    SUB_PATH = ""

    class Models:
        decision_model_preview = "decision-model-preview"

    @classmethod
    def call(  # type: ignore[override]  # pylint: disable=arguments-renamed
        cls,
        model: str,
        state: Union[str, Dict[str, Any]],
        questions: Dict[str, Dict[str, Any]],
        api_key: str = None,
        workspace: str = None,
        extra_headers: Dict = None,
        **kwargs,
    ) -> DecisionModelResponse:
        """Call decision model service.

        Args:
            model (str): The requested model,
                such as decision-model-preview.
            state (Union[str, Dict[str, Any]]): The business state to
                decide on: ticket text, conversation or a structured
                object(objects are serialized before sending).
            questions (Dict[str, Dict[str, Any]]): The question table.
                Key is a caller defined question id, value is a question
                object with the following fields:

                - type (str): ``choice`` (pick one of many), ``noul``
                  (yes/no judgement) or ``score`` (ordered scale).
                - instructions (str): The question description or
                  judgement criteria.
                - criteria: ``choice``: mapping of option name to
                  option description(1-255 items); ``noul``: optional
                  ``{"true": ..., "false": ...}`` descriptions;
                  ``score``: level descriptions ordered from low to
                  high(2-10 levels).

                example::

                    {
                        "department": {
                            "type": "choice",
                            "instructions": "Which team should handle?",
                            "criteria": {
                                "billing": "payment and refund",
                                "technical": "product defect",
                            },
                        },
                        "escalate": {
                            "type": "noul",
                            "instructions": "Notify the on-call staff?",
                        },
                        "severity": {
                            "type": "score",
                            "instructions": "How severe is the problem?",
                            "criteria": ["minor", "major", "critical"],
                        },
                    }
            api_key (str, optional): The DashScope api key, if None,
                will get by default rule. Defaults to None.
            workspace (str, optional): The DashScope workspace id.
                Defaults to None.
            extra_headers (Dict, optional): Extra http headers.
                Defaults to None.
            **kwargs:
                request_timeout: set request timeout in seconds.

        Raises:
            InputRequired: The state and questions are required.
            ModelRequired: The model is required.

        Returns:
            DecisionModelResponse: The decision result, check
            ``status_code == 200`` before consuming ``answers``.
        """
        _validate_parameters(model, state, questions)
        if kwargs.pop("stream", None):
            raise InvalidParameter(
                "Decision model does not support stream output",
            )
        data = {
            "model": model,
            "state": state,
            "questions": questions,
        }
        if extra_headers is not None and extra_headers:
            kwargs["headers"] = {
                **kwargs.get("headers", {}),
                **extra_headers,
            }
        base_address = kwargs.pop(
            "base_address",
            dashscope.base_compatible_api_url,
        )
        response = super().call(
            data=data,
            path=SYSTEMONE_PATH,
            base_address=base_address,
            api_key=api_key,
            flattened_output=True,
            workspace=workspace,
            **kwargs,
        )
        return DecisionModelResponse(**response)


class AioDecisionModel:
    """Async API for decision models(TypeSafe System One)."""

    Models = DecisionModel.Models

    @classmethod
    async def call(
        cls,
        model: str,
        state: Union[str, Dict[str, Any]],
        questions: Dict[str, Dict[str, Any]],
        api_key: str = None,
        workspace: str = None,
        extra_headers: Dict = None,
        **kwargs,
    ) -> DecisionModelResponse:
        """Call decision model service asynchronously.

        Args:
            model (str): The requested model,
                such as decision-model-preview.
            state (Union[str, Dict[str, Any]]): The business state to
                decide on: ticket text, conversation or a structured
                object(objects are serialized before sending).
            questions (Dict[str, Dict[str, Any]]): The question table,
                see :meth:`DecisionModel.call` for the schema.
            api_key (str, optional): The DashScope api key, if None,
                will get by default rule. Defaults to None.
            workspace (str, optional): The DashScope workspace id.
                Defaults to None.
            extra_headers (Dict, optional): Extra http headers.
                Defaults to None.
            **kwargs:
                request_timeout: set request timeout in seconds.

        Raises:
            InputRequired: The state and questions are required.
            ModelRequired: The model is required.

        Returns:
            DecisionModelResponse: The decision result, check
            ``status_code == 200`` before consuming ``answers``.
        """
        _validate_parameters(model, state, questions)
        if kwargs.pop("stream", None):
            raise InvalidParameter(
                "Decision model does not support stream output",
            )
        data = {
            "model": model,
            "state": state,
            "questions": questions,
        }
        base_url = kwargs.pop("base_address", None)
        if base_url:
            base_url = resolve_base_url(base_url, workspace)
        else:
            base_url = resolve_base_url(
                dashscope.base_compatible_api_url,
                workspace,
            )
        url = join_url(base_url, SYSTEMONE_PATH)
        timeout = kwargs.pop(
            REQUEST_TIMEOUT_KEYWORD,
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        if api_key is None:
            api_key = get_default_api_key()
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            **_workspace_header(workspace),
            **default_headers(api_key, module=get_api_module(cls.__module__)),
        }
        if extra_headers is not None and extra_headers:
            headers = {**headers, **extra_headers}
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        session = await get_shared_aio_session()

        async def _send() -> aiohttp.ClientResponse:
            return await session.post(
                url,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )

        logger.debug("Starting request: %s", url)
        response = await send_with_retry_async(_send)
        async with response:
            if "application/json" in response.content_type:
                json_content = await response.json()
            else:
                json_content = {"message": await response.text()}
            json_content["status_code"] = response.status
            return DecisionModelResponse(**json_content)
