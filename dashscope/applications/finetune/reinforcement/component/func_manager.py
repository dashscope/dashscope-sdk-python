# -*- coding: utf-8 -*-
"""
component/func_manager.py

FuncManager - Unified management of parameter parsing and business
processing routing.

Responsibilities:
1. Retrieve corresponding parameter parser based on FunctionType (internal
fixed components)
2. Manage processor registration and retrieval (user-customizable)
3. Provide unified request handling with automatic parse→process pipeline
"""

import asyncio
import contextvars
import importlib
import inspect
from concurrent.futures import Executor
from functools import partial
from typing import Any, Dict, Optional, Type

from ..common.log import logger
from ..common.model_types import (
    FunctionType as FuncType,
)
from .data.base_data_model import (
    BaseDataModel,
)
from .observability.tracing import (
    is_tracing_enabled,
    trace_processor_process,
    async_trace_processor_process,
)
from .parser import (
    BaseRequestParser,
)
from .parser import (
    GroupRewardRequestParser,
)
from .parser import (
    RewardRequestParser,
)
from .parser import (
    RolloutRequestParser,
)
from .processor.abstract_processor import (  # noqa: E501  # pylint: disable=line-too-long
    AbstractProcessor,
)


class FuncManager:
    """
    Function service manager for unified parameter parsing and processing
    routing.

    Usage Example:
        >>> # Create with custom processor
        >>> processor = MyRewardProcessor()
        >>> manager = FuncManager(FuncType.REWARD, processor=processor)
        >>> result = manager.process({"rollout_id": "ro-1", "content": "..."})
    """

    # Internal fixed parser mapping (non-customizable)
    _PARSER_MAP: Dict[FuncType, Type[BaseRequestParser]] = {
        FuncType.REWARD: RewardRequestParser,
        FuncType.ROLLOUT: RolloutRequestParser,
        FuncType.GROUP_REWARD: GroupRewardRequestParser,
    }

    def __init__(
        self,
        func_type: FuncType,
        processor: AbstractProcessor,
        *,
        observe: bool = True,
        executor: Optional[Executor] = None,
    ):
        """
        Initialize FuncManager.

        Args:
            func_type: Function type determining parser
            processor: Processor instance (required)
            observe: Whether to enable OpenTelemetry tracing on process()
            calls (default: True).
                     Effective only when ENABLE_TRAJECTORY env var is set.
                     Set to False to disable tracing for this manager.

        Raises:
            ValueError: If processor is None
            TypeError: If processor is not a AbstractProcessor subclass
        """
        if processor is None:
            raise ValueError(
                f"Processor is required for FuncManager. "
                f"Please provide a processor instance for func_type"
                f"={func_type.value}",
            )
        if not isinstance(processor, AbstractProcessor):
            raise TypeError(
                f"Processor must be a AbstractProcessor subclass, got"
                f" {type(processor)}",
            )

        self._func_type = func_type
        self._parser = self._create_parser(func_type)
        self._processor = processor
        self._observe = observe
        self.executor = executor

        logger.info(
            f"[FuncManager] initialized | func_type={func_type.value} | "
            f"parser={type(self._parser).__name__} | "
            f"processor={type(self._processor).__name__} | "
            f"observe={observe}",
        )

    def set_executor(self, executor: Optional[Executor]) -> None:
        """Set the executor used to offload sync processors (optional)."""
        self.executor = executor

    @property
    def func_type(self) -> FuncType:
        """Get current function type."""
        return self._func_type

    @property
    def parser(self) -> BaseRequestParser:
        """Get current parameter parser."""
        return self._parser

    @property
    def processor(self) -> AbstractProcessor:
        """Get current business processor."""
        return self._processor

    def _create_parser(self, func_type: FuncType) -> BaseRequestParser:
        """Create parser instance (internal component, non-customizable)."""
        parser_cls = self._PARSER_MAP.get(func_type)
        if parser_cls is None:
            raise ValueError(
                f"No parser registered for func_type={func_type.value}",
            )
        return parser_cls()

    def register_processor(self, processor: AbstractProcessor) -> None:
        """
        Register custom business processor.

        Args:
            processor: Custom processor instance (must subclass
            AbstractProcessor)

        Raises:
            TypeError: If processor is not a AbstractProcessor subclass
        """
        if not isinstance(processor, AbstractProcessor):
            raise TypeError(
                f"Processor must be a AbstractProcessor subclass, got"
                f" {type(processor)}",
            )
        self._processor = processor
        logger.info(
            f"[FuncManager] registered custom processor:"
            f" {type(processor).__name__}",
        )

    def parses(self, raw_data: Dict[str, Any]) -> BaseDataModel:
        """
        Parse raw request data.

        Args:
            raw_data: Raw dictionary from HTTP request body

        Returns:
            Parsed ProcessorInput object
        """
        return self._parser.parse(raw_data)

    async def _run_sync(self, fn) -> Any:
        # Avoid blocking the uvicorn event loop when processor.process is sync.
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(self.executor, ctx.run, fn)

    async def setups(self):
        try:
            if inspect.iscoroutinefunction(self._processor.setup):
                await self._processor.setup()
            else:
                await self._run_sync(self._processor.setup)
        except Exception as e:
            logger.error(f"Processor setup failed: {e}")
            raise

    async def processes(self, input_data: BaseDataModel) -> Any:
        """
        Execute business processing.

        When ``observe=True`` (default) and tracing is enabled via env vars
        (``ENABLE_TRAJECTORY``),
        wraps the processor call in an OpenTelemetry span.

        Supports both sync and async processors transparently.

        Args:
            input_data: Parsed ProcessorInput object

        Returns:
            Processing result
        """
        if self._observe and is_tracing_enabled():
            if inspect.iscoroutinefunction(self._processor.process):
                return await async_trace_processor_process(
                    self._func_type,
                    self._processor,
                    input_data,
                )
            return await self._run_sync(
                partial(
                    trace_processor_process,
                    self._func_type,
                    self._processor,
                    input_data,
                ),
            )

        if inspect.iscoroutinefunction(self._processor.process):
            return await self._processor.process(input_data)
        return await self._run_sync(
            partial(self._processor.process, input_data),
        )

    def execute(self, raw_data: Dict[str, Any]) -> Any:
        """
        Complete processing pipeline: parse + execute.

        Args:
            raw_data: Raw dictionary from HTTP request body

        Returns:
            Processing result
        """
        input_data = self.parse(raw_data)
        return self.process(input_data)

    @classmethod
    def create_from_env(
        cls,
        func_type: FuncType,
        processor_class_path: str,
    ) -> "FuncManager":
        """
        Create FuncManager instance from environment configuration.

        Args:
            func_type: Function type
            processor_class_path: Full path to processor class (e.g.,
            "my_module.MyProcessor")

        Returns:
            FuncManager instance

        Raises:
            ValueError: If processor_class_path is None or empty
            ImportError: If class cannot be imported
            TypeError: If class is not a AbstractProcessor subclass
        """
        if not processor_class_path:
            raise ValueError(
                f"processor_class_path is required for "
                f"FuncManager.create_from_env(). "
                f"Please provide a processor class path for func_type"
                f"={func_type.value}",
            )

        logger.info(f"[FuncManager] loading processor: {processor_class_path}")

        # Dynamically load processor class
        processor = cls._load_processor_class(processor_class_path)
        return cls(func_type, processor=processor)

    @staticmethod
    def _load_processor_class(class_path: str) -> AbstractProcessor:
        """
        Dynamically load and instantiate processor class.

        Args:
            class_path: Full class path (e.g.,
            "my_module.sub_module.MyProcessor")

        Returns:
            Processor instance

        Raises:
            ImportError: For module/class not found
            TypeError: For invalid class type
        """
        parts = class_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(
                f"Invalid class path '{class_path}'. Expected "
                f"'module.submodule.ClassName'",
            )

        module_path, class_name = parts

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Module import failed: '{module_path}' - " f"{e}",
            ) from e

        try:
            processor_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(
                f"Class '{class_name}' not found in '{module_path}'",
            ) from exc

        # Instantiate
        try:
            processor = processor_cls()
        except Exception as e:
            raise TypeError(
                f"Instantiation failed for '{class_path}': {e}",
            ) from e

        # Type validation
        if not isinstance(processor, AbstractProcessor):
            raise TypeError(
                f"Class '{class_path}' must inherit AbstractProcessor, "
                f"got: {processor_cls.__bases__}",
            )

        return processor
