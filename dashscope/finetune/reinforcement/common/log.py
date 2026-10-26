# -*- coding: utf-8 -*-
import logging
import sys

from dashscope.finetune.reinforcement.common.constants import (
    LOGGER_NAME,
    LOG_LEVEL,
)


def setup_logger():
    """Initialize and configure the logger with value masking."""
    _logger = logging.getLogger(LOGGER_NAME)
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    _logger.setLevel(level_map.get(LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    handler.setFormatter(formatter)

    _logger.addHandler(handler)
    _logger.propagate = False
    return _logger


logger = setup_logger()
