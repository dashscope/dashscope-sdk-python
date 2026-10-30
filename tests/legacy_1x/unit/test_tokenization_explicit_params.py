# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Test explicit parameters for Tokenization class."""

# pylint: disable=protected-access,unused-variable
import pytest
from dashscope.tokenizers.tokenization import Tokenization


class TestTokenizationExplicitParams:
    """Test explicit parameters in Tokenization class."""

    def test_build_llm_parameters_with_enable_search(self):
        """Test _build_llm_parameters with enable_search parameter."""
        # Test with enable_search=True for qwen model
        input_data, parameters = Tokenization._build_llm_parameters(
            model="qwen-turbo",
            prompt="test prompt",
            history=None,
            messages=None,
            enable_search=True,
        )

        # Verify enable_search is in parameters
        assert parameters["enable_search"] is True
        # Verify input is correctly built
        assert input_data["prompt"] == "test prompt"

    def test_build_llm_parameters_with_enable_search_false(self):
        """Test _build_llm_parameters with enable_search=False."""
        input_data, parameters = Tokenization._build_llm_parameters(
            model="qwen-turbo",
            prompt="test prompt",
            history=None,
            messages=None,
            enable_search=False,
        )

        # Verify enable_search is not in parameters when False
        assert "enable_search" not in parameters

    def test_build_llm_parameters_with_customized_model_id(self):
        """Test _build_llm_parameters with customized_model_id parameter."""
        # Test with customized_model_id for bailian model
        input_data, parameters = Tokenization._build_llm_parameters(
            model="bailian-test",
            prompt="test prompt",
            history=None,
            messages=None,
            customized_model_id="custom-model-123",
        )

        # Verify customized_model_id is in input
        assert input_data["customized_model_id"] == "custom-model-123"

    def test_build_llm_parameters_without_customized_model_id(self):
        """Test _build_llm_parameters without customized_model_id."""
        # Should raise InputRequired error for bailian model
        with pytest.raises(Exception) as exc_info:
            Tokenization._build_llm_parameters(
                model="bailian-test",
                prompt="test prompt",
                history=None,
                messages=None,
                customized_model_id=None,
            )

        assert "customized_model_id is required" in str(exc_info.value)

    def test_build_llm_parameters_with_messages(self):
        """Test _build_llm_parameters with messages parameter."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        input_data, parameters = Tokenization._build_llm_parameters(
            model="qwen-turbo",
            prompt=None,
            history=None,
            messages=messages,
            enable_search=True,
        )

        # Verify messages are correctly built
        assert "messages" in input_data
        assert len(input_data["messages"]) == 2
        # Verify enable_search is in parameters
        assert parameters["enable_search"] is True

    def test_build_llm_parameters_with_extra_kwargs(self):
        """Test _build_llm_parameters with extra kwargs."""
        input_data, parameters = Tokenization._build_llm_parameters(
            model="qwen-turbo",
            prompt="test prompt",
            history=None,
            messages=None,
            enable_search=True,
            custom_param="custom_value",
            another_param=123,
        )

        # Verify explicit parameter
        assert parameters["enable_search"] is True
        # Verify extra kwargs are passed through
        assert parameters["custom_param"] == "custom_value"
        assert parameters["another_param"] == 123

    def test_build_llm_parameters_non_qwen_non_bailian(self):
        """Test _build_llm_parameters with non-qwen, non-bailian model."""
        input_data, parameters = Tokenization._build_llm_parameters(
            model="other-model",
            prompt="test prompt",
            history=None,
            messages=None,
            enable_search=True,  # Should be ignored for non-qwen models
        )

        # Verify enable_search is not in parameters for non-qwen models
        assert "enable_search" not in parameters
        # Verify input is correctly built
        assert input_data["prompt"] == "test prompt"

    def test_build_llm_parameters_with_history_deprecated(self):
        """Test _build_llm_parameters with deprecated history parameter."""
        history = [
            {"user": "Hello"},
            {"bot": "Hi"},
        ]

        input_data, parameters = Tokenization._build_llm_parameters(
            model="qwen-turbo",
            prompt="test prompt",
            history=history,
            messages=None,
            enable_search=True,
        )

        # Verify history is used (deprecated but still supported)
        assert "history" in input_data
        assert input_data["prompt"] == "test prompt"
        # Verify enable_search is in parameters
        assert parameters["enable_search"] is True
