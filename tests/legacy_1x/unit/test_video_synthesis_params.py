# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from http import HTTPStatus

from dashscope import VideoSynthesis
from tests.legacy_1x.unit.mock_request_base import MockServerBase
from tests.legacy_1x.unit.mock_server import MockServer


class TestVideoSynthesis(MockServerBase):
    """Test cases for VideoSynthesis new parameters."""

    text_response_obj = {
        "status_code": 200,
        "request_id": "effd2cb1-1a8c-9f18-9a49-a396f673bd40",
        "code": "",
        "message": "",
        "output": {
            "task_id": "test_task_123",
            "task_status": "SUCCEEDED",
            "video_url": "https://example.com/video.mp4",
        },
        "usage": {
            "video_count": 1,
        },
    }

    def test_with_size_parameter(self, mock_server: MockServer):
        """Test size parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            size="1280*720",
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["size"] == "1280*720"
        assert response.status_code == HTTPStatus.OK

    def test_with_duration_parameter(self, mock_server: MockServer):
        """Test duration parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            duration=10,
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["duration"] == 10
        assert response.status_code == HTTPStatus.OK

    def test_with_seed_parameter(self, mock_server: MockServer):
        """Test seed parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            seed=42,
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["seed"] == 42
        assert response.status_code == HTTPStatus.OK

    def test_with_prompt_extend_parameter(self, mock_server: MockServer):
        """Test prompt_extend parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            prompt_extend=True,
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["prompt_extend"] is True
        assert response.status_code == HTTPStatus.OK

    def test_with_watermark_parameter(self, mock_server: MockServer):
        """Test watermark parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            watermark=True,
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["watermark"] is True
        assert response.status_code == HTTPStatus.OK

    def test_with_resolution_parameter(self, mock_server: MockServer):
        """Test resolution parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            resolution="1080P",
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["resolution"] == "1080P"
        assert response.status_code == HTTPStatus.OK

    def test_with_ratio_parameter(self, mock_server: MockServer):
        """Test ratio parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            ratio="16:9",
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["ratio"] == "16:9"
        assert response.status_code == HTTPStatus.OK

    def test_with_shot_type_parameter(self, mock_server: MockServer):
        """Test shot_type parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            shot_type="multi",
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["shot_type"] == "multi"
        assert response.status_code == HTTPStatus.OK

    def test_with_audio_setting_parameter(self, mock_server: MockServer):
        """Test audio_setting parameter is correctly passed."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            audio_setting="auto",
        )

        req = mock_server.requests.get(block=True)
        assert req["parameters"]["audio_setting"] == "auto"
        assert response.status_code == HTTPStatus.OK

    def test_with_all_new_parameters(self, mock_server: MockServer):
        """Test all new parameters together."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            size="1280*720",
            duration=10,
            seed=42,
            prompt_extend=True,
            watermark=False,
            resolution="720P",
            ratio="16:9",
            shot_type="single",
            audio_setting="origin",
        )

        req = mock_server.requests.get(block=True)
        params = req["parameters"]

        assert params["size"] == "1280*720"
        assert params["duration"] == 10
        assert params["seed"] == 42
        assert params["prompt_extend"] is True
        assert params["watermark"] is False
        assert params["resolution"] == "720P"
        assert params["ratio"] == "16:9"
        assert params["shot_type"] == "single"
        assert params["audio_setting"] == "origin"

        assert response.status_code == HTTPStatus.OK

    def test_with_none_parameters_not_included(self, mock_server: MockServer):
        """Test that None parameters are not included in request."""
        response_str = json.dumps(TestVideoSynthesis.text_response_obj)
        mock_server.responses.put(response_str)

        response = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_2_1_t2v_plus,
            prompt="一只小猫在奔跑",
            size=None,
            duration=None,
            seed=None,
        )

        req = mock_server.requests.get(block=True)
        params = req["parameters"]

        # None parameters should not be in the request
        assert "size" not in params
        assert "duration" not in params
        assert "seed" not in params

        assert response.status_code == HTTPStatus.OK
