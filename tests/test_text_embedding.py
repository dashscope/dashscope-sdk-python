# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

from dashscope import TextEmbedding
from tests.mock_request_base import MockRequestBase


class TestTextEmbeddingRequest(MockRequestBase):
    def test_call_with_string(self, http_server):
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v3,
            input="hello",
        )
        assert resp.status_code == HTTPStatus.OK
        assert len(resp.output["embeddings"]) == 1

    def test_call_with_list_str(self, http_server):
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v3,
            input=["hello", "world"],
        )
        assert resp.status_code == HTTPStatus.OK
        assert len(resp.output["embeddings"]) == 1

    def test_call_with_opened_file(self, http_server):
        with open("tests/data/multi_line.txt") as f:
            response = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=f,
            )
            assert response.status_code == HTTPStatus.OK
            assert len(response.output["embeddings"]) == 1
