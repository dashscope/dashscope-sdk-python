# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import json
import logging
from http import HTTPStatus
from urllib.parse import urlparse

import pytest

import dashscope
from dashscope.audio.asr import AsrPhraseManager
from tests.unit.constants import TEST_JOB_ID
from tests.unit.mock_request_base import MockRequestBase

logger = logging.getLogger("dashscope")
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# add formatter to ch
console_handler.setFormatter(formatter)

# add ch to logger
logger.addHandler(console_handler)


class TestAsrPhrases(MockRequestBase):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = "asr"
        cls.phrase = {"黄鸡": 5}
        cls.update_phrase = {"黄鸡": 2, "红鸡": 1}
        cls.phrase_id = TEST_JOB_ID

    def test_create_phrases(self, http_server):
        result = AsrPhraseManager.create_phrases(
            model=self.model,
            phrases=self.phrase,
            headers={"X-Request-Id": "empty_file_ids"},
        )

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        assert result.output is not None
        assert result.output["finetuned_output"] is not None
        assert len(result.output["finetuned_output"]) > 0
        self.phrase_id = result.output["finetuned_output"]

    def test_update_phrases(self, http_server):
        result = AsrPhraseManager.update_phrases(
            model=self.model,
            phrase_id=self.phrase_id,
            phrases=self.update_phrase,
            headers={"X-Request-Id": "empty_file_ids"},
        )
        assert result is not None
        assert result.status_code == HTTPStatus.OK
        assert result.output["finetuned_output"] is not None
        assert len(result.output["finetuned_output"]) > 0

    def test_query_phrases(self, http_server):
        result = AsrPhraseManager.query_phrases(phrase_id=self.phrase_id)
        assert result is not None
        assert result.status_code == HTTPStatus.OK
        assert result.output["finetuned_output"] is not None
        assert len(result.output["finetuned_output"]) > 0
        assert result.output["model"] is not None
        assert len(result.output["model"]) > 0

    def test_list_phrases(self, http_server):
        result = AsrPhraseManager.list_phrases(page=1, page_size=10)
        assert result is not None
        assert result.status_code == HTTPStatus.OK
        assert result.output["finetuned_outputs"] is not None
        assert len(result.output["finetuned_outputs"]) > 0

    def test_delete_phrases(self, http_server):
        result = AsrPhraseManager.delete_phrases(phrase_id=self.phrase_id)
        assert result is not None
        assert result.status_code == HTTPStatus.OK
        assert result.output["finetuned_output"] is not None
        assert len(result.output["finetuned_output"]) > 0


def str2bool(str):
    return True if str.lower() == "true" else False


def complete_url(url: str) -> str:
    parsed = urlparse(url)
    base_url = "".join([parsed.scheme, "://", parsed.netloc])
    dashscope.base_websocket_api_url = "/".join(
        [base_url, "api-ws", dashscope.common.env.api_version, "inference"],
    )
    dashscope.base_http_api_url = url = "/".join(
        [base_url, "api", dashscope.common.env.api_version],
    )
    print("Set base_websocket_api_url: ", dashscope.base_websocket_api_url)
    print("Set base_http_api_url: ", dashscope.base_http_api_url)


def phrases(
    model,
    phrase_id: str,
    phrases: dict,
    page: int,
    page_size: int,
    delete: bool,
):
    print("phrase_id: ", phrase_id)
    print("phrase: ", phrases)
    print("delete flag: ", delete)
    if len(phrases) > 0:
        if phrase_id is not None:
            print("Update phrases -->")
            return AsrPhraseManager.update_phrases(
                model=model,
                phrase_id=phrase_id,
                phrases=phrases,
            )
        else:
            print("Create phrases -->")
            return AsrPhraseManager.create_phrases(
                model=model,
                phrases=phrases,
            )
    else:
        if delete:
            print("Delete phrases -->")
            return AsrPhraseManager.delete_phrases(phrase_id=phrase_id)
        else:
            if phrase_id is not None:
                print("Query phrases -->")
                return AsrPhraseManager.query_phrases(phrase_id=phrase_id)
            if page is not None and page_size is not None:
                print(
                    "List phrases page %d page_size %d -->"
                    % (page, page_size),
                )
                return AsrPhraseManager.list_phrases(
                    page=page,
                    page_size=page_size,
                )


@pytest.mark.skip
def test_by_user():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="paraformer-realtime-v1")
    parser.add_argument("--phrase", type=str, default="")
    parser.add_argument("--phrase_id", type=str, default=None)
    parser.add_argument("--delete", type=str2bool, default="False")
    parser.add_argument("--page", type=int, default=None)
    parser.add_argument("--page_size", type=int, default=None)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_url", type=str)
    args = parser.parse_args()

    if args.api_key is not None:
        dashscope.api_key = args.api_key
    if args.base_url is not None:
        complete_url(args.base_url)

    phrase_dict = {}
    if len(args.phrase) > 0:
        phrase_dict = json.loads(args.phrase)
    resp = phrases(
        model=args.model,
        phrase_id=args.phrase_id,
        phrases=phrase_dict,
        page=args.page,
        page_size=args.page_size,
        delete=args.delete,
    )
    if resp.status_code == HTTPStatus.OK:
        print("Response of phrases: ", resp)
        if resp is not None and resp.output is not None:
            output = resp.output
            print("\nGet output: %s\n" % (str(output)))

            if (
                "finetuned_output" in output
                and output["finetuned_output"] is not None
            ):
                print("Get phrase_id: %s" % (output["finetuned_output"]))
            if "job_id" in output and output["job_id"] is not None:
                print("Get job_id: %s" % (output["job_id"]))
            if "create_time" in output and output["create_time"] is not None:
                print("Get create_time: %s" % (output["create_time"]))
            if "model" in output and output["model"] is not None:
                print("Get model_id: %s" % (output["model"]))
            if "output_type" in output and output["output_type"] is not None:
                print("Get output_type: %s" % (output["output_type"]))

            if (
                "finetuned_outputs" in output
                and output["finetuned_outputs"] is not None
            ):
                outputs = output["finetuned_outputs"]
                print(
                    "Get %d info from page_no:%d page_size:%d total:%d ->"
                    % (
                        len(outputs),
                        output["page_no"],
                        output["page_size"],
                        output["total"],
                    ),
                )
                for item in outputs:
                    print("  get phrase_id: %s" % (item["finetuned_output"]))
                    print("  get job_id: %s" % (item["job_id"]))
                    print("  get create_time: %s" % (item["create_time"]))
                    print("  get model_id: %s" % (item["model"]))
                    print("  get output_type: %s\n" % (item["output_type"]))
    else:
        print(
            "ERROR, status_code:%d, code_message:%s, error_message:%s"
            % (resp.status_code, resp.code, resp.message),
        )


if __name__ == "__main__":
    test_by_user()
