# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import uuid
from http import HTTPStatus

from dashscope import FineTunes
from tests.unit.constants import TEST_JOB_ID
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer

# yapf: disable


class TestFineTuneRequest(MockServerBase):
    case_data = None

    @classmethod
    def setup_class(cls):
        cls.case_data = json.load(
            open('tests/data/fine_tune.json', 'r', encoding='utf-8'),
        )
        super().setup_class()

    def test_create_fine_tune_job(self, mock_server: MockServer):
        response_body = self.case_data['create_response']
        mock_server.responses.put(json.dumps(response_body))
        model = 'gpt'
        training_file_ids = 'training_001'
        validation_file_ids = 'validation_001'
        hyper_parameters = {
            'epochs': 10,
            'learning_rate': 0.001,
        }
        resp = FineTunes.call(
            model=model,
            training_file_ids=training_file_ids,
            validation_file_ids=validation_file_ids,
            hyper_parameters=hyper_parameters,
        )
        req = mock_server.requests.get(block=True)
        assert req['path'] == '/api/v1/fine-tunes'
        assert req['body']['model'] == model
        assert req['body']['training_file_ids'] == [training_file_ids]
        assert req['body']['validation_file_ids'] == [validation_file_ids]
        assert req['body']['hyper_parameters'] == hyper_parameters
        assert resp.output.job_id == response_body['output']['job_id']
        assert resp.output.status == response_body['output']['status']
        assert resp.output.hyper_parameters == {'learning_rate': '2e-5', 'n_epochs': 10, 'batch_size': 32}

    def test_create_fine_tune_job_with_files(self, mock_server: MockServer):
        response_body = self.case_data['create_multi_files_response']
        mock_server.responses.put(json.dumps(response_body))
        model = 'gpt'
        training_file_ids = ['training_001', 'training_002']
        validation_file_ids = ['validation_001', 'validation_002']
        hyper_parameters = {
                                  'epochs': 10,
                                  'learning_rate': 0.001,
        }
        resp = FineTunes.call(
            model=model,
            training_file_ids=training_file_ids,
            validation_file_ids=validation_file_ids,
            hyper_parameters=hyper_parameters,
        )
        req = mock_server.requests.get(block=True)
        assert req['path'] == '/api/v1/fine-tunes'
        assert req['body']['model'] == model
        assert req['body']['training_file_ids'] == training_file_ids
        assert req['body']['validation_file_ids'] == validation_file_ids
        assert req['body']['hyper_parameters'] == hyper_parameters
        assert resp.output.job_id == response_body['output']['job_id']
        assert resp.output.status == response_body['output']['status']
        assert resp.output.training_file_ids == training_file_ids
        assert resp.output.validation_file_ids == validation_file_ids
        assert resp.output.hyper_parameters == response_body['output']['hyper_parameters']

    def test_list_fine_tune_job(self, mock_server: MockServer):
        response_body = self.case_data['list_response']
        mock_server.responses.put(json.dumps(response_body))
        response = FineTunes.list(
            page_no=10,
            page_size=101,
        )
        req = mock_server.requests.get(block=True)
        assert req['path'] == '/api/v1/fine-tunes?page_no=10&page_size=101'
        assert len(response.output.jobs) == 2
        assert response.output.jobs[0].job_id == 'ft-202403261454-d8b4'

    def test_get_fine_tune_job(self, mock_server: MockServer):
        response_body = self.case_data['query_response']
        mock_server.responses.put(json.dumps(response_body))
        job_id = str(uuid.uuid4())
        response = FineTunes.get(job_id=job_id)
        req = mock_server.requests.get(block=True)
        assert req['path'] == f'/api/v1/fine-tunes/{job_id}'
        assert response.output.job_id == 'ft-202403261451-d26b'

    def test_delete_fine_tune_job(self, mock_server: MockServer):
        request_id = str(uuid.uuid4())
        response_body = '{"output": {"status": "success"}, "request_id": "%s", "code": null, "message": "", "usage": null}' % request_id  # noqa E501
        mock_server.responses.put(response_body)
        rsp = FineTunes.delete(TEST_JOB_ID)
        req = mock_server.requests.get(block=True)
        assert req['path'] == f'/api/v1/fine-tunes/{TEST_JOB_ID}'
        assert rsp.status_code == HTTPStatus.OK
        assert rsp.request_id == request_id

    def test_cancel_fine_tune_job(self, mock_server: MockServer):
        request_id = str(uuid.uuid4())
        response_body = '{"output": {"status": "success"}, "request_id": "%s", "code": null, "message": "", "usage": null}' % request_id  # noqa E501
        mock_server.responses.put(response_body)
        rsp = FineTunes.cancel(TEST_JOB_ID)
        req = mock_server.requests.get(block=True)
        assert req['path'] == f'/api/v1/fine-tunes/{TEST_JOB_ID}/cancel'
        assert rsp.status_code == HTTPStatus.OK
        assert rsp.request_id == request_id

    def test_stream_event(self, mock_server: MockServer):
        responses = FineTunes.stream_events(TEST_JOB_ID)
        idx = 0
        for rsp in responses:
            assert rsp.status_code == HTTPStatus.OK
            idx += 1
            print(rsp.output)
        assert idx == 10
