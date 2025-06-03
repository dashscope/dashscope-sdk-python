# Copyright (c) Alibaba, Inc. and its affiliates.

import ssl
from typing import Optional
import json
from http import HTTPStatus

import aiohttp
import requests
from urllib3.util.ssl_ import create_urllib3_context

from dashscope.api_entities.base_request import AioBaseRequest
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.common.constants import (DEFAULT_REQUEST_TIMEOUT_SECONDS,
                                        SSE_CONTENT_TYPE, HTTPMethod)
from dashscope.common.error import UnsupportedHTTPMethod
from dashscope.common.logging import logger
from dashscope.common.utils import (_handle_aio_stream,
                                    _handle_aiohttp_failed_response,
                                    _handle_http_failed_response,
                                    _handle_stream)


class HttpRequest(AioBaseRequest):
    # 类级别的连接池和SSL上下文缓存
    _session_pool = {}
    _aio_session_pool = {}
    _ssl_context = None
    _aio_ssl_context = None

    @classmethod
    def _init_ssl_context(cls):
        """初始化优化的SSL上下文"""
        if cls._ssl_context is None:
            # 同步请求的SSL上下文
            cls._ssl_context = create_urllib3_context()
            # 优化配置
            cls._ssl_context.options |= ssl.OP_NO_COMPRESSION
            cls._ssl_context.verify_mode = ssl.CERT_REQUIRED

            # 异步请求的SSL上下文
            cls._aio_ssl_context = ssl.create_default_context()
            cls._aio_ssl_context.options |= ssl.OP_NO_COMPRESSION
            cls._aio_ssl_context.verify_mode = ssl.CERT_REQUIRED

    def __init__(self,
                 url: str,
                 api_key: str,
                 http_method: str,
                 stream: bool = True,
                 async_request: bool = False,
                 query: bool = False,
                 timeout: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
                 task_id: Optional[str] = None,
                 flattened_output: bool = False) -> None:
        self._init_ssl_context()  # 确保SSL上下文已初始化

        super().__init__()
        self.url = url
        self.flattened_output = flattened_output
        self.async_request = async_request
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            **self.headers,
        }
        self.query = query

        if self.async_request and not self.query:
            self.headers['X-DashScope-Async'] = 'enable'

        self.method = http_method
        if self.method == HTTPMethod.POST:
            self.headers['Content-Type'] = 'application/json'

        self.stream = stream
        if self.stream:
            self.headers.update({
                'Accept': SSE_CONTENT_TYPE,
                'X-Accel-Buffering': 'no',
                'X-DashScope-SSE': 'enable'
            })

        if self.query and task_id:
            self.url = self.url.replace('api', 'api-task') + f'{task_id}'

        self.timeout = timeout or DEFAULT_REQUEST_TIMEOUT_SECONDS

    def _get_session(self) -> requests.Session:
        session_key = f"sync_{self.timeout}"
        if session_key not in self._session_pool:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=3,
                pool_connections=10,
                pool_maxsize=100
            )
            # 新版设置SSL上下文的方式
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            # 添加SSL配置
            if hasattr(adapter, 'init_poolmanager'):  # 新版适配
                adapter.init_poolmanager(
                    connections=10,
                    maxsize=100,
                    ssl_context=self._ssl_context
                )
            self._session_pool[session_key] = session
        return self._session_pool[session_key]

    async def _get_aio_session(self) -> aiohttp.ClientSession:
        """获取或创建异步会话"""
        session_key = f"aio_{self.timeout}"
        if session_key not in self._aio_session_pool:
            connector = aiohttp.TCPConnector(
                ssl_context=self._aio_ssl_context,
                limit=100,
                limit_per_host=20,
                enable_cleanup_closed=True,
                force_close=False
            )
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                trust_env=False
            )
            self._aio_session_pool[session_key] = session
        return self._aio_session_pool[session_key]

    def add_header(self, key, value):
        self.headers[key] = value

    def add_headers(self, headers):
        self.headers = {**self.headers, **headers}

    def call(self):
        response = self._handle_request()
        if self.stream:
            return (item for item in response)
        else:
            output = next(response)
            try:
                next(response)
            except StopIteration:
                pass
            return output

    async def aio_call(self):
        response = self._handle_aio_request()
        if self.stream:
            return (item async for item in response)
        else:
            result = await response.__anext__()
            try:
                await response.__anext__()
            except StopAsyncIteration:
                pass
            return result

    async def _handle_aio_request(self):
        try:
            session = await self._get_aio_session()
            logger.debug('Starting async request: %s', self.url)

            if self.method == HTTPMethod.POST:
                is_form, obj = self.data.get_aiohttp_payload()
                if is_form:
                    headers = {**self.headers, **obj.headers}
                    async with session.post(
                            url=self.url, data=obj, headers=headers
                    ) as response:
                        async for rsp in self._handle_aio_response(response):
                            yield rsp
                else:
                    async with session.post(
                            url=self.url, json=obj, headers=self.headers
                    ) as response:
                        async for rsp in self._handle_aio_response(response):
                            yield rsp
            elif self.method == HTTPMethod.GET:
                async with session.get(
                        url=self.url, params=self.data.parameters, headers=self.headers
                ) as response:
                    async for rsp in self._handle_aio_response(response):
                        yield rsp
            else:
                raise UnsupportedHTTPMethod(f'Unsupported http method: {self.method}')

        except aiohttp.ClientError as e:
            logger.error('HTTP client error: %s', e)
            raise
        except Exception as e:
            logger.error('Unexpected error: %s', e)
            raise

    async def _handle_aio_response(self, response: aiohttp.ClientResponse):
        request_id = ''
        if (response.status == HTTPStatus.OK and self.stream
                and SSE_CONTENT_TYPE in response.content_type):
            async for is_error, status_code, data in _handle_aio_stream(
                    response):
                try:
                    output = None
                    usage = None
                    msg = json.loads(data)
                    if not is_error:
                        if 'output' in msg:
                            output = msg['output']
                        if 'usage' in msg:
                            usage = msg['usage']
                    if 'request_id' in msg:
                        request_id = msg['request_id']
                except json.JSONDecodeError:
                    yield DashScopeAPIResponse(
                        request_id=request_id,
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        code='Unknown',
                        message=data)
                    continue
                if is_error:
                    yield DashScopeAPIResponse(request_id=request_id,
                                               status_code=status_code,
                                               code=msg['code'],
                                               message=msg['message'])
                else:
                    yield DashScopeAPIResponse(request_id=request_id,
                                               status_code=HTTPStatus.OK,
                                               output=output,
                                               usage=usage)
        elif (response.status == HTTPStatus.OK
              and 'multipart' in response.content_type):
            reader = aiohttp.MultipartReader.from_response(response)
            output = {}
            while True:
                part = await reader.next()
                if part is None:
                    break
                output[part.name] = await part.read()
            if 'request_id' in output:
                request_id = output['request_id']
            yield DashScopeAPIResponse(request_id=request_id,
                                       status_code=HTTPStatus.OK,
                                       output=output)
        elif response.status == HTTPStatus.OK:
            json_content = await response.json()
            output = None
            usage = None
            if 'output' in json_content and json_content['output'] is not None:
                output = json_content['output']
            if 'usage' in json_content:
                usage = json_content['usage']
            if 'request_id' in json_content:
                request_id = json_content['request_id']
            yield DashScopeAPIResponse(request_id=request_id,
                                       status_code=HTTPStatus.OK,
                                       output=output,
                                       usage=usage)
        else:
            yield await _handle_aiohttp_failed_response(response)

    def _handle_response(self, response: requests.Response):
        request_id = ''
        if (response.status_code == HTTPStatus.OK and self.stream
                and SSE_CONTENT_TYPE in response.headers.get(
                    'content-type', '')):
            for is_error, status_code, event in _handle_stream(response):
                try:
                    data = event.data
                    output = None
                    usage = None
                    msg = json.loads(data)
                    logger.debug('Stream message: %s' % msg)
                    if not is_error:
                        if 'output' in msg:
                            output = msg['output']
                        if 'usage' in msg:
                            usage = msg['usage']
                    if 'request_id' in msg:
                        request_id = msg['request_id']
                except json.JSONDecodeError:
                    yield DashScopeAPIResponse(
                        request_id=request_id,
                        status_code=HTTPStatus.BAD_REQUEST,
                        output=None,
                        code='Unknown',
                        message=data)
                    continue
                if is_error:
                    yield DashScopeAPIResponse(
                        request_id=request_id,
                        status_code=status_code,
                        output=None,
                        code=msg['code']
                        if 'code' in msg else None,  # noqa E501
                        message=msg['message']
                        if 'message' in msg else None)  # noqa E501
                else:
                    if self.flattened_output:
                        yield msg
                    else:
                        yield DashScopeAPIResponse(request_id=request_id,
                                                   status_code=HTTPStatus.OK,
                                                   output=output,
                                                   usage=usage)
        elif response.status_code == HTTPStatus.OK:
            json_content = response.json()
            logger.debug('Response: %s' % json_content)
            output = None
            usage = None
            if 'task_id' in json_content:
                output = {'task_id': json_content['task_id']}
            if 'output' in json_content:
                output = json_content['output']
            if 'usage' in json_content:
                usage = json_content['usage']
            if 'request_id' in json_content:
                request_id = json_content['request_id']
            if self.flattened_output:
                yield json_content
            else:
                yield DashScopeAPIResponse(request_id=request_id,
                                           status_code=HTTPStatus.OK,
                                           output=output,
                                           usage=usage)
        else:
            yield _handle_http_failed_response(response)

    def _handle_request(self):
        try:
            session = self._get_session()
            if self.method == HTTPMethod.POST:
                is_form, form, obj = self.data.get_http_payload()
                if is_form:
                    headers = {**self.headers}
                    headers.pop('Content-Type', None)
                    response = session.post(
                        url=self.url,
                        data=obj,
                        files=form,
                        headers=headers,
                        timeout=self.timeout,
                        stream=self.stream
                    )
                else:
                    logger.debug('Request body: %s', obj)
                    response = session.post(
                        url=self.url,
                        json=obj,
                        headers=self.headers,
                        timeout=self.timeout,
                        stream=self.stream
                    )
            elif self.method == HTTPMethod.GET:
                response = session.get(
                    url=self.url,
                    params=self.data.parameters,
                    headers=self.headers,
                    timeout=self.timeout,
                    stream=self.stream
                )
            else:
                raise UnsupportedHTTPMethod(f'Unsupported http method: {self.method}')

            for rsp in self._handle_response(response):
                yield rsp

        except requests.RequestException as e:
            logger.error('Request error: %s', e)
            raise
        except Exception as e:
            logger.error('Unexpected error: %s', e)
            raise


    @classmethod
    def cleanup(cls):
        """清理连接池资源"""
        for session in cls._session_pool.values():
            session.close()
        cls._session_pool.clear()

        for session in cls._aio_session_pool.values():
            if not session.closed:
                session.close()
        cls._aio_session_pool.clear()