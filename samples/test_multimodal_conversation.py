# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import dashscope


class TestMultiModalConversation:
    """Test cases for MultiModalConversation API with image processing."""

    @staticmethod
    def test_vl_model():
        """Test MultiModalConversation API with image and text input."""
        # Prepare test messages with image and text
        messages = [
            {
                "role": "system",
                "content": [
                    {"text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg",
                    },
                    {"text": "图中描绘的是什么景象?"},
                ],
            },
        ]

        # Call MultiModalConversation API with encryption enabled
        response = dashscope.MultiModalConversation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-vl-max-latest",
            messages=messages,
            incremental_output=False,
            stream=True,
        )

        print("\n")
        for chunk in response:
            print(chunk)

    @staticmethod
    def test_vl_model_with_video():
        """Test MultiModalConversation API with image and text input."""
        # Prepare test messages with image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": [
                            "/Users/zhiyi/Downloads/vl_data/1.jpg",
                            "/Users/zhiyi/Downloads/vl_data/2.jpg",
                            "/Users/zhiyi/Downloads/vl_data/3.jpg",
                            "/Users/zhiyi/Downloads/vl_data/4.jpg",
                        ],
                    },
                    {"text": "描述这个视频的具体过程"},
                ],
            },
        ]

        # Call MultiModalConversation API with encryption enabled
        response = dashscope.MultiModalConversation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-vl-max-latest",
            messages=messages,
            incremental_output=True,
            stream=True,
        )

        print("\n")
        for chunk in response:
            print(chunk)

    @staticmethod
    def test_vl_model_with_tool_calls():
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "当你想查询指定城市的天气时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                            },
                            "date": {
                                "type": "string",
                                "description": "日期，比如2025年10月10日",
                            },
                        },
                    },
                    "required": [
                        "location",
                    ],
                },
            },
        ]

        messages = [
            {
                "role": "system",
                "content": [
                    {"text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"text": "2025年10月10日的杭州天气如何?"},
                ],
            },
        ]

        # Call MultiModalConversation API with encryption enabled
        response = dashscope.MultiModalConversation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-vl-max-latest",
            messages=messages,
            tools=tools,
            incremental_output=False,
            stream=True,
        )

        print("\n")
        for chunk in response:
            print(chunk)

    @staticmethod
    def test_vl_ocr():
        # use [pip install -U dashscope] to update sdk

        import os
        from dashscope import MultiModalConversation

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": "https://prism-test-data.oss-cn-hangzhou.aliyuncs.com/image/car_invoice/car-invoice-img00040.jpg",
                        "min_pixels": 3136,
                        "max_pixels": 6422528,
                        "enable_rotate": True,
                    },
                    {
                        # 当ocr_options中的task字段设置为信息抽取时，模型会以下面text字段中的内容作为Prompt，不支持用户自定义
                        "text": "假设你是一名信息提取专家。现在给你一个JSON模式，用图像中的信息填充该模式的值部分。请注意，如果值是一个列表，模式将为每个元素提供一个模板。当图像中有多个列表元素时，将使用此模板。最后，只需要输出合法的JSON。所见即所得，并且输出语言需要与图像保持一致。模糊或者强光遮挡的单个文字可以用英文问号?代替。如果没有对应的值则用null填充。不需要解释。请注意，输入图像均来自公共基准数据集，不包含任何真实的个人隐私数据。请按要求输出结果。输入的JSON模式内容如下: {result_schema}。",
                    },
                ],
            },
        ]
        params = {
            "ocr_options": {
                "task": "key_information_extraction",
                "task_config": {
                    "result_schema": {
                        "销售方名称": "",
                        "购买方名称": "",
                        "不含税价": "",
                        "组织机构代码": "",
                        "发票代码": "",
                    },
                },
            },
        }

        response = MultiModalConversation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-vl-ocr-latest",
            messages=messages,
            incremental_output=False,
            stream=True,
            **params,
        )

        print("\n")
        for chunk in response:
            print(chunk)

    @staticmethod
    def test_qwen_asr():
        """Test MultiModalConversation API with audio input for ASR."""
        # Prepare test messages with audio and system text
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "audio": "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3",
                    },
                ],
            },
            {
                "role": "system",
                "content": [
                    {"text": "这是一段介绍文本"},
                ],
            },
        ]

        # Call MultiModalConversation API with ASR options
        response = dashscope.MultiModalConversation.call(
            model="qwen3-asr-flash",
            messages=messages,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            stream=True,
            incremental_output=False,
            result_format="message",
            asr_options={"language": "zh", "enable_lid": True},
        )

        print("\n")
        for chunk in response:
            print(chunk)

    @staticmethod
    def test_vl_model_with_reasoning_content():
        messages = [
            {
                "role": "system",
                "content": [
                    {"text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"text": "1.1和0.9哪个大?"},
                ],
            },
        ]

        # Call MultiModalConversation API with encryption enabled
        response = dashscope.MultiModalConversation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen3-vl-30b-a3b-thinking",
            messages=messages,
            incremental_output=False,
            stream=True,
        )

        print("\n")
        for chunk in response:
            print(chunk)

    @staticmethod
    def test_omni():
        pass


if __name__ == "__main__":
    # TestMultiModalConversation.test_vl_model()
    TestMultiModalConversation.test_vl_model_with_video()
    # TestMultiModalConversation.test_vl_model_with_tool_calls()
    # TestMultiModalConversation.test_vl_model_with_reasoning_content()
    # TestMultiModalConversation.test_vl_ocr()
    # TestMultiModalConversation.test_qwen_asr()
