# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from http import HTTPStatus

from dashscope import CodeGeneration
from dashscope.aigc.code_generation import (
    AttachmentRoleMessageParam,
    UserRoleMessageParam,
)
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer

model = CodeGeneration.Models.tongyi_lingma_v1

# yapf: disable


class TestCodeGenerationRequest(MockServerBase):
    def test_custom_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694702346.730724,
                    'index': 0,
                    'content':
                    '以下是生成Python函数的代码：\n\n```python\ndef file_size(path):\n    total_size = 0\n    for root, dirs, files in os.walk(path):\n        for file in files:\n            full_path = os.path.join(root, file)\n            total_size += os.path.getsize(full_path)\n    return total_size\n```\n\n函数名为`file_size`，输入参数是给定路径`path`。函数通过递归遍历给定路径下的所有文件，使用`os.walk`函数遍历根目录及其子目录下的文件，计算每个文件的大小并累加到总大小上。最后，返回总大小作为函数的返回值。',  # noqa E501
                    'frame_id': 25,
                }],
            },
            'usage': {
                'output_tokens': 198,
                'input_tokens': 46,
            },
            'request_id': 'bf321b27-a3ff-9674-a70e-be5f40a435e4',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.custom,
            message=[
                UserRoleMessageParam(
                    content='根据下面的功能描述生成一个python函数。代码的功能是计算给定路径下所有文件的总大小。',
                ),
            ],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'custom'
        assert json.dumps(
            req['input']['message'], ensure_ascii=False,
        ) == '[{"role": "user", "content": "根据下面的功能描述生成一个python函数。代码的功能是计算给定路径下所有文件的总大小。"}]'

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == 'bf321b27-a3ff-9674-a70e-be5f40a435e4'
        assert response.output['choices'][0][
            'content'
        ] == '以下是生成Python函数的代码：\n\n```python\ndef file_size(path):\n    total_size = 0\n    for root, dirs, files in os.walk(path):\n        for file in files:\n            full_path = os.path.join(root, file)\n            total_size += os.path.getsize(full_path)\n    return total_size\n```\n\n函数名为`file_size`，输入参数是给定路径`path`。函数通过递归遍历给定路径下的所有文件，使用`os.walk`函数遍历根目录及其子目录下的文件，计算每个文件的大小并累加到总大小上。最后，返回总大小作为函数的返回值。'  # noqa E501
        assert response.output['choices'][0]['frame_id'] == 25
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 198
        assert response.usage['input_tokens'] == 46

    def test_custom_dict_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694702346.730724,
                    'index': 0,
                    'content':
                    '以下是生成Python函数的代码：\n\n```python\ndef file_size(path):\n    total_size = 0\n    for root, dirs, files in os.walk(path):\n        for file in files:\n            full_path = os.path.join(root, file)\n            total_size += os.path.getsize(full_path)\n    return total_size\n```\n\n函数名为`file_size`，输入参数是给定路径`path`。函数通过递归遍历给定路径下的所有文件，使用`os.walk`函数遍历根目录及其子目录下的文件，计算每个文件的大小并累加到总大小上。最后，返回总大小作为函数的返回值。',  # noqa E501
                    'frame_id': 25,
                }],
            },
            'usage': {
                'output_tokens': 198,
                'input_tokens': 46,
            },
            'request_id': 'bf321b27-a3ff-9674-a70e-be5f40a435e4',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.custom,
            message=[{
                'role': 'user',
                'content': '根据下面的功能描述生成一个python函数。代码的功能是计算给定路径下所有文件的总大小。',
            }],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'custom'
        assert json.dumps(
            req['input']['message'], ensure_ascii=False,
        ) == '[{"role": "user", "content": "根据下面的功能描述生成一个python函数。代码的功能是计算给定路径下所有文件的总大小。"}]'

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == 'bf321b27-a3ff-9674-a70e-be5f40a435e4'
        assert response.output['choices'][0][
            'content'
        ] == '以下是生成Python函数的代码：\n\n```python\ndef file_size(path):\n    total_size = 0\n    for root, dirs, files in os.walk(path):\n        for file in files:\n            full_path = os.path.join(root, file)\n            total_size += os.path.getsize(full_path)\n    return total_size\n```\n\n函数名为`file_size`，输入参数是给定路径`path`。函数通过递归遍历给定路径下的所有文件，使用`os.walk`函数遍历根目录及其子目录下的文件，计算每个文件的大小并累加到总大小上。最后，返回总大小作为函数的返回值。'  # noqa E501
        assert response.output['choices'][0]['frame_id'] == 25
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 198
        assert response.usage['input_tokens'] == 46

    def test_nl2code_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694692088.1848974,
                    'index': 0,
                    'content':
                    "```java\n/**\n * 计算给定路径下所有文件的总大小\n * @param path 路径\n * @return 总大小，单位为字节\n */\npublic static long getTotalFileSize(String path) {\n    long size = 0;\n    try {\n        File file = new File(path);\n        File[] files = file.listFiles();\n        for (File f : files) {\n            if (f.isFile()) {\n                size += f.length();\n            }\n        }\n    } catch (Exception e) {\n        e.printStackTrace();\n    }\n    return size;\n}\n```\n\n使用方式:\n```java\nlong size = getTotalFileSize(\"/home/user/Documents/\");\nSystem.out.println(\"总大小：\" + size + \"字节\");\n```\n\n示例输出:\n```\n总大小：37144952字节\n```",  # noqa E501
                    'frame_id': 29,
                }],
            },
            'usage': {
                'output_tokens': 229,
                'input_tokens': 39,
            },
            'request_id': '59bbbea3-29a7-94d6-8c39-e4d6e465f640',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.nl2code,
            message=[
                UserRoleMessageParam(content='计算给定路径下所有文件的总大小'),
                AttachmentRoleMessageParam(meta={'language': 'java'}),
            ],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'nl2code'
        assert json.dumps(
            req['input']['message'], ensure_ascii=False,
        ) == '[{"role": "user", "content": "计算给定路径下所有文件的总大小"}, {"role": "attachment", "meta": {"language": "java"}}]'

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == '59bbbea3-29a7-94d6-8c39-e4d6e465f640'
        assert response.output['choices'][0][
            'content'
        ] == "```java\n/**\n * 计算给定路径下所有文件的总大小\n * @param path 路径\n * @return 总大小，单位为字节\n */\npublic static long getTotalFileSize(String path) {\n    long size = 0;\n    try {\n        File file = new File(path);\n        File[] files = file.listFiles();\n        for (File f : files) {\n            if (f.isFile()) {\n                size += f.length();\n            }\n        }\n    } catch (Exception e) {\n        e.printStackTrace();\n    }\n    return size;\n}\n```\n\n使用方式:\n```java\nlong size = getTotalFileSize(\"/home/user/Documents/\");\nSystem.out.println(\"总大小：\" + size + \"字节\");\n```\n\n示例输出:\n```\n总大小：37144952字节\n```"  # noqa E501
        assert response.output['choices'][0]['frame_id'] == 29
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 229
        assert response.usage['input_tokens'] == 39

    def test_code2comment_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694692326.983717,
                    'index': 0,
                    'content':
                    '```java\n/**\n * 取消导出任务的回调函数\n *\n * @param cancelExportTask 取消导出任务的请求对象\n * @return 取消导出任务的响应对象\n */\n@Override\npublic CancelExportTaskResponse cancelExportTask(CancelExportTask cancelExportTask) {\n\tAmazonEC2SkeletonInterface ec2Service = ServiceProvider.getInstance().getServiceImpl(AmazonEC2SkeletonInterface.class);\n\treturn ec2Service.cancelExportTask(cancelExportTask);\n}\n```',  # noqa E501
                    'frame_id': 17,
                }],
            },
            'usage': {
                'output_tokens': 133,
                'input_tokens': 141,
            },
            'request_id': 'b5e55877-bfa3-9863-88d8-09a72124cf8a',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.code2comment,
            message=[
                UserRoleMessageParam(
                    content='1. 生成中文注释\n2. 仅生成代码部分，不需要额外解释函数功能\n',
                ),
                AttachmentRoleMessageParam(
                    meta={
                        'code':
                        '\t\t@Override\n\t\tpublic  CancelExportTaskResponse  cancelExportTask(\n\t\t\t\tCancelExportTask  cancelExportTask)  {\n\t\t\tAmazonEC2SkeletonInterface  ec2Service  =  ServiceProvider.getInstance().getServiceImpl(AmazonEC2SkeletonInterface.class);\n\t\t\treturn  ec2Service.cancelExportTask(cancelExportTask);\n\t\t}',  # noqa E501
                        'language': 'java',
                    },
                ),
            ],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'code2comment'

        assert json.dumps(
            req['input']['message'], ensure_ascii=False,
        ) == '[{"role": "user", "content": "1. 生成中文注释\n2. 仅生成代码部分，不需要额外解释函数功能\n"}, {"role": "attachment", "meta": {"code": "\t\t@Override\n\t\tpublic  CancelExportTaskResponse  cancelExportTask(\n\t\t\t\tCancelExportTask  cancelExportTask)  {\n\t\t\tAmazonEC2SkeletonInterface  ec2Service  =  ServiceProvider.getInstance().getServiceImpl(AmazonEC2SkeletonInterface.class);\n\t\t\treturn  ec2Service.cancelExportTask(cancelExportTask);\n\t\t}", "language": "java"}}]'.replace('\t', '\\t').replace('\n', '\\n')  # noqa E501

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == 'b5e55877-bfa3-9863-88d8-09a72124cf8a'
        assert response.output['choices'][0][
            'content'
        ] == '```java\n/**\n * 取消导出任务的回调函数\n *\n * @param cancelExportTask 取消导出任务的请求对象\n * @return 取消导出任务的响应对象\n */\n@Override\npublic CancelExportTaskResponse cancelExportTask(CancelExportTask cancelExportTask) {\n\tAmazonEC2SkeletonInterface ec2Service = ServiceProvider.getInstance().getServiceImpl(AmazonEC2SkeletonInterface.class);\n\treturn ec2Service.cancelExportTask(cancelExportTask);\n}\n```'  # noqa E501
        assert response.output['choices'][0]['frame_id'] == 17
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 133
        assert response.usage['input_tokens'] == 141

    def test_code2explain_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694697070.7664366,
                    'index': 0,
                    'content':
                    '这个Java函数是一个覆盖了另一个方法的函数，名为`getHeaderCacheSize()`。这个方法是从另一个已覆盖的方法继承过来的。在`@Override`声明中，可以确定这个函数覆盖了一个其他的函数。这个函数的返回类型是`int`。\n\n函数内容是：返回0。这个值意味着在`getHeaderCacheSize()`方法中，不会进行任何处理或更新。因此，返回的`0`值应该是没有被处理或更新的值。\n\n总的来说，这个函数的作用可能是为了让另一个方法返回一个预设的值。但是由于`@Override`的提示，我们无法确定它的真正目的，需要进一步查看代码才能得到更多的信息。',  # noqa E501
                    'frame_id': 30,
                }],
            },
            'usage': {
                'output_tokens': 235,
                'input_tokens': 55,
            },
            'request_id': '089e525f-d28f-9e08-baa2-01dde87c90a7',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.code2explain,
            message=[
                UserRoleMessageParam(content='要求不低于200字'),
                AttachmentRoleMessageParam(
                    meta={
                        'code':
                        '@Override\n                                public  int  getHeaderCacheSize()\n                                {\n                                        return  0;\n                                }\n\n',  # noqa E501
                        'language': 'java',
                    },
                ),
            ],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'code2explain'
        assert json.dumps(
            req['input']['message'], ensure_ascii=False,
        ) == '[{"role": "user", "content": "要求不低于200字"}, {"role": "attachment", "meta": {"code": "@Override\n                                public  int  getHeaderCacheSize()\n                                {\n                                        return  0;\n                                }\n\n", "language": "java"}}]'.replace('\t', '\\t').replace('\n', '\\n')  # noqa E501

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == '089e525f-d28f-9e08-baa2-01dde87c90a7'
        assert response.output['choices'][0][
            'content'
        ] == '这个Java函数是一个覆盖了另一个方法的函数，名为`getHeaderCacheSize()`。这个方法是从另一个已覆盖的方法继承过来的。在`@Override`声明中，可以确定这个函数覆盖了一个其他的函数。这个函数的返回类型是`int`。\n\n函数内容是：返回0。这个值意味着在`getHeaderCacheSize()`方法中，不会进行任何处理或更新。因此，返回的`0`值应该是没有被处理或更新的值。\n\n总的来说，这个函数的作用可能是为了让另一个方法返回一个预设的值。但是由于`@Override`的提示，我们无法确定它的真正目的，需要进一步查看代码才能得到更多的信息。'  # noqa E501
        assert response.output['choices'][0]['frame_id'] == 30
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 235
        assert response.usage['input_tokens'] == 55

    def test_commit2msg_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694697276.4451804,
                    'index': 0,
                    'content': 'Remove old listFolder method',
                    'frame_id': 1,
                }],
            },
            'usage': {
                'output_tokens': 5,
                'input_tokens': 197,
            },
            'request_id': '8f400a4e-6448-94ab-89bf-a97b1a7e6fe6',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.commit2msg,
            message=[
                AttachmentRoleMessageParam(
                    meta={
                        'diff_list': [{
                            'diff':
                            '--- src/com/siondream/core/PlatformResolver.java\n+++ src/com/siondream/core/PlatformResolver.java\n@@ -1,11 +1,8 @@\npackage com.siondream.core;\n-\n-import com.badlogic.gdx.files.FileHandle;\n\npublic interface PlatformResolver {\npublic void openURL(String url);\npublic void rateApp();\npublic void sendFeedback();\n-\tpublic FileHandle[] listFolder(String path);\n}\n',  # noqa E501
                            'old_file_path':
                            'src/com/siondream/core/PlatformResolver.java',
                            'new_file_path':
                            'src/com/siondream/core/PlatformResolver.java',
                        }],
                    },
                ),
            ],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'commit2msg'
        assert json.dumps(
            req['input']['message'], ensure_ascii=False,
        ) == '[{"role": "attachment", "meta": {"diff_list": [{"diff": "--- src/com/siondream/core/PlatformResolver.java\n+++ src/com/siondream/core/PlatformResolver.java\n@@ -1,11 +1,8 @@\npackage com.siondream.core;\n-\n-import com.badlogic.gdx.files.FileHandle;\n\npublic interface PlatformResolver {\npublic void openURL(String url);\npublic void rateApp();\npublic void sendFeedback();\n-\tpublic FileHandle[] listFolder(String path);\n}\n", "old_file_path": "src/com/siondream/core/PlatformResolver.java", "new_file_path": "src/com/siondream/core/PlatformResolver.java"}]}}]'.replace('\t', '\\t').replace('\n', '\\n')  # noqa E501

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == '8f400a4e-6448-94ab-89bf-a97b1a7e6fe6'
        assert response.output['choices'][0][
            'content'
        ] == 'Remove old listFolder method'
        assert response.output['choices'][0]['frame_id'] == 1
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 5
        assert response.usage['input_tokens'] == 197

    def test_unittest_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694697446.0802872,
                    'index': 0,
                    'content':
                    "这个函数用于解析时间戳映射表的输入字符串并返回该映射表的实例。函数有两个必选参数：typeClass - 用于标识数据类型的泛型；input - 输入的时间戳映射表字符串。如果typeClass为null，将抛出IllegalArgumentException异常；如果input为null，则返回null。函数内部首先检查输入的字符串是否等于\"空字符串\"，如果是，则直接返回null；如果不是，则创建TimestampMap的实例，并使用input字符串创建字符串Reader对象。然后使用读取器逐个字符解析时间戳字符串，并在解析完成后返回相应的TimestampMap对象。函数的行为取决于传入的时间戳字符串类型。",  # noqa E501
                    'frame_id': 29,
                }],
            },
            'usage': {
                'output_tokens': 227,
                'input_tokens': 659,
            },
            'request_id': '6ec31e35-f355-9289-a18d-103abc36dece',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.unit_test,
            message=[
                AttachmentRoleMessageParam(
                    meta={
                        'code':
                        "public static <T> TimestampMap<T> parseTimestampMap(Class<T> typeClass, String input, DateTimeZone timeZone) throws IllegalArgumentException {\n        if (typeClass == null) {\n            throw new IllegalArgumentException(\"typeClass required\");\n        }\n\n        if (input == null) {\n            return null;\n        }\n\n        TimestampMap result;\n\n        typeClass = AttributeUtils.getStandardizedType(typeClass);\n        if (typeClass.equals(String.class)) {\n            result = new TimestampStringMap();\n        } else if (typeClass.equals(Byte.class)) {\n            result = new TimestampByteMap();\n        } else if (typeClass.equals(Short.class)) {\n            result = new TimestampShortMap();\n        } else if (typeClass.equals(Integer.class)) {\n            result = new TimestampIntegerMap();\n        } else if (typeClass.equals(Long.class)) {\n            result = new TimestampLongMap();\n        } else if (typeClass.equals(Float.class)) {\n            result = new TimestampFloatMap();\n        } else if (typeClass.equals(Double.class)) {\n            result = new TimestampDoubleMap();\n        } else if (typeClass.equals(Boolean.class)) {\n            result = new TimestampBooleanMap();\n        } else if (typeClass.equals(Character.class)) {\n            result = new TimestampCharMap();\n        } else {\n            throw new IllegalArgumentException(\"Unsupported type \" + typeClass.getClass().getCanonicalName());\n        }\n\n        if (input.equalsIgnoreCase(EMPTY_VALUE)) {\n            return result;\n        }\n\n        StringReader reader = new StringReader(input + ' ');// Add 1 space so\n                                                            // reader.skip\n                                                            // function always\n                                                            // works when\n                                                            // necessary (end of\n                                                            // string not\n                                                            // reached).\n\n        try {\n            int r;\n            char c;\n            while ((r = reader.read()) != -1) {\n                c = (char) r;\n                switch (c) {\n                    case LEFT_BOUND_SQUARE_BRACKET:\n                    case LEFT_BOUND_BRACKET:\n                        parseTimestampAndValue(typeClass, reader, result, timeZone);\n                        break;\n                    default:\n                        // Ignore other chars outside of bounds\n                }\n            }\n        } catch (IOException ex) {\n            throw new RuntimeException(\"Unexpected expection while parsing timestamps\", ex);\n        }\n\n        return result;\n    }",  # noqa E501
                        'language': 'java',
                    },
                ),
            ],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'unittest'
        assert req['input']['message'][0]['role'] == 'attachment'
        assert req['input']['message'][0]['meta'][
            'code'
        ] == """public static <T> TimestampMap<T> parseTimestampMap(Class<T> typeClass, String input, DateTimeZone timeZone) throws IllegalArgumentException {\n        if (typeClass == null) {\n            throw new IllegalArgumentException(\"typeClass required\");\n        }\n\n        if (input == null) {\n            return null;\n        }\n\n        TimestampMap result;\n\n        typeClass = AttributeUtils.getStandardizedType(typeClass);\n        if (typeClass.equals(String.class)) {\n            result = new TimestampStringMap();\n        } else if (typeClass.equals(Byte.class)) {\n            result = new TimestampByteMap();\n        } else if (typeClass.equals(Short.class)) {\n            result = new TimestampShortMap();\n        } else if (typeClass.equals(Integer.class)) {\n            result = new TimestampIntegerMap();\n        } else if (typeClass.equals(Long.class)) {\n            result = new TimestampLongMap();\n        } else if (typeClass.equals(Float.class)) {\n            result = new TimestampFloatMap();\n        } else if (typeClass.equals(Double.class)) {\n            result = new TimestampDoubleMap();\n        } else if (typeClass.equals(Boolean.class)) {\n            result = new TimestampBooleanMap();\n        } else if (typeClass.equals(Character.class)) {\n            result = new TimestampCharMap();\n        } else {\n            throw new IllegalArgumentException(\"Unsupported type \" + typeClass.getClass().getCanonicalName());\n        }\n\n        if (input.equalsIgnoreCase(EMPTY_VALUE)) {\n            return result;\n        }\n\n        StringReader reader = new StringReader(input + ' ');// Add 1 space so\n                                                            // reader.skip\n                                                            // function always\n                                                            // works when\n                                                            // necessary (end of\n                                                            // string not\n                                                            // reached).\n\n        try {\n            int r;\n            char c;\n            while ((r = reader.read()) != -1) {\n                c = (char) r;\n                switch (c) {\n                    case LEFT_BOUND_SQUARE_BRACKET:\n                    case LEFT_BOUND_BRACKET:\n                        parseTimestampAndValue(typeClass, reader, result, timeZone);\n                        break;\n                    default:\n                        // Ignore other chars outside of bounds\n                }\n            }\n        } catch (IOException ex) {\n            throw new RuntimeException(\"Unexpected expection while parsing timestamps\", ex);\n        }\n\n        return result;\n    }"""  # noqa E501
        assert req['input']['message'][0]['meta']['language'] == 'java'

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == '6ec31e35-f355-9289-a18d-103abc36dece'
        assert response.output['choices'][0][
            'content'
        ] == "这个函数用于解析时间戳映射表的输入字符串并返回该映射表的实例。函数有两个必选参数：typeClass - 用于标识数据类型的泛型；input - 输入的时间戳映射表字符串。如果typeClass为null，将抛出IllegalArgumentException异常；如果input为null，则返回null。函数内部首先检查输入的字符串是否等于\"空字符串\"，如果是，则直接返回null；如果不是，则创建TimestampMap的实例，并使用input字符串创建字符串Reader对象。然后使用读取器逐个字符解析时间戳字符串，并在解析完成后返回相应的TimestampMap对象。函数的行为取决于传入的时间戳字符串类型。"  # noqa E501
        assert response.output['choices'][0]['frame_id'] == 29
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 227
        assert response.usage['input_tokens'] == 659

    def test_codeqa_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694700989.0357094,
                    'index': 0,
                    'content':
                    "Yes, this is possible:\nclass MyRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):\n  [...]\n\n  def doGET(self):\n    # some stuff\n    if \"X-Port\" in self.headers:\n      # change the port in this request\n      self.server_port = int(self.headers[\"X-Port\"])\n      print(\"Changed port: %s\" % self.server_port)\n    [...]\n\nclass ThreadingHTTPServer(ThreadingMixIn, HTTPServer): \n    pass\n\nserver = ThreadingHTTPServer(('localhost', self.server_port), MyRequestHandler)\nserver.serve_forever()",  # noqa E501
                    'frame_id': 19,
                }],
            },
            'usage': {
                'output_tokens': 150,
                'input_tokens': 127,
            },
            'request_id': 'e09386b7-5171-96b0-9c6f-7128507e14e6',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.code_qa,
            message=[
                UserRoleMessageParam(
                    content="I'm writing a small web server in Python, using BaseHTTPServer and a custom subclass of BaseHTTPServer.BaseHTTPRequestHandler. Is it possible to make this listen on more than one port?\nWhat I'm doing now:\nclass MyRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):\n  def doGET\n  [...]\n\nclass ThreadingHTTPServer(ThreadingMixIn, HTTPServer): \n    pass\n\nserver = ThreadingHTTPServer(('localhost', 80), MyRequestHandler)\nserver.serve_forever()",  # noqa E501
                ),
            ],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'codeqa'
        assert req['input']['message'][0]['role'] == 'user'
        assert req['input']['message'][0][
            'content'
        ] == """I'm writing a small web server in Python, using BaseHTTPServer and a custom subclass of BaseHTTPServer.BaseHTTPRequestHandler. Is it possible to make this listen on more than one port?\nWhat I'm doing now:\nclass MyRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):\n  def doGET\n  [...]\n\nclass ThreadingHTTPServer(ThreadingMixIn, HTTPServer): \n    pass\n\nserver = ThreadingHTTPServer(('localhost', 80), MyRequestHandler)\nserver.serve_forever()"""  # noqa E501

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == 'e09386b7-5171-96b0-9c6f-7128507e14e6'
        assert response.output['choices'][0][
            'content'
        ] == "Yes, this is possible:\nclass MyRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):\n  [...]\n\n  def doGET(self):\n    # some stuff\n    if \"X-Port\" in self.headers:\n      # change the port in this request\n      self.server_port = int(self.headers[\"X-Port\"])\n      print(\"Changed port: %s\" % self.server_port)\n    [...]\n\nclass ThreadingHTTPServer(ThreadingMixIn, HTTPServer): \n    pass\n\nserver = ThreadingHTTPServer(('localhost', self.server_port), MyRequestHandler)\nserver.serve_forever()"  # noqa E501
        assert response.output['choices'][0]['frame_id'] == 19
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 150
        assert response.usage['input_tokens'] == 127

    def test_nl2sql_sample(self, mock_server: MockServer):
        response_obj = {
            'output': {
                'choices': [{
                    'finish_reason': 'stop',
                    'frame_timestamp': 1694701323.4553578,
                    'index': 0,
                    'content':
                    "SELECT SUM(score) as '小明的总分数' FROM student_score WHERE name = '小明';",
                    'frame_id': 3,
                }],
            },
            'usage': {
                'output_tokens': 25,
                'input_tokens': 420,
            },
            'request_id': 'e61a35b7-db6f-90c2-8677-9620ffea63b6',
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = CodeGeneration.call(
            model=model,
            scene=CodeGeneration.Scenes.nl2sql,
            message=[
                UserRoleMessageParam(content='小明的总分数是多少'),
                AttachmentRoleMessageParam(
                    meta={
                        'synonym_infos': {
                            '学生姓名': '姓名|名字|名称',
                            '学生分数': '分数|得分',
                        },
                        'recall_infos': [{
                            'content': "student_score.id='小明'",
                            'score': '0.83',
                        }],
                        'schema_infos': [{
                            'table_id':
                            'student_score',
                            'table_desc':
                            '学生分数表',
                            'columns': [
                                {
                                    'col_name': 'id',
                                    'col_caption': '学生id',
                                    'col_desc': '例值为:1,2,3',
                                    'col_type': 'string',
                                }, {
                                    'col_name': 'name',
                                    'col_caption': '学生姓名',
                                    'col_desc': '例值为:张三,李四,小明',
                                    'col_type': 'string',
                                }, {
                                    'col_name': 'score',
                                    'col_caption': '学生分数',
                                    'col_desc': '例值为:98,100,66',
                                    'col_type': 'string',
                                },
                            ],
                        }],
                    },
                ),
            ],
        )
        req = mock_server.requests.get(block=True)
        assert req['model'] == model
        assert req['input']['scene'] == 'nl2sql'
        assert json.dumps(req['input']['message'], ensure_ascii=False) == """[{"role": "user", "content": "小明的总分数是多少"}, {"role": "attachment", "meta": {"synonym_infos": {"学生姓名": "姓名|名字|名称", "学生分数": "分数|得分"}, "recall_infos": [{"content": "student_score.id='小明'", "score": "0.83"}], "schema_infos": [{"table_id": "student_score", "table_desc": "学生分数表", "columns": [{"col_name": "id", "col_caption": "学生id", "col_desc": "例值为:1,2,3", "col_type": "string"}, {"col_name": "name", "col_caption": "学生姓名", "col_desc": "例值为:张三,李四,小明", "col_type": "string"}, {"col_name": "score", "col_caption": "学生分数", "col_desc": "例值为:98,100,66", "col_type": "string"}]}]}}]"""  # noqa E501

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == 'e61a35b7-db6f-90c2-8677-9620ffea63b6'
        assert response.output['choices'][0][
            'content'
        ] == "SELECT SUM(score) as '小明的总分数' FROM student_score WHERE name = '小明';"
        assert response.output['choices'][0]['frame_id'] == 3
        assert response.output['choices'][0]['finish_reason'] == 'stop'
        assert response.usage['output_tokens'] == 25
        assert response.usage['input_tokens'] == 420
