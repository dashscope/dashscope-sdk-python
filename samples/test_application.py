import os
from http import HTTPStatus
from dashscope import Application
responses = Application.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            app_id=os.getenv("DASHSCOPE_APP_ID"),
            prompt='你是谁？',
            stream=True,  # 流式输出
            has_thoughts=True, # 输出节点内容
            incremental_output=True,
            flow_stream_mode='agent_format')  # 设置为Agent模式，透出指定节点的输出

for response in responses:
    if response.status_code != HTTPStatus.OK:
        print(f'request_id={response.request_id}')
        print(f'code={response.status_code}')
        print(f'message={response.message}')
        print(f'请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code')
    else:
        print(f'response: {response}\n')