import os
from dashscope import Generation

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "abc" * 1024 + "你是谁？",
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": "5m"
                }
            }
        ]
    }
]
response = Generation.call(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model=os.getenv("MODEL_NAME"),
    messages=messages,
    result_format="message",
    incremental_output=True,
    stream=True,
)

for chunk in response:
    print(chunk)

# if response.status_code == 200:
#     print(response.output.choices[0].message.content)
# else:
#     print(f"HTTP返回码：{response.status_code}")
#     print(f"错误码：{response.code}")
#     print(f"错误信息：{response.message}")
#     print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")