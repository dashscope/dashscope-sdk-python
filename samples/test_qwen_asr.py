import os
import dashscope

messages = [
    {
        "role": "user",
        "content": [
            {"audio": "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3"},
        ]
    },
    {
        "role": "system",
        "content": [
            {"text": "这是一段介绍文本"},
        ]
    }
]
dashscope.base_http_api_url = "https://poc-dashscope.aliyuncs.com/api/v1/"
response = dashscope.MultiModalConversation.call(
    model="qwen3-asr-flash",
    messages=messages,
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    result_format="message",
    asr_options={"language": "zh", "enable_lid": True}
)
print(response)
print("recognized language: ", response.output.choices[0].message.get("annotations"))
