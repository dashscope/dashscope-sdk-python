import os

import dashscope
import logging

logger = logging.getLogger('dashscope')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
console_handler.setFormatter(formatter)

# add ch to logger
logger.addHandler(console_handler)

# switch stream or non-stream mode
use_stream = True

response = dashscope.MultiModalConversation.call(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="qwen-tts",
    text="Today is a wonderful day to build something people love!",
    voice="Cherry",
    stream=use_stream
)
if use_stream:
    # print the audio data in stream mode
    for chunk in response:
        audio = chunk.output.audio
        print("base64 audio data is: {}", chunk.output.audio.data)
        if chunk.output.finish_reason == "stop":
            print("finish at: {} ", chunk.output.audio.expires_at)
else:
    # print the audio url in non-stream mode
    print("synthesized audio url is: {}", response.output.audio.url)
    print("finish at: {} ", response.output.audio.expires_at)
