# -*- coding: utf-8 -*-
# 调用TingWu
from dashscope.multimodal.tingwu.tingwu import TingWu
import os

# 创建TingWu实例
tingwu = TingWu()
resp = TingWu.call(
    model="tingwu-automotive-service-inspection",
    user_defined_input={
        "fileUrl": "http://demo.com/test.mp3",
        "appid": "123456",
    },
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_address="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
)
print(resp)
