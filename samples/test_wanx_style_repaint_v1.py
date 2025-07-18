# Copyright (c) Alibaba, Inc. and its affiliates.
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis
import os

rsp = ImageSynthesis.call(api_key=os.getenv("DASHSCOPE_API_KEY"),
                          model="wanx-style-repaint-v1",
                          image_url='https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/public/dashscope/test.png',
                          style_index=-1,
                          style_ref_url="https://vigen-video.oss-cn-shanghai.aliyuncs.com/VideoGeneration/Data/cosplay%E8%A7%92%E8%89%B2%E5%BA%93/%E6%96%B0%E7%89%88%E9%9D%A2%E5%BD%A2%E8%B1%A1%E5%BA%93/ACG%E9%A3%8E%E6%A0%BC%EF%BC%88%E7%94%B7%EF%BC%89/65ba3ee96b1b868dfad0cf96c52c86112e035bb893329b8dda2a90b8a38485e0.png"
                          )

print('response: %s' % rsp)
if rsp.status_code == HTTPStatus.OK:
    # 在当前目录下保存图片
    for result in rsp.output.results:
        file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
        with open('./%s' % file_name, 'wb+') as f:
            f.write(requests.get(result.url).content)
else:
    print('sync_call Failed, status_code: %s, code: %s, message: %s' %
          (rsp.status_code, rsp.code, rsp.message))