# Copyright (c) Alibaba, Inc. and its affiliates.

from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Role, Message

if __name__ == '__main__':
    t2i_model = ImageGeneration.Models.wan2_6_t2i
    t2i_message = Message(
        role=Role.USER,
        content=[
            {
                'text': '一间有着精致窗户的花店，漂亮的木质门，摆放着花朵'
            }
        ]
    )

    image_model = ImageGeneration.Models.wan2_6_image
    image_message = Message(
        role=Role.USER,
        # 支持本地文件 如 "image": "file://umbrella1.png"
        content=[
            {
                "text": "参考图1的风格和图2的背景，生成番茄炒蛋"
            },
            {
                "image": "https://cdn.wanx.aliyuncs.com/tmp/pressure/umbrella1.png"
            },
            {
                "image": "https://img.alicdn.com/imgextra/i3/O1CN01SfG4J41UYn9WNt4X1_!!6000000002530-49-tps-1696-960.webp"
            }
        ]
    )

    t2i_sync_res = ImageGeneration.call(
            model=t2i_model,
            messages=[t2i_message]
        )
    print("-----------sync-t2i-call-res-----------")
    print(t2i_sync_res)


    image_sync_res = ImageGeneration.call(
            model=image_model,
            messages=[image_message]
        )
    print("-----------sync-image-call-res-----------")
    print(image_sync_res)


    t2i_async_res = ImageGeneration.async_call(
            model=t2i_model,
            messages=[t2i_message]
        )
    print("-----------async-t2i-call-res-----------")
    print(t2i_async_res)


    res = ImageGeneration.cancel(t2i_async_res.output.task_id)
    print("-----------async-t2i-cancel-res-----------")
    print(res)


    res = ImageGeneration.cancel(t2i_async_res)
    print("-----------async-t2i-cancel-res-----------")
    print(res)


    res = ImageGeneration.wait(t2i_async_res.output.task_id)
    print("-----------async-t2i-wait-res-----------")
    print(res)


    res = ImageGeneration.wait(t2i_async_res)
    print("-----------async-t2i-wait-res-----------")
    print(res)


    res = ImageGeneration.fetch(t2i_async_res.output.task_id)
    print("-----------async-t2i-fetch-res-----------")
    print(res)


    res = ImageGeneration.fetch(t2i_async_res)
    print("-----------async-t2i-fetch-res-----------")
    print(res)


    res = ImageGeneration.list()
    print("-----------async-task-list-res-----------")
    print(res)

    print("-" * 100)

    image_message = Message(
        role=Role.USER,
        # 支持本地文件 如 "image": "file://umbrella1.png"
        content=[
            {
                "text": "给我一个3张图辣椒炒肉教程"
            }
        ]
    )

    image_stream_res = ImageGeneration.call(
        model=image_model,
        messages=[image_message],
        stream=True,
        enable_interleave=True,
        max_images=3
    )
    print("-----------sync-image-stream-call-res-----------")
    for stream_res in image_stream_res:
        print(stream_res)