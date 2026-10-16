# -*- coding: utf-8 -*-
"""TingWu realtime speech-understanding example: stream audio over
WebSocket and receive recognition results.

Before running:
1. Set your Model Studio API key via the DASHSCOPE_API_KEY
   environment variable;
2. Replace APP_ID / TERMINOLOGY with the app ID and terminology ID
   you created on the TingWu platform;
3. Prepare a 16 kHz PCM audio file (defaults to the sample audio
   under tests/data in this repository; it can be overridden via
   the DASHSCOPE_DEMO_AUDIO environment variable).
"""

import logging
import os
import time

from dashscope.multimodal.tingwu.tingwu_realtime import (
    TingWuRealtime,
    TingWuRealtimeCallback,
)

logger = logging.getLogger("tingwu_realtime_example")

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
MODEL = os.environ.get("DASHSCOPE_MODEL", "tingwu-industrial-instruction")
APP_ID = os.environ.get("DASHSCOPE_TINGWU_APP_ID", "your-app-id")
TERMINOLOGY = os.environ.get(
    "DASHSCOPE_TINGWU_TERMINOLOGY",
    "your-terminology-id",
)
AUDIO_FILE = os.environ.get(
    "DASHSCOPE_DEMO_AUDIO",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "data",
        "asr_example.wav",
    ),
)


class DemoCallback(TingWuRealtimeCallback):
    """Print TingWu realtime events and track the
    send/completion state."""

    def __init__(self):
        super().__init__()
        self.can_send_audio = False
        self.task_completed = False

    def on_open(self) -> None:
        logger.info("on websocket open.")

    def on_started(self, task_id: str) -> None:
        logger.info("on task started. task_id: %s", task_id)

    def on_speech_listen(self, result: dict):
        logger.info("on speech listen. result: %s", result)
        # audio data can only be sent after listening starts
        self.can_send_audio = True

    def on_recognize_result(self, result: dict):
        logger.info("on recognize result. result: %s", result)

    def on_ai_result(self, result: dict):
        logger.info("on ai result. result: %s", result)

    def on_stopped(self) -> None:
        logger.info("on task stopped.")
        self.can_send_audio = False
        self.task_completed = True

    def on_error(self, error_code: str, error_msg: str) -> None:
        logger.info(
            "on error. error_code: %s, error_msg: %s",
            error_code,
            error_msg,
        )
        self.task_completed = True

    def on_close(self, close_status_code, close_msg):
        logger.info(
            "on websocket close. close_status_code: %s, close_msg: %s",
            close_status_code,
            close_msg,
        )
        self.task_completed = True


def main() -> None:
    callback = DemoCallback()
    tingwu_realtime = TingWuRealtime(
        model=MODEL,
        audio_format="pcm",
        sample_rate=16000,
        app_id=APP_ID,
        base_address="wss://dashscope.aliyuncs.com/api-ws/v1/inference",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        terminology=TERMINOLOGY,
        callback=callback,
        max_end_silence=3000,
    )

    print("启动 TingWu 连接...")
    tingwu_realtime.start()
    time.sleep(2)  # wait for the connection to be established

    try:
        with open(AUDIO_FILE, "rb") as f:
            chunk_count = 0
            while True:
                chunk = f.read(3200)
                if not chunk:
                    print("文件读取完毕")
                    break
                if callback.can_send_audio:
                    tingwu_realtime.send_audio_frame(chunk)
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        print(f"已发送 {chunk_count} 个音频块")
                else:
                    time.sleep(0.1)  # wait until audio can be sent
    except FileNotFoundError:
        print(f"文件未找到: {AUDIO_FILE}")
    finally:
        print("发送 stop 指令...")
        tingwu_realtime.stop()

        print("等待任务完成...")
        wait_time = 0
        while not callback.task_completed and wait_time < 30:
            time.sleep(1)
            wait_time += 1

        print("任务已完成" if callback.task_completed else "等待超时，强制关闭连接")
        tingwu_realtime.close()
        print("TingWu 连接已关闭")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
