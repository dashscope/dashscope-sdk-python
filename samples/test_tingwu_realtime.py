import logging
import os
import time
import sys
from dashscope.multimodal.tingwu.tingwu_realtime import TingWuRealtime, TingWuRealtimeCallback

# 配置日志 - 关键改进
logger = logging.getLogger('dashscope')
logger.setLevel(logging.DEBUG)

# 创建控制台处理器并设置级别为debug
console_handler = logging.StreamHandler(sys.stdout)  # 明确指定输出到stdout
console_handler.setLevel(logging.DEBUG)

# 创建格式化器
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 添加格式化器到处理器
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(console_handler)

# 强制刷新日志输出
logger.propagate = False


class TestCallback(TingWuRealtimeCallback):
    def __init__(self):  # 修复：__init__ 方法名
        super().__init__()
        self.can_send_audio = False
        self.task_completed = False  # 新增：标记任务是否完成
        print("TestCallback initialized")  # 添加调试输出

    def on_open(self) -> None:
        logger.info('TingWuClient:: on websocket open.')

    def on_started(self, task_id: str) -> None:
        logger.info('TingWuClient:: on task started. task_id: %s', task_id)

    def on_speech_listen(self, result: dict):
        logger.info('TingWuClient:: on speech listen. result: %s', result)
        self.can_send_audio = True  # 标记可以发送语音数据

    def on_recognize_result(self, result: dict):
        logger.info('TingWuClient:: on recognize result. result: %s', result)

    def on_ai_result(self, result: dict):
        print(f'TingWuClient:: on ai result. result: {result}')
        logger.info('TingWuClient:: on ai result. result: %s', result)

    def on_stopped(self) -> None:
        logger.info('TingWuClient:: on task stopped.')
        self.can_send_audio = False  # 标记不能发送语音数据
        self.task_completed = True

    def on_error(self, error_code: str, error_msg: str) -> None:
        logger.info('TingWuClient:: on error. error_code: %s, error_msg: %s',
                    error_code, error_msg)
        self.task_completed = True

    def on_close(self, close_status_code, close_msg):
        logger.info('TingWuClient:: on websocket close. close_status_code: %s, close_msg: %s',
                    close_status_code, close_msg)
        self.task_completed = True


class TestTingWuRealtime():
    @classmethod
    def setup_class(cls):
        cls.model = 'tingwu-industrial-instruction'  # replace model name
        cls.format = 'pcm'
        cls.sample_rate = 16000
        cls.file = './data/tingwu_test_audio.wav'
        cls.appId = 'your-app-id'
        cls.base_address = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'
        cls.api_key = os.getenv('DASHSCOPE_API_KEY')
        cls.terminology = 'your-terminology-id'

    def test_async_start_with_stream(self):
        print("开始测试...")
        sys.stdout.flush()

        callback = TestCallback()

        print("创建 TingWuRealtime 实例...")
        sys.stdout.flush()

        tingwu_realtime = TingWuRealtime(
            model=self.model,
            audio_format=self.format,
            sample_rate=self.sample_rate,
            app_id=self.appId,
            base_address=self.base_address,
            api_key=self.api_key,
            terminology=self.terminology,
            callback=callback,
            max_end_silence=3000
        )

        print("启动 TingWu 连接...")
        sys.stdout.flush()
        tingwu_realtime.start()

        # 等待连接建立
        time.sleep(2)

        try:
            print(f"打开文件: {self.file}")
            sys.stdout.flush()

            with open(self.file, 'rb') as f:
                chunk_count = 0
                while True:
                    chunk = f.read(3200)
                    if not chunk:
                        print("文件读取完毕")
                        sys.stdout.flush()
                        break

                    # 修复逻辑错误：应该在can_send_audio为True时发送
                    if callback.can_send_audio:
                        tingwu_realtime.send_audio_frame(chunk)
                        chunk_count += 1
                        if chunk_count % 10 == 0:  # 每10个chunk打印一次
                            print(f"已发送 {chunk_count} 个音频块")
                            sys.stdout.flush()
                    else:
                        print("等待可以发送音频...")
                        sys.stdout.flush()
                        time.sleep(0.1)  # 短暂等待

        except FileNotFoundError:
            print(f"文件未找到: {self.file}")
            sys.stdout.flush()
        except Exception as e:
            print(f"发生错误: {e}")
            sys.stdout.flush()
        finally:
            print("发送 stop 指令...")
            sys.stdout.flush()
            tingwu_realtime.stop()

            print("等待任务完成...")
            sys.stdout.flush()
            wait_time = 0
            max_wait_time = 30  # 最多等待30秒
            while not callback.task_completed and wait_time < max_wait_time:
                time.sleep(1)
                wait_time += 1
                print(f"已等待 {wait_time} 秒...")
                sys.stdout.flush()

            if callback.task_completed:
                print("任务已完成")
            else:
                print("等待超时，强制关闭连接")

            sys.stdout.flush()
            tingwu_realtime.close()
            print("TingWu 连接已关闭")


if __name__ == '__main__':
    logger.debug('Start test_tingwu_realtime.')

    tingwu_realtime = TestTingWuRealtime()
    tingwu_realtime.setup_class()
    tingwu_realtime.test_async_start_with_stream()

    print('End test_tingwu_realtime.')
    sys.stdout.flush()
