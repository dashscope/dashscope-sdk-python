from http import HTTPStatus
from dashscope import VideoSynthesis
import os

prompt = "一幅史诗级可爱的场景。一只小巧可爱的卡通小猫将军，身穿细节精致的金色盔甲，头戴一个稍大的头盔，勇敢地站在悬崖上。他骑着一匹虽小但英勇的战马。悬崖下方，一支由老鼠组成的、数量庞大、无穷无尽的军队正带着临时制作的武器向前冲锋。这是一个戏剧性的、大规模的战斗场景，灵感来自中国古代的战争史诗。远处的雪山上空，天空乌云密布。整体氛围是“可爱”与“霸气”的搞笑和史诗般的融合"
audio_url = 'https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/ozwpvi/rap.mp3'
api_key = os.getenv("DASHSCOPE_API_KEY")


def simple_call():
    print('----sync call, please wait a moment----')
    rsp = VideoSynthesis.call(api_key=api_key,
                              model="wan2.5-t2v-preview",
                              prompt=prompt,
                              audio_url=audio_url)
    if rsp.status_code == HTTPStatus.OK:

        print('response: %s' % rsp)
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))


if __name__ == '__main__':
    simple_call()