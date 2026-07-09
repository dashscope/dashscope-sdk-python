#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证脚本：测试四处错误处理问题

运行方式：python verify_error_handling.py
"""

import json
from http import HTTPStatus
from unittest.mock import Mock, MagicMock
from dashscope.client.base_api import StreamEventMixin
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse


def test_issue_1_stream_event_mixin():
    """
    问题 1: StreamEventMixin 错误码和消息被丢弃
    
    预期行为：当 SSE 流返回错误时，code 和 message 应该是空字符串
    修复后：应该能正确解析 data 字段中的 JSON 错误信息
    """
    print("\n" + "="*70)
    print("测试 1: StreamEventMixin 错误信息丢失问题")
    print("="*70)
    
    # 模拟 SSE 错误响应
    mock_response = Mock()
    mock_response.status_code = HTTPStatus.OK
    mock_response.headers = {"content-type": "text/event-stream"}
    
    # 模拟服务器返回的错误数据（JSON 格式）
    error_data = {
        "code": "invalid_request_error",
        "message": "Invalid parameter: model not found",
        "request_id": "req_test_123"
    }
    
    mock_response.iter_lines.return_value = [
        b"event:error",
        b"status:400",
        f"data:{json.dumps(error_data)}".encode('utf-8'),
    ]
    
    # 调用 _handle_response
    results = list(StreamEventMixin._handle_response(mock_response))
    
    print(f"\n📋 测试结果:")
    print(f"   返回结果数量: {len(results)}")
    
    if results:
        result = results[0]
        print(f"   status_code: {result.status_code}")
        print(f"   code: '{result.code}' (长度: {len(result.code)})")
        print(f"   message: '{result.message}' (长度: {len(result.message)})")
        print(f"   request_id: '{result.request_id}'")
        
        # 验证问题是否存在
        if result.code == "" and result.message == "":
            print("\n❌ 问题确认: code 和 message 都是空字符串，错误信息被丢弃！")
            print(f"   期望 code: 'invalid_request_error'")
            print(f"   期望 message: 'Invalid parameter: model not found'")
            return False
        else:
            print("\n✅ 问题已修复: 成功解析到错误信息")
            return True
    else:
        print("\n⚠️  警告: 没有返回任何结果")
        return False


def test_issue_2_websocket_handshake():
    """
    问题 2: WebSocket 握手错误消息被替换
    
    预期行为：对于 401/403/503 状态码，原始错误消息被硬编码提示覆盖
    修复后：应该保留原始消息并追加友好提示
    """
    print("\n" + "="*70)
    print("测试 2: WebSocket 握手错误消息替换问题")
    print("="*70)
    
    import aiohttp
    
    # 模拟 WSServerHandshakeError
    mock_error = Mock(spec=aiohttp.WSServerHandshakeError)
    mock_error.status = HTTPStatus.UNAUTHORIZED
    mock_error.message = "Token expired at 2026-07-09 10:00:00"
    
    print(f"\n📋 模拟场景:")
    print(f"   状态码: {mock_error.status}")
    print(f"   原始错误消息: '{mock_error.message}'")
    
    # 当前代码的行为（修复前）
    code = mock_error.status
    msg = mock_error.message
    if mock_error.status in [HTTPStatus.FORBIDDEN, HTTPStatus.UNAUTHORIZED]:
        msg = "Unauthorized, your api-key is invalid!"
    elif mock_error.status == HTTPStatus.SERVICE_UNAVAILABLE:
        from dashscope.common.constants import SERVICE_503_MESSAGE
        msg = SERVICE_503_MESSAGE
    
    print(f"\n❌ 当前行为（修复前）:")
    print(f"   最终消息: '{msg}'")
    print(f"   ⚠️  原始消息 '{mock_error.message}' 被完全覆盖！")
    
    # 修复后的预期行为
    original_msg = mock_error.message or ""
    if mock_error.status in [HTTPStatus.FORBIDDEN, HTTPStatus.UNAUTHORIZED]:
        friendly_hint = "Unauthorized, your api-key may be invalid!"
        expected_msg = f"{friendly_hint} (Server details: {original_msg})" if original_msg else friendly_hint
    else:
        expected_msg = original_msg
    
    print(f"\n✅ 修复后预期行为:")
    print(f"   最终消息: '{expected_msg}'")
    print(f"   ✓ 保留了原始消息 '{original_msg}'")
    
    return False  # 这个问题需要查看实际代码才能确认


def test_issue_3_iter_over_async():
    """
    问题 3: iter_over_async 桥接包装为自定义格式
    
    预期行为：异步迭代器异常时，错误码固定为 "Unknown"
    修复后：应该使用更明确的标识或空字符串
    """
    print("\n" + "="*70)
    print("测试 3: iter_over_async 错误码硬编码问题")
    print("="*70)
    
    # 模拟一个异步迭代器抛出异常
    async def failing_async_gen():
        yield "data1"
        raise ValueError("Test error from async generator")
    
    from dashscope.common.utils import iter_over_async
    
    print(f"\n📋 模拟场景:")
    print(f"   异步生成器抛出: ValueError('Test error from async generator')")
    
    # 捕获所有结果
    results = []
    try:
        for item in iter_over_async(failing_async_gen()):
            results.append(item)
    except Exception as e:
        print(f"   ⚠️  迭代过程中抛出异常: {e}")
    
    print(f"\n📋 检查结果:")
    if results:
        last_result = results[-1]
        if isinstance(last_result, DashScopeAPIResponse):
            print(f"   最后一个结果类型: DashScopeAPIResponse")
            print(f"   code: '{last_result.code}'")
            print(f"   message: '{last_result.message}'")
            
            if last_result.code == "Unknown":
                print(f"\n❌ 问题确认: 错误码硬编码为 'Unknown'")
                print(f"   期望: 更明确的错误分类或空字符串表示 SDK 内部错误")
                return False
            else:
                print(f"\n✅ 问题已改进: 错误码不是 'Unknown'")
                return True
        else:
            print(f"   最后一个结果类型: {type(last_result)}")
    else:
        print(f"   没有捕获到任何结果")
    
    return False


def test_issue_4_unknown_fallback():
    """
    问题 4: 非 JSON 响应的 Unknown fallback
    
    预期行为：未分类异常使用 code="Unknown"
    修复后：应该使用空字符串或其他明确标识
    """
    print("\n" + "="*70)
    print("测试 4: BaseException 兜底处理的 Unknown 错误码")
    print("="*70)
    
    print(f"\n📋 问题分析:")
    print(f"   当前代码: code='Unknown'")
    print(f"   问题: 'Unknown' 无法区分是 API 未提供错误码，还是 SDK 内部错误")
    
    print(f"\n✅ 修复建议:")
    print(f"   方案 A: code='' (空字符串表示 SDK 内部错误)")
    print(f"   方案 B: code='__sdk_internal_error__' (特殊标识)")
    print(f"   方案 C: 保持 code='Unknown'，但改进 message 格式")
    
    print(f"\n💡 推荐: 方案 A + 改进 message 格式")
    print(f"   修改前: message='Error type: <class 'ValueError'>, message: xxx'")
    print(f"   修改后: message='[SDK Internal Error] ValueError: xxx'")
    
    return False  # 这是设计决策，需要人工确认


def main():
    """主测试函数"""
    print("\n" + "🔍"*35)
    print("DashScope SDK 错误处理问题验证")
    print("🔍"*35)
    
    results = []
    
    # 测试 1: 可以自动化验证
    try:
        result1 = test_issue_1_stream_event_mixin()
        results.append(("StreamEventMixin", result1))
    except Exception as e:
        print(f"\n❌ 测试 1 执行失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("StreamEventMixin", None))
    
    # 测试 2-4: 需要查看实际代码或人工确认
    test_issue_2_websocket_handshake()
    test_issue_3_iter_over_async()
    test_issue_4_unknown_fallback()
    
    # 总结
    print("\n" + "="*70)
    print("📊 验证总结")
    print("="*70)
    print(f"\n✅ 已自动化验证: 测试 1 (StreamEventMixin)")
    print(f"⚠️  需人工确认: 测试 2-4 (需要查看实际运行时的行为)")
    
    if results and results[0][1] is False:
        print(f"\n🎯 结论: 问题 1 确实存在，建议立即修复")
    elif results and results[0][1] is True:
        print(f"\n🎉 结论: 问题 1 已经修复")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
