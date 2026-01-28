# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import os
import aiohttp
import ssl
import certifi
from dashscope.aigc.generation import AioGeneration


class TestAioGeneration:
    """Test cases for AioGeneration API with cache control and streaming."""

    @staticmethod
    async def test_response_with_content():
        """Test async generation with content response."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "你好",
                        "cache_control": {
                            "type": "ephemeral",
                            "ttl": "5m",
                        },
                    },
                ],
            },
        ]

        # Call AioGeneration API with streaming enabled
        response = await AioGeneration.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen3-max",
            messages=messages,
            result_format="message",
            incremental_output=False,
            stream=True,
        )

        print("\n=== Async Content Response Test ===")
        async for chunk in response:
            print(chunk)

        """
        content的示例如下：
        {
            "status_code": 200,
            "request_id": "a0700200-eb09-4d6c-ae76-d891ec1ae77b",
            "code": "",
            "message": "",
            "output": {
                "text": null,
                "finish_reason": null,
                "choices": [
                    {
                        "finish_reason": "null",
                        "message": {
                            "role": "assistant",
                            "content": "你好"   ---> 需要merge的内容
                        }
                    }
                ]
            },
            "usage": {
                "input_tokens": 28,
                "output_tokens": 4,
                "total_tokens": 32,
                "cached_tokens": 0
            }
        }
        """

    @staticmethod
    async def test_response_with_reasoning_content():
        """Test async generation with reasoning content response."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "1.1和0.9哪个大",
                        "cache_control": {
                            "type": "ephemeral",
                            "ttl": "5m",
                        },
                    },
                ],
            },
        ]

        # Call AioGeneration API with streaming enabled
        response = await AioGeneration.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-plus",
            messages=messages,
            result_format="message",
            enable_thinking=True,
            incremental_output=False,  # enable_thinking为true时，只能设置为true
            stream=True,
        )

        print("\n=== Async Reasoning Content Response Test ===")
        async for chunk in response:
            print(chunk)

        """
        reasoning_content 的示例如下：
        {
            "status_code": 200,
            "request_id": "5bef2386-ce04-4d63-a650-fa6f72c84bfb",
            "code": "",
            "message": "",
            "output": {
                "text": null,
                "finish_reason": null,
                "choices": [
                    {
                        "finish_reason": "null",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "首先"  ---> 不需要merge的内容，因为只支持增量模式
                        }
                    }
                ]
            },
            "usage": {
                "input_tokens": 28,
                "output_tokens": 3,
                "total_tokens": 31,
                "output_tokens_details": {
                    "reasoning_tokens": 1
                },
                "prompt_tokens_details": {
                    "cached_tokens": 0
                }
            }
        }
        """

    @staticmethod
    async def test_response_with_tool_calls():
        """Test async generation with tool calls response."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "当你想知道现在的时间时非常有用。",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "当你想查询指定城市的天气时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                            },
                        },
                    },
                    "required": [
                        "location",
                    ],
                },
            },
        ]
        messages = [{"role": "user", "content": "杭州天气怎么样"}]
        response = await AioGeneration.call(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-plus",
            messages=messages,
            tools=tools,
            result_format="message",
            incremental_output=False,
            stream=True,
        )

        print("\n=== Async Tool Calls Response Test ===")
        async for chunk in response:
            print(chunk)

        """
        tool calls 示例：
        {
            "status_code": 200,
            "request_id": "fb9231ea-723c-4e72-b504-e075c7d312de",
            "code": "",
            "message": "",
            "output": {
                "text": null,
                "finish_reason": null,
                "choices": [
                    {
                        "finish_reason": "null",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [  ---> 需要merge的内容
                                {
                                    "index": 0,
                                    "id": "call_b9c94cf8838d46a4911a7e",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_weather",
                                        "arguments": "{\"location\":"
                                    }
                                }
                            ]
                        },
                        "index": 0
                    }
                ]
            },
            "usage": {
                "input_tokens": 204,
                "output_tokens": 16,
                "total_tokens": 220,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                }
            }
        }
        """

    @staticmethod
    async def test_response_with_search_info():
        """Test async generation with search info response."""
        # 配置API Key
        # 若没有配置环境变量，请用百炼API Key将下行替换为：API_KEY = "sk-xxx"
        API_KEY = os.getenv("DASHSCOPE_API_KEY")

        async def call_deep_research_model(messages, step_name):
            print(f"\n=== {step_name} ===")

            try:
                responses = await AioGeneration.call(
                    api_key=API_KEY,
                    model="qwen-deep-research",
                    messages=messages,
                    # qwen-deep-research模型目前仅支持流式输出
                    stream=True,
                    # incremental_output=True #使用增量输出请添加此参数
                )

                return await process_responses(responses, step_name)

            except Exception as e:
                print(f"调用API时发生错误: {e}")
                return ""

        # 显示阶段内容
        def display_phase_content(phase, content, status):
            if content:
                print(f"\n[{phase}] {status}: {content}")
            else:
                print(f"\n[{phase}] {status}")

        # 处理响应
        async def process_responses(responses, step_name):
            current_phase = None
            phase_content = ""
            research_goal = ""
            web_sites = []
            keepalive_shown = False  # 标记是否已经显示过KeepAlive提示

            async for response in responses:
                # 检查响应状态码
                if (
                    hasattr(response, "status_code")
                    and response.status_code != 200
                ):
                    print(f"HTTP返回码：{response.status_code}")
                    if hasattr(response, "code"):
                        print(f"错误码：{response.code}")
                    if hasattr(response, "message"):
                        print(f"错误信息：{response.message}")
                    print(
                        "请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code",
                    )
                    continue

                if hasattr(response, "output") and response.output:
                    message = response.output.get("message", {})
                    phase = message.get("phase")
                    content = message.get("content", "")
                    status = message.get("status")
                    extra = message.get("extra", {})

                    # 阶段变化检测
                    if phase != current_phase:
                        if current_phase and phase_content:
                            # 根据阶段名称和步骤名称来显示不同的完成描述
                            if (
                                step_name == "第一步：模型反问确认"
                                and current_phase == "answer"
                            ):
                                print(f"\n 模型反问阶段完成")
                            else:
                                print(f"\n {current_phase} 阶段完成")
                        current_phase = phase
                        phase_content = ""
                        keepalive_shown = False  # 重置KeepAlive提示标记

                        # 根据阶段名称和步骤名称来显示不同的描述
                        if step_name == "第一步：模型反问确认" and phase == "answer":
                            print(f"\n 进入模型反问阶段")
                        else:
                            print(f"\n 进入 {phase} 阶段")

                    # 处理WebResearch阶段的特殊信息
                    if phase == "WebResearch":
                        if extra.get("deep_research", {}).get("research"):
                            research_info = extra["deep_research"]["research"]

                            # 处理streamingQueries状态
                            if status == "streamingQueries":
                                if "researchGoal" in research_info:
                                    goal = research_info["researchGoal"]
                                    if goal:
                                        research_goal += goal
                                        print(
                                            f"\n   研究目标: {goal}",
                                            end="",
                                            flush=True,
                                        )

                            # 处理streamingWebResult状态
                            elif status == "streamingWebResult":
                                if "webSites" in research_info:
                                    sites = research_info["webSites"]
                                    if sites and sites != web_sites:  # 避免重复显示
                                        web_sites = sites
                                        print(f"\n   找到 {len(sites)} 个相关网站:")
                                        for i, site in enumerate(sites, 1):
                                            print(
                                                f"     {i}. {site.get('title', '无标题')}",
                                            )
                                            print(
                                                f"        描述: {site.get('description', '无描述')[:100]}...",
                                            )
                                            print(
                                                f"        URL: {site.get('url', '无链接')}",
                                            )
                                            if site.get("favicon"):
                                                print(
                                                    f"        图标: {site['favicon']}",
                                                )
                                            print()

                            # 处理WebResultFinished状态
                            elif status == "WebResultFinished":
                                print(
                                    f"\n   网络搜索完成，共找到 {len(web_sites)} 个参考信息源",
                                )
                                if research_goal:
                                    print(f"   研究目标: {research_goal}")

                    # 累积内容并显示
                    if content:
                        phase_content += content
                        # 实时显示内容
                        print(content, end="", flush=True)

                    # 显示阶段状态变化
                    if status and status != "typing":
                        print(f"\n   状态: {status}")

                        # 显示状态说明
                        if status == "streamingQueries":
                            print("   → 正在生成研究目标和搜索查询（WebResearch阶段）")
                        elif status == "streamingWebResult":
                            print("   → 正在执行搜索、网页阅读和代码执行（WebResearch阶段）")
                        elif status == "WebResultFinished":
                            print("   → 网络搜索阶段完成（WebResearch阶段）")

                    # 当状态为finished时，显示token消耗情况
                    if status == "finished":
                        if hasattr(response, "usage") and response.usage:
                            usage = response.usage
                            print(f"\n    Token消耗统计:")
                            print(
                                f"      输入tokens: {usage.get('input_tokens', 0)}",
                            )
                            print(
                                f"      输出tokens: {usage.get('output_tokens', 0)}",
                            )
                            print(
                                f"      请求ID: {response.get('request_id', '未知')}",
                            )

                    if phase == "KeepAlive":
                        # 只在第一次进入KeepAlive阶段时显示提示
                        if not keepalive_shown:
                            print("当前步骤已经完成，准备开始下一步骤工作")
                            keepalive_shown = True
                        continue

            if current_phase and phase_content:
                if step_name == "第一步：模型反问确认" and current_phase == "answer":
                    print(f"\n 模型反问阶段完成")
                else:
                    print(f"\n {current_phase} 阶段完成")

            return phase_content

        # 检查API Key
        if not API_KEY:
            print("错误：未设置 DASHSCOPE_API_KEY 环境变量")
            print("请设置环境变量或直接在代码中修改 API_KEY 变量")
            return

        print("=== Async Search Info Response Test ===")
        print("用户发起对话：研究一下人工智能在教育中的应用")

        # 第一步：模型反问确认
        # 模型会分析用户问题，提出细化问题来明确研究方向
        messages = [{"role": "user", "content": "研究一下人工智能在教育中的应用"}]
        step1_content = await call_deep_research_model(messages, "第一步：模型反问确认")

        # 第二步：深入研究
        # 基于第一步的反问内容，模型会执行完整的研究流程
        messages = [
            {"role": "user", "content": "研究一下人工智能在教育中的应用"},
            {"role": "assistant", "content": step1_content},  # 包含模型的反问内容
            {"role": "user", "content": "我主要关注个性化学习和智能评估这两个方面"},
        ]

        await call_deep_research_model(messages, "第二步：深入研究")
        print("\n 研究完成！")

    @staticmethod
    async def test_with_custom_session():
        """示例：使用自定义 ClientSession 进行连接复用"""
        print("\n=== 使用自定义 ClientSession 示例 ===")

        # 配置 TCPConnector
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        )

        # 创建自定义 ClientSession
        async with aiohttp.ClientSession(connector=connector) as session:
            # 使用同一个 session 进行多次请求
            for i in range(3):
                print(f"\n--- 请求 {i+1} ---")

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"你好"},
                ]

                response = await AioGeneration.call(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    model="qwen-turbo",
                    messages=messages,
                    result_format="message",
                    session=session,  # ← 传入自定义 session
                )

                print(f"响应: {response.output.choices[0].message.content}")

        print("\n✅ ClientSession 已自动关闭")

    @staticmethod
    async def test_with_custom_session_streaming():
        """示例：使用自定义 ClientSession 进行流式输出"""
        print("\n=== 使用自定义 ClientSession 流式输出示例 ===")

        # 配置连接池
        connector = aiohttp.TCPConnector(
            limit=100,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "请写一首关于秋天的诗"},
            ]

            print("\n流式输出:")
            response = await AioGeneration.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model="qwen-turbo",
                messages=messages,
                result_format="message",
                stream=True,
                incremental_output=True,
                session=session,  # ← 传入自定义 session
            )

            async for chunk in response:
                if chunk.status_code == 200:
                    print(chunk.output.choices[0].message.content, end='', flush=True)

            print("\n")

        print("✅ ClientSession 已自动关闭")

    @staticmethod
    async def test_with_custom_session_concurrent():
        """示例：使用自定义 ClientSession 进行并发请求"""
        print("\n=== 使用自定义 ClientSession 并发请求示例 ===")

        # 配置连接池
        connector = aiohttp.TCPConnector(
            limit=100,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            # 创建多个并发任务
            tasks = []
            topics = ["Python", "JavaScript", "Go"]

            for topic in topics:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"请用一句话介绍：{topic}"},
                ]

                task = AioGeneration.call(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    model="qwen-turbo",
                    messages=messages,
                    result_format="message",
                    session=session,  # ← 所有请求共享同一个 session
                )
                tasks.append(task)

            # 并发执行所有任务
            print("\n开始并发请求...")
            responses = await asyncio.gather(*tasks)

            # 处理响应
            for i, response in enumerate(responses):
                print(f"\n请求 {i+1} 响应: {response.output.choices[0].message.content}")

        print("\n✅ ClientSession 已自动关闭")


async def main():
    """Main function to run all async tests."""
    print("开始运行异步生成测试用例...")

    # Run individual test cases
    await TestAioGeneration.test_response_with_content()
    # await TestAioGeneration.test_response_with_tool_calls()
    # await TestAioGeneration.test_response_with_search_info()
    # await TestAioGeneration.test_response_with_reasoning_content()

    # 自定义 Session 示例
    # await TestAioGeneration.test_with_custom_session()
    # await TestAioGeneration.test_with_custom_session_streaming()
    # await TestAioGeneration.test_with_custom_session_concurrent()

    print("\n所有异步测试用例执行完成！")


if __name__ == "__main__":
    asyncio.run(main())
