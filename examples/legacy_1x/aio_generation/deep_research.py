# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AioGeneration deep-research (qwen-deep-research) async example.

Demonstrates the two-step interaction flow:
1. Clarification: the model analyzes the question and asks
   follow-up questions;
2. Deep research: based on the user's answer, the model runs the
   full research workflow.

qwen-deep-research currently only supports streaming, and the
response is streamed by phase. This example shows how to parse and
display the progress of each phase.
"""

import asyncio
import os

from dashscope.aigc.generation import AioGeneration

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-deep-research")

USER_QUESTION = "研究一下人工智能在教育中的应用"
USER_FEEDBACK = "我主要关注个性化学习和智能评估这两个方面"


# pylint: disable=too-many-branches,too-many-statements
async def process_responses(responses):
    """Process the streamed responses, display per-phase progress,
    and return the full text of this step."""
    current_phase = None
    phase_content = ""
    research_goal = ""
    web_sites = []
    keepalive_shown = False

    async for response in responses:
        if hasattr(response, "status_code") and response.status_code != 200:
            print(f"HTTP返回码：{response.status_code}")
            if hasattr(response, "code"):
                print(f"错误码：{response.code}")
            if hasattr(response, "message"):
                print(f"错误信息：{response.message}")
            print(
                "请参考文档：https://help.aliyun.com/zh/model-studio/"
                "developer-reference/error-code",
            )
            continue

        if not (hasattr(response, "output") and response.output):
            continue

        message = response.output.get("message", {})
        phase = message.get("phase")
        content = message.get("content", "")
        status = message.get("status")
        extra = message.get("extra", {})

        # Phase-change detection
        if phase != current_phase:
            if current_phase and phase_content:
                print(f"\n {current_phase} 阶段完成")
            current_phase = phase
            phase_content = ""
            keepalive_shown = False
            print(f"\n 进入 {phase} 阶段")

        # Search-process details of the WebResearch phase
        if phase == "WebResearch":
            research_info = extra.get("deep_research", {}).get("research")
            if research_info:
                if status == "streamingQueries":
                    goal = research_info.get("researchGoal")
                    if goal:
                        research_goal += goal
                        print(f"\n   研究目标: {goal}", end="", flush=True)
                elif status == "streamingWebResult":
                    sites = research_info.get("webSites")
                    if sites and sites != web_sites:
                        web_sites = sites
                        print(f"\n   找到 {len(sites)} 个相关网站:")
                        for i, site in enumerate(sites, 1):
                            print(f"     {i}. {site.get('title', '无标题')}")
                            print(f"        URL: {site.get('url', '无链接')}")
                elif status == "WebResultFinished":
                    print(
                        f"\n   网络搜索完成，共找到 {len(web_sites)} 个参考信息源",
                    )
                    if research_goal:
                        print(f"   研究目标: {research_goal}")

        # Stream the content in real time
        if content:
            phase_content += content
            print(content, end="", flush=True)

        # Show token usage when the status is finished
        if status == "finished":
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                print("\n    Token消耗统计:")
                print(f"      输入tokens: {usage.get('input_tokens', 0)}")
                print(f"      输出tokens: {usage.get('output_tokens', 0)}")
                print(f"      请求ID: {response.get('request_id', '未知')}")

        if phase == "KeepAlive" and not keepalive_shown:
            print("当前步骤已经完成，准备开始下一步骤工作")
            keepalive_shown = True

    if current_phase and phase_content:
        print(f"\n {current_phase} 阶段完成")

    return phase_content


async def call_deep_research_model(messages, step_name):
    """Call the qwen-deep-research model and display one full step."""
    print(f"\n=== {step_name} ===")
    try:
        responses = await AioGeneration.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=MODEL,
            messages=messages,
            # qwen-deep-research currently only supports streaming
            stream=True,
            # incremental_output=True  # uncomment for incremental output
        )
        return await process_responses(responses)
    except Exception as e:
        print(f"调用API时发生错误: {e}")
        return ""


async def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误：未设置 DASHSCOPE_API_KEY 环境变量")
        return

    print(f"用户发起对话：{USER_QUESTION}")

    # Step 1: clarification — the model asks follow-up questions to
    # pin down the research direction
    step1_content = await call_deep_research_model(
        [{"role": "user", "content": USER_QUESTION}],
        "第一步：模型反问确认",
    )

    # Step 2: deep research — the model runs the full research
    # workflow based on the user's follow-up answer
    await call_deep_research_model(
        [
            {"role": "user", "content": USER_QUESTION},
            {"role": "assistant", "content": step1_content},
            {"role": "user", "content": USER_FEEDBACK},
        ],
        "第二步：深入研究",
    )
    print("\n 研究完成！")


if __name__ == "__main__":
    asyncio.run(main())
