# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 agent recipe: a managed multi-turn tool-call loop.

This is the classic agent state machine. Send the conversation plus a
tool schema; whenever the model asks to call a tool
(``message.tool_calls``), execute it locally, append the result as a
``role="tool"`` message, and call the model again -- until it returns a
plain answer with no further tool calls. A ``MAX_TURNS`` guard stops a
misbehaving model from looping forever.

Requires DASHSCOPE_API_KEY (it calls the real chat endpoint).
"""

import json
import os

import dashscope

MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")
MAX_TURNS = 5

# Tool schemas the model may choose to call (OpenAI function shape,
# which the native gateway also accepts).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市当前的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，例如 杭州",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "查询某个城市的常住人口（万人）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                },
                "required": ["city"],
            },
        },
    },
]


# ---- local tool implementations (stand-ins for real services) ---- #
def get_weather(city):
    table = {"杭州": "晴，26℃", "北京": "多云，21℃"}
    return table.get(city, "未知")


def get_population(city):
    table = {"杭州": 1237, "北京": 2184}
    return table.get(city, 0)


TOOL_IMPL = {
    "get_weather": get_weather,
    "get_population": get_population,
}


def _run_tool(name, arguments_json):
    """Execute one tool call, always returning a JSON string result."""
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "invalid JSON arguments"})
    impl = TOOL_IMPL.get(name)
    if impl is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return json.dumps({"result": impl(**args)}, ensure_ascii=False)
    except TypeError as exc:
        return json.dumps({"error": f"bad arguments: {exc}"})


def _dump_tool_calls(tool_calls):
    """Serialize ToolCall models back to plain wire dicts."""
    return [
        {
            "id": call.id,
            "type": call.type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in tool_calls
    ]


def agent_loop(client, user_input):
    """Drive the tool-call state machine until a final answer."""
    messages = [
        {
            "role": "system",
            "content": "你是一个助手，可以调用工具获取实时信息。",
        },
        {"role": "user", "content": user_input},
    ]
    for turn in range(MAX_TURNS):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        message = resp.choices[0].message
        if not message.tool_calls:
            print(f"[turn {turn}] final: {message.content}")
            return message.content

        # Echo the assistant tool-call turn back verbatim ...
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": _dump_tool_calls(message.tool_calls),
            },
        )
        # ... then run each tool and append its result.
        for call in message.tool_calls:
            name = call.function.name
            args = call.function.arguments
            result = _run_tool(name, args)
            print(f"[turn {turn}] {name}({args}) -> {result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": result,
                },
            )
    print(f"reached MAX_TURNS={MAX_TURNS} without a final answer")
    return None


def main():
    """Run the agent loop against the real chat endpoint."""
    client = dashscope.Client()
    agent_loop(client, "杭州现在天气怎么样？顺便说下它有多少人口。")


if __name__ == "__main__":
    main()
