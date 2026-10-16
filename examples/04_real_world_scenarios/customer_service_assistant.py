# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 real-world scenario: an online customer-service
assistant with guardrails, tool access and multi-turn memory.

This is a complete, production-shaped case rather than a one-line model
call. It stitches together the patterns a real support bot needs:

* **Guardrailed persona** -- the system prompt scopes the bot to
  order / logistics / refund topics, forbids inventing order data, and
  tells it when to escalate to a human.
* **Business tools** -- the model reads real order state through
  ``query_order`` / ``track_shipment`` / ``check_refund_eligible`` and
  can open a support ticket via ``create_ticket`` (all backed here by a
  small in-memory "database"; swap in your own service calls).
* **Sliding-window memory** -- :class:`ConversationMemory` pins the
  system prompt, keeps only the most recent turns under a budget, and
  never splits an assistant tool-call turn from its tool results, so a
  long chat cannot blow the context window or corrupt the wire format.
* **Managed tool-call loop** -- the classic agent state machine with a
  ``MAX_TURNS`` guard and defensive JSON handling.

The demo runs a three-turn conversation where turn 2 ("can I refund
*it*?") only makes sense if the bot remembers the order id from turn 1,
which is exactly what the memory layer buys you.

Requires DASHSCOPE_API_KEY (it calls the real chat endpoint).
"""

import json
import os

import dashscope

MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")
# Hard stop so a misbehaving model cannot loop on tools forever.
MAX_TURNS = 6
# Number of post-system messages kept in context; older turns are
# dropped oldest-first. Tune against your model's context window.
MAX_HISTORY_TURNS = 20

SYSTEM_PROMPT = (
    "你是「示例商城」的在线客服助手。请严格遵守以下规则：\n"
    "1. 只回答与订单、物流、退款、售后相关的问题；对无关问题礼貌说明"
    "无法处理并引导用户回到售后话题。\n"
    "2. 涉及具体订单时，必须调用工具查询真实数据后再回答，"
    "禁止编造订单状态、物流或退款结论。\n"
    "3. 当无法解决问题、或用户明确要求人工时，调用 create_ticket "
    "创建工单，并把工单号告诉用户。\n"
    "4. 回答简洁、友好，使用中文。"
)

# ---- a tiny stand-in order database (replace with your real backend) ---- #
ORDERS = {
    "SO2024001": {
        "item": "无线蓝牙耳机",
        "amount": 299.0,
        "status": "已发货",
        "shipped_days": 2,
        "carrier": "示例A速运",
        "tracking_no": "EX1234567890",
        "latest_event": "【杭州市】快件已到达杭州转运中心",
        "signed": False,
    },
    "SO2024002": {
        "item": "机械键盘",
        "amount": 459.0,
        "status": "已签收",
        "shipped_days": 12,
        "carrier": "示例B物流",
        "tracking_no": "EX9988776655",
        "latest_event": "【北京市】已签收，签收人：本人",
        "signed": True,
    },
}

# Refund window in days after shipping, per store policy.
REFUND_WINDOW_DAYS = 7


def query_order(order_id):
    """Return the core state of one order."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"found": False, "message": f"未找到订单 {order_id}"}
    return {
        "found": True,
        "order_id": order_id,
        "item": order["item"],
        "amount": order["amount"],
        "status": order["status"],
    }


def track_shipment(order_id):
    """Return carrier / tracking / latest logistics event."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"found": False, "message": f"未找到订单 {order_id}"}
    return {
        "found": True,
        "order_id": order_id,
        "carrier": order["carrier"],
        "tracking_no": order["tracking_no"],
        "latest_event": order["latest_event"],
    }


def check_refund_eligible(order_id):
    """Decide refund eligibility from the store's 7-day policy."""
    order = ORDERS.get(order_id)
    if order is None:
        return {
            "found": False,
            "eligible": False,
            "reason": f"未找到订单 {order_id}",
        }
    if order["shipped_days"] <= REFUND_WINDOW_DAYS and not order["signed"]:
        return {
            "found": True,
            "eligible": True,
            "reason": "发货未超过 7 天且未签收，可申请退款",
        }
    if order["signed"]:
        reason = "订单已签收，需走退货流程，建议转人工处理"
    else:
        reason = f"发货已超过 {REFUND_WINDOW_DAYS} 天，暂不支持自助退款"
    return {"found": True, "eligible": False, "reason": reason}


def create_ticket(order_id, reason):
    """Open a human-support ticket; returns the ticket id."""
    ticket_id = f"TK{abs(hash(order_id + reason)) % 100000:05d}"
    return {
        "created": True,
        "ticket_id": ticket_id,
        "order_id": order_id,
        "reason": reason,
    }


# Tool schemas the model may call (OpenAI function shape, accepted by
# the native gateway too).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单的商品、金额与当前状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 SO2024001",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_shipment",
            "description": "查询订单的承运商、运单号与最新物流轨迹",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_refund_eligible",
            "description": "按商城退款政策判断订单是否可退款并给出原因",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "创建人工客服工单，用于无法自助解决或用户要求转人工",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "相关订单号"},
                    "reason": {
                        "type": "string",
                        "description": "转人工的原因，简述用户诉求",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]

TOOL_IMPL = {
    "query_order": query_order,
    "track_shipment": track_shipment,
    "check_refund_eligible": check_refund_eligible,
    "create_ticket": create_ticket,
}


class ConversationMemory:
    """Sliding-window history that pins the system prompt.

    Two production concerns are handled here:

    * **Context budget** -- only the most recent ``max_turns`` messages
      after the system prompt are kept, so a long conversation cannot
      exceed the model's window.
    * **Wire integrity** -- trimming from the front can leave an orphan
      ``role="tool"`` message whose ``assistant`` tool-call turn was
      dropped, which the API rejects. ``_trim`` drops any leading tool
      messages so every request stays well-formed.
    """

    def __init__(self, system_prompt, max_turns=MAX_HISTORY_TURNS):
        self._system = {"role": "system", "content": system_prompt}
        self._turns = []
        self._max_turns = max_turns

    def add(self, message):
        """Append one message and re-apply the window."""
        self._turns.append(message)
        self._trim()

    def extend(self, messages):
        """Append several messages and re-apply the window."""
        self._turns.extend(messages)
        self._trim()

    def _trim(self):
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]
        while self._turns and self._turns[0]["role"] == "tool":
            self._turns.pop(0)

    def messages(self):
        """The full wire-ready message list (system + kept turns)."""
        return [self._system] + list(self._turns)


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
        return json.dumps(impl(**args), ensure_ascii=False)
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


def handle_user_turn(client, memory, user_input):
    """Process one user message through the tool-call loop.

    The whole context is re-sent each turn (the API is stateless); the
    model keeps calling tools until it returns a plain answer, which is
    recorded back into ``memory`` and returned.
    """
    memory.add({"role": "user", "content": user_input})
    for turn in range(MAX_TURNS):
        # Non-streaming on purpose: a tool-call turn must be read as a
        # complete message before we can execute it. For a pure-chat
        # streaming reply see 01_atomic_capabilities/text_generation.
        resp = client.chat.completions.create(
            model=MODEL,
            messages=memory.messages(),
            tools=TOOLS,
        )
        message = resp.choices[0].message
        if not message.tool_calls:
            answer = message.content or ""
            memory.add({"role": "assistant", "content": answer})
            return answer

        # Echo the assistant tool-call turn verbatim, then feed back
        # each tool result. Both go through memory so the window and
        # wire-integrity rules keep applying.
        memory.add(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": _dump_tool_calls(message.tool_calls),
            },
        )
        for call in message.tool_calls:
            result = _run_tool(
                call.function.name,
                call.function.arguments,
            )
            print(
                f"    · [turn {turn}] tool {call.function.name}"
                f"({call.function.arguments}) -> {result}",
            )
            memory.add(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": result,
                },
            )
    fallback = "抱歉，我暂时无法处理该请求，正在为您转接人工客服。"
    memory.add({"role": "assistant", "content": fallback})
    return fallback


def main():
    """Run a three-turn support conversation against the real API."""
    client = dashscope.Client()
    memory = ConversationMemory(SYSTEM_PROMPT)

    # Turn 2 refers to "this order" without repeating the id: it only
    # works because ConversationMemory carries turn 1 forward.
    conversation = [
        "我的订单 SO2024001 到哪了？",
        "那这个订单现在能退款吗？",
        "帮我转人工吧。",
    ]
    for user_input in conversation:
        print(f"\n用户: {user_input}")
        answer = handle_user_turn(client, memory, user_input)
        print(f"客服: {answer}")

    print(
        f"\n----- 对话结束，上下文保留 " f"{len(memory.messages())} 条消息 -----",
    )


if __name__ == "__main__":
    main()
