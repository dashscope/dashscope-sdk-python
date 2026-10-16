# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgentStudio sync API example: end-to-end quickstart.

Flow: create environment -> create agent -> create session ->
send message -> stream events -> archive session.
"""

import os

from dashscope.agentstudio import Client, user_message


def main() -> None:
    # Authentication: set your Model Studio API key via the
    # DASHSCOPE_API_KEY environment variable, and the workspace via
    # DASHSCOPE_WORKSPACE (defaults to trial)
    client = Client(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        workspace=os.environ.get("DASHSCOPE_WORKSPACE", "trial"),
    )

    print("==> creating environment")
    env = client.environments.create(
        name="quickstart-env",
        config={
            "type": "cloud",
            "networking": {"type": "unrestricted"},
        },
    )
    print(f"   environment_id={env.id}")

    print("==> creating agent")
    agent = client.agents.create(
        name="quickstart-agent",
        system_prompt="You are a friendly demo agent. Reply briefly.",
        model="qwen3.8-max",
        tools=[
            {
                "type": "builtin_toolkit",
                "default_config": {"enabled": False},
                "configs": [{"name": "bash", "enabled": True}],
            },
        ],
    )
    print(f"   agent_id={agent.id}")

    print("==> creating session")
    session = client.sessions.create(
        agent=agent.id,
        environment_id=env.id,
        title="quickstart",
    )
    print(f"   session_id={session.id}")

    print("==> sending user message")
    client.sessions.events.send(
        session.id,
        [user_message("Hello! Tell me one fun fact about Hangzhou.")],
    )

    print("==> streaming events")
    with client.sessions.events.stream(session.id) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()  # newline after streaming

    print("==> archiving session")
    client.sessions.archive(session.id)
    print("done.")


if __name__ == "__main__":
    main()
