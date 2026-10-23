# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 agent recipe: AgentStudio end-to-end quickstart.

Flow: create environment -> create agent -> create session ->
send message -> stream events -> archive session.

AgentStudio is shared by the 1.x and 2.0 SDK generations: the 2.0
client mounts it as ``client.agentstudio`` (workspace and region
forward from the client config) and translates its errors into the
``dashscope.v2`` exception family, so one ``client`` covers both the
2.0 resources and AgentStudio.

`memory_stores_demo` covers the memory store flow (create store ->
write memory -> list -> update -> list versions -> delete). It creates
and deletes a real store, so it is opt-in: uncomment its call below.

Requires DASHSCOPE_API_KEY; DASHSCOPE_WORKSPACE selects the
AgentStudio workspace (defaults to trial).
"""

import os

import dashscope
from dashscope.agentstudio import ModelConfig, user_message


def make_client():
    # Authentication: the client reads the API key from the
    # DASHSCOPE_API_KEY environment variable by default.
    return dashscope.Client(
        workspace=os.environ.get("DASHSCOPE_WORKSPACE", "trial"),
    )


def main() -> None:
    client = make_client()
    studio = client.agentstudio

    print("==> creating environment")
    env = studio.environments.create(
        name="quickstart-env",
        config={
            "type": "cloud",
            "networking": {"type": "unrestricted"},
        },
    )
    print(f"   environment_id={env.id}")

    print("==> creating agent")
    # model is a ModelConfig (a bare id string is the ModelConfig(id=...)
    # shorthand); unset params keep the server defaults.
    agent = studio.agents.create(
        name="quickstart-agent",
        system_prompt="You are a friendly demo agent. Reply briefly.",
        model=ModelConfig(
            id="qwen3.8-max",
            temperature=0.7,
            max_tokens=4096,
        ),
        tools=[
            {
                "type": "builtin_toolkit",
                "default_config": {"enabled": False},
                "configs": [{"name": "bash", "enabled": True}],
            },
        ],
    )
    print(f"   agent_id={agent.id} model={agent.model_id}")

    print("==> creating session")
    session = studio.sessions.create(
        agent=agent.id,
        environment_id=env.id,
        title="quickstart",
    )
    print(f"   session_id={session.id}")

    print("==> sending user message")
    studio.sessions.events.send(
        session.id,
        [user_message("Hello! Tell me one fun fact about Hangzhou.")],
    )

    print("==> streaming events")
    with studio.sessions.events.stream(session.id) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()  # newline after streaming

    print("==> archiving session")
    studio.sessions.archive(session.id)
    print("done.")


def memory_stores_demo() -> None:
    """Memory store flow: create store -> seed memory -> read back
    -> audit versions -> cleanup."""
    studio = make_client().agentstudio

    print("==> creating memory store")
    store = studio.memory_stores.create(
        name="demo-preferences",
        description="Per-user preferences and project context.",
    )
    print(f"   memory_store_id={store.id}")

    print("==> seeding a memory")
    mem = studio.memory_stores.memories.create(
        store.id,
        path="/formatting_standards.md",
        content="All reports use GAAP formatting.",
    )
    print(f"   memory_id={mem.id} size={mem.content_size_bytes}")

    print("==> listing memories")
    page = studio.memory_stores.memories.list(store.id, view="full")
    for item in page:
        print(f"   {item.path} -> {item.content}")
    # Directory nodes ride the page's prefixes field (per page):
    print(f"   prefixes={page.prefixes}")

    print("==> updating with optimistic lock")
    updated = studio.memory_stores.memories.update(
        store.id,
        mem.id,
        content="All reports use GAAP formatting. ISO-8601 dates.",
        content_sha256=mem.content_sha256,
    )
    print(f"   new sha={updated.content_sha256[:12]}...")

    print("==> listing versions")
    for v in studio.memory_stores.memory_versions.list(store.id):
        print(f"   {v.id}: {v.operation}")

    print("==> deleting memory store")
    studio.memory_stores.delete(store.id)
    print("done.")


if __name__ == "__main__":
    main()
    # Memory flow creates and deletes a real store; opt in explicitly:
    # memory_stores_demo()
