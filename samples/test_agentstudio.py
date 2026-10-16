# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from dashscope.agentstudio import Client, user_message


class TestAgentStudio:
    """Test cases for AgentStudio synchronous API."""

    @staticmethod
    def test_quickstart():
        """Test end-to-end flow: create env -> agent -> session
        -> send message -> stream events -> archive session."""
        api_key = os.environ["DASHSCOPE_API_KEY"]
        workspace = os.environ.get("DASHSCOPE_WORKSPACE", "trial")
        client = Client(api_key=api_key, workspace=workspace)

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
            model="qwen-max",
            tools=[{"type": "builtin_toolkit", "default_config": {"enabled": False},
                    "configs": [{"name": "bash", "enabled": True}]}],
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

    @staticmethod
    def test_memory_stores():
        """Memory store flow: create store -> seed memory -> read back
        -> audit versions -> cleanup."""
        api_key = os.environ["DASHSCOPE_API_KEY"]
        workspace = os.environ.get("DASHSCOPE_WORKSPACE", "trial")
        client = Client(api_key=api_key, workspace=workspace)

        print("==> creating memory store")
        store = client.memory_stores.create(
            name="demo-preferences",
            description="Per-user preferences and project context.",
        )
        print(f"   memory_store_id={store.id}")

        print("==> seeding a memory")
        mem = client.memory_stores.memories.create(
            store.id,
            path="/formatting_standards.md",
            content="All reports use GAAP formatting.",
        )
        print(f"   memory_id={mem.id} size={mem.content_size_bytes}")

        print("==> listing memories")
        page = client.memory_stores.memories.list(store.id, view="full")
        for item in page:
            print(f"   {item.path} -> {item.content}")
        # Directory nodes ride the page's prefixes field (per page):
        print(f"   prefixes={page.prefixes}")

        print("==> updating with optimistic lock (precondition)")
        updated = client.memory_stores.memories.update(
            store.id,
            mem.id,
            content="All reports use GAAP formatting. ISO-8601 dates.",
            precondition=mem.content_sha256,
        )
        print(f"   new sha={updated.content_sha256[:12]}...")

        print("==> listing versions")
        for v in client.memory_stores.memory_versions.list(store.id):
            print(f"   {v.id}: {v.operation}")

        print("==> deleting memory store")
        client.memory_stores.delete(store.id)
        print("done.")


if __name__ == "__main__":
    TestAgentStudio.test_quickstart()
    # Memory flow creates and deletes a real store; opt in explicitly:
    # TestAgentStudio.test_memory_stores()
