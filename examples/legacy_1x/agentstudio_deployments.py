# -*- coding: utf-8 -*-
"""Create and run a Managed Agent deployment.

Set DASHSCOPE_API_KEY and DASHSCOPE_WORKSPACE before running this sample.
"""

from dashscope.agentstudio import Client, user_message


def main() -> None:
    with Client() as client:
        deployment = client.deployments.create(
            name="daily-summary",
            description="Generate a daily summary",
            agent={"id": "agent_xxx"},
            schedule={
                "type": "cron",
                "expression": "0 9 * * 1-5",
                "timezone": "Asia/Shanghai",
            },
            initial_events=[user_message("Summarize yesterday's orders")],
            metadata={"biz": "summary"},
        )
        print("deployment:", deployment.id, deployment.status)

        run = client.deployments.run(deployment.id)
        print("run:", run.id, run.status)

        for item in client.deployments.list_runs(deployment.id, limit=20):
            print(item.id, item.trigger_source, item.status)


if __name__ == "__main__":
    main()
