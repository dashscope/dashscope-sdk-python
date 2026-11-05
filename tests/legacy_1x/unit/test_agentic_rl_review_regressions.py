# -*- coding: utf-8 -*-
"""Regression tests for the dashscope-2.0-agentic-rl-review findings.

Each test maps to a specific issue in the review so a future refactor that
re-introduces the bug fails loudly:

- P0 (§2): API Key must be resolved at call time / instance scope, never from
  the import-time constant, and an explicit key must reach FC + file requests.
- P1 (§4): two Reward Processor instances must not share ``reward_meta``; a
  later instance must not rebind an earlier instance's sub-functions.
- P1 (§5): ``FuncManager.execute()`` must be a working async entry point that
  runs ``parses()`` + ``processes()`` (it used to call non-existent
  ``parse()`` / ``process()`` and raise ``AttributeError``).

Imports use the canonical ``dashscope.applications.finetune.reinforcement``
package so the error/data classes are the exact v2 ones under test.
"""
import inspect
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dashscope.applications.finetune.agentic_rl import AgenticRL
from dashscope.applications.finetune.reinforcement import (
    AbstractRewardProcessor,
    AgentOutput,
    FileSpec,
    FunctionType,
    Reward,
    RewardInput,
    RewardOutput,
    TaskStatus,
    TuningModel,
    client_fc,
    resolve_api_key,
    resolve_fc_api_key,
    reward_func,
    sub_reward_func,
    to_bailian_data,
)
from dashscope.applications.finetune.reinforcement.common import constants
from dashscope.applications.finetune.reinforcement.component.demo.reward_processor_demo import (  # noqa: E501
    DemoRewardProcessor,
)
from dashscope.applications.finetune.reinforcement.component.func_manager import (  # noqa: E501
    FuncManager,
)

# Patch target: ``client_fc`` / ``to_bailian_data`` call ``async_http_request``
# as a module global inside ``common.utils``.
_HTTP = (
    "dashscope.applications.finetune.reinforcement.common.utils"
    ".async_http_request"
)


@pytest.fixture(autouse=True)
def _restore_api_key_env():
    """Snapshot/restore credential env vars.

    ``set_api_key`` (called by ``AgenticRL.__init__``) writes directly to
    ``os.environ``, which monkeypatch does not track, so restore manually to
    keep tests isolated.
    """
    keys = ("DASHSCOPE_API_KEY", "FC_API_KEY")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ========================================================================== #
#                     P0 (§2): credential resolution                          #
# ========================================================================== #


class TestCredentialResolution:
    def test_resolve_fc_api_key_precedence(self, monkeypatch):
        """explicit > FC_API_KEY > DASHSCOPE_API_KEY > '' (call-time)."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-dash")
        monkeypatch.setenv("FC_API_KEY", "env-fc")

        assert resolve_fc_api_key("explicit") == "explicit"
        assert resolve_fc_api_key(None) == "env-fc"

        monkeypatch.delenv("FC_API_KEY")
        assert resolve_fc_api_key(None) == "env-dash"

        monkeypatch.delenv("DASHSCOPE_API_KEY")
        assert resolve_fc_api_key(None) == ""

    def test_resolve_api_key_precedence(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-dash")
        assert resolve_api_key("explicit") == "explicit"
        assert resolve_api_key(None) == "env-dash"

    def test_runtime_resolution_tracks_env_not_import_snapshot(
        self,
        monkeypatch,
    ):
        """The review repro: a key set AFTER import must still be honored.

        ``constants.DASHSCOPE_API_KEY`` is a frozen import-time snapshot; the
        resolvers must read the live environment instead so a stale/empty
        snapshot can never be used as the runtime credential.
        """
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-new-review")

        assert resolve_api_key(None) == "sk-new-review"
        assert resolve_fc_api_key(None) == "sk-new-review"
        # The snapshot is not the runtime source (it may still be '' / old).
        assert constants.DASHSCOPE_API_KEY != "sk-new-review"


class TestExplicitKeyReachesRequests:
    @pytest.mark.asyncio
    async def test_client_fc_sets_authorization_from_passed_key(self):
        """FC requests must carry the key passed in, not a global."""
        captured = {}

        async def _fake_http(**kwargs):
            captured.update(kwargs)
            return {"status": {"code": 200}}

        with patch(_HTTP, new=_fake_http):
            await client_fc(
                "sk-explicit",
                "https://example.com/api",
                {"a": 1},
            )

        assert captured["headers"]["Authorization"] == "Bearer sk-explicit"

    @pytest.mark.asyncio
    async def test_to_bailian_data_uses_explicit_key(self, tmp_path):
        """File upload must use the explicit key even when env differs."""
        os.environ["DASHSCOPE_API_KEY"] = "sk-stale-env"
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"prompt": "hi"}', encoding="utf-8")

        captured = {}

        async def _fake_http(**kwargs):
            captured.update(kwargs)
            return {
                "status": {"code": 200},
                "data": {"uploaded_files": [{"file_id": "fid-1"}]},
            }

        with patch(_HTTP, new=_fake_http):
            ids = await to_bailian_data(
                [FileSpec(path=str(data_file))],
                api_key="sk-file",
            )

        assert ids == ["fid-1"]
        assert captured["headers"]["Authorization"] == "Bearer sk-file"


class TestAgenticRLInstanceScope:
    def test_instance_key_isolated_from_env_and_other_instances(
        self,
        monkeypatch,
    ):
        """Multiple instances in one process must not clobber each other."""
        rl = AgenticRL(api_key="sk-instance")
        assert rl._resolve_api_key() == "sk-instance"

        # Another tenant overwrites the shared process env ...
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-other-tenant")
        # ... but this instance still resolves its own key.
        assert rl._resolve_api_key() == "sk-instance"

        rl2 = AgenticRL(api_key="sk-instance-2")
        assert rl2._resolve_api_key() == "sk-instance-2"
        assert rl._resolve_api_key() == "sk-instance"

    @pytest.mark.asyncio
    async def test_instance_key_threaded_into_upload(self, monkeypatch):
        """upload_datasets must forward the instance key (leads to file
        requests), not the clobbered env value."""
        rl = AgenticRL(api_key="sk-instance")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-other-tenant")

        rl.tuning = MagicMock(spec=TuningModel)
        rl.tuning.upload_datasets = AsyncMock(return_value=(["t"], ["v"]))

        await rl.upload_datasets(
            training_files=["a.jsonl"],
            validation_files=["b.jsonl"],
        )

        assert (
            rl.tuning.upload_datasets.await_args.kwargs["api_key"]
            == "sk-instance"
        )

    @pytest.mark.asyncio
    async def test_instance_key_threaded_into_register(self, monkeypatch):
        """register_functions must forward the instance key (leads to FC
        requests)."""
        rl = AgenticRL(api_key="sk-instance")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-other-tenant")

        rl.tuning = MagicMock(spec=TuningModel)
        rl.tuning.register_functions = AsyncMock(
            return_value=([], [], [], [], [], []),
        )

        await rl.register_functions(lazy_load=True)

        assert (
            rl.tuning.register_functions.await_args.kwargs["api_key"]
            == "sk-instance"
        )


# ========================================================================== #
#              P1 (§4): Reward Processor instance isolation                   #
# ========================================================================== #


@reward_func("isolation")
class _IsolationProcessor(AbstractRewardProcessor):
    """Reward processor whose sub-function reads per-instance state."""

    def __init__(self, score_value: float, executor=None):
        self.score_value = score_value
        super().__init__(executor=executor)

    # pylint: disable=unused-argument
    @sub_reward_func("metric", sub_weight=1.0)
    def metric(self, input_data: RewardInput) -> RewardOutput:
        return RewardOutput(
            reward=Reward(reward_score=self.score_value),
            status=TaskStatus.SUCCESS,
        )


class TestRewardProcessorInstanceIsolation:
    @pytest.mark.asyncio
    async def test_two_instances_do_not_share_metadata(self):
        """The review repro: creating a 2nd instance must not rebind the
        1st instance's sub-functions (old bug made p1 return p2's value)."""
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[{"role": "assistant", "content": "x"}],
            ),
        )

        p1 = _IsolationProcessor(1.0)
        p2 = _IsolationProcessor(2.0)

        # Metadata objects are per-instance, not the shared class attribute.
        assert p1.reward_meta is not p2.reward_meta
        assert p1.reward_meta is not _IsolationProcessor.reward_meta

        # Each collected sub-function is bound to its own instance.
        assert p1.reward_meta.sub_functions["metric"].func.__self__ is p1
        assert p2.reward_meta.sub_functions["metric"].func.__self__ is p2

        # functions_collected is an instance field, not shared class state.
        assert p1.functions_collected is True
        assert p2.functions_collected is True

        # Behavioral isolation: p1 keeps its own score after p2 exists.
        assert (await p1.process(ri)).reward.reward_score == 1.0
        assert (await p2.process(ri)).reward.reward_score == 2.0
        assert (await p1.process(ri)).reward.reward_score == 1.0


# ========================================================================== #
#                 P1 (§5): FuncManager.execute() entry point                  #
# ========================================================================== #


class TestFuncManagerExecute:
    def test_execute_is_async_and_uses_real_methods(self):
        """execute()/processes() are async and the real method names exist.

        Guards against reintroducing the dead ``parse()`` / ``process()``
        calls that raised ``AttributeError``.
        """
        assert hasattr(FuncManager, "parses")
        assert hasattr(FuncManager, "processes")
        assert not hasattr(FuncManager, "parse")
        assert not hasattr(FuncManager, "process")
        assert inspect.iscoroutinefunction(FuncManager.execute)
        assert inspect.iscoroutinefunction(FuncManager.processes)

    @pytest.mark.asyncio
    async def test_execute_runs_parse_then_process(self):
        manager = FuncManager(
            FunctionType.REWARD,
            processor=DemoRewardProcessor(),
        )
        raw = {
            "agent_output": {
                "messages": [{"role": "assistant", "content": "hello"}],
            },
            "ground_truth": "hello",
        }

        result = await manager.execute(raw)

        assert isinstance(result, RewardOutput)
        assert result.status == TaskStatus.SUCCESS
        # DemoRewardProcessor scores 1.0 when ground_truth is in the messages.
        assert result.reward.reward_score == 1.0
