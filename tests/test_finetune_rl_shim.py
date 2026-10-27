# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The finetune RL implementation and its 1.x compatibility surface.

``dashscope.applications.finetune`` owns the single workflow core.
The historical ``dashscope.finetune`` paths expose the 1.x facade, while
``client.rl`` exposes the 2.0 facade. The tests also pin Function Compute,
``uvicorn`` / ``python -m``, lazy optional dependencies and the submit wire
format.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib
import platform
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import dashscope
from dashscope._shared import _sdk_ident
from dashscope._shared._sdk_ident import set_sdk_client
from dashscope.legacy_1x.finetune.agentic_rl import (
    AgenticRL as LegacyAgenticRL,
)
from dashscope.version import __version__
from dashscope.errors import SdkError
from dashscope.applications.finetune import _submit
from dashscope.applications.finetune.agentic_rl import AgenticRL as V2AgenticRL
from dashscope.applications.finetune.agentic_rl import (
    SubmitOutput,
    SubmitResult,
)
from dashscope.applications.finetune.reinforcement import (
    DASHSCOPE_HTTP_BASE_URL,
)
from dashscope.applications.finetune.reinforcement import TrainingType
from dashscope.applications.finetune.reinforcement.common.errors import (
    _LEGACY_ERROR_CODES,
    AgenticRLError,
    RuntimeErrorWithCode,
    ValueErrorWithCode,
)
from dashscope.resources import reinforcement_learning as _rl_mount

PUBLIC = "dashscope.finetune.reinforcement"
V2 = "dashscope.applications.finetune.reinforcement"
PATHS = [PUBLIC, V2]

ENVELOPE = {
    "status_code": 200,
    "request_id": "req-1",
    "code": None,
    "message": "",
    "output": {
        "job_id": "ft-1",
        "status": "RUNNING",
        "create_time": "2026-01-02 03:04:05",
        "finetuned_output": "qwen-ft",
    },
    "usage": None,
    "headers": {"x-request-id": "h-1"},
}


class _StubTuning:
    """The smallest ``tuning`` that ``_build_submit_request`` accepts."""

    def __init__(self):
        self.teacher_model = None
        self.name = "stub-job"
        self.model = SimpleNamespace(name="qwen-test")
        self.functions = []
        self.datasets = []
        self.training = SimpleNamespace(
            hyper_parameters=None,
            resources=None,
            type=TrainingType.TRAINING_TYPE,
        )

    def combine_ids_runtimes(self, functype, id_str=None):
        return []

    def check_function_names(self):
        return True


def _stub(cls, api_key="sk-submit"):
    # ``set_api_key`` writes DASHSCOPE_API_KEY for real; these tests are
    # about the submit call, not about credential discovery.
    with patch("dashscope.applications.finetune.agentic_rl.set_api_key"):
        instance = cls(api_key=api_key)
    instance.tuning = _StubTuning()
    return instance


class _FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    content = b""

    @staticmethod
    def json():
        return {"request_id": "req-1", "output": {"job_id": "ft-1"}}


@pytest.fixture(name="post_calls")
def fixture_post_calls(monkeypatch):
    """Record every ``post_submit`` call instead of hitting the network."""
    calls = []

    # ``_SDK_CLIENT`` is process-wide mutable state, and importing the CLI
    # (which the compatibility-path ``app`` test does) marks the whole
    # process ``python-cli``. Pinning it here keeps these
    # assertions independent of test order; monkeypatch restores whatever
    # the ambient value was.
    monkeypatch.setattr(_sdk_ident, "_SDK_CLIENT", "python-sdk")

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def post(self, url, data=None, headers=None, timeout=None):
            calls.append(
                {
                    "url": url,
                    "data": data,
                    "headers": headers,
                    "timeout": timeout,
                },
            )
            return _FakeResponse()

    monkeypatch.setattr(_submit.requests, "Session", _Session)
    return calls


# -- the two AgenticRL surfaces -------------------------------------------


def test_legacy_agentic_rl_subclasses_the_v2_one():
    assert LegacyAgenticRL is not V2AgenticRL
    assert issubclass(LegacyAgenticRL, V2AgenticRL)


def test_only_the_1x_surface_keeps_the_job_verbs():
    assert "CreateMixin" not in [c.__name__ for c in V2AgenticRL.__mro__]
    for verb in ("cancel", "list", "get", "delete", "logs"):
        assert verb not in V2AgenticRL.__dict__, verb
        assert verb in LegacyAgenticRL.__dict__, verb


# -- laziness -------------------------------------------------------------


@pytest.mark.parametrize("alias", PATHS)
def test_server_module_resolves_without_importing_fastapi(alias):
    submit_path = f"{alias}.component.server.server"
    spec = importlib.util.find_spec(submit_path)
    assert spec is not None
    assert (
        "dashscope/applications/finetune/reinforcement"
        in spec.origin.replace(
            "\\",
            "/",
        )
    )
    assert spec.loader.get_code(submit_path) is not None
    assert "fastapi" not in sys.modules


@pytest.mark.parametrize("alias", PATHS)
def test_python_dash_m_resolves_the_compatibility_server_path(alias):
    # ``runpy`` refuses a loader that yields no code object, which is how
    # examples/workspace/scripts/start_local_*.sh launches the server.
    submit_path = f"{alias}.component.server.server"
    name, _, code = runpy._get_module_details(submit_path)
    assert name == submit_path
    assert code is not None
    assert "fastapi" not in sys.modules


@pytest.mark.skipif(
    importlib.util.find_spec("opentelemetry") is not None,
    reason="opentelemetry is installed, so the guard cannot be observed",
)
def test_genai_stays_behind_its_optional_dependency_on_all_paths():
    # ``genai/__init__`` imports ``llm_dashscope`` eagerly, which needs
    # opentelemetry. All spellings must fail identically rather than one
    # quietly succeeding through the alias.
    for alias in [V2]:
        name = f"{alias}.component.observability.genai.llm_dashscope"
        with pytest.raises(ModuleNotFoundError) as ei:
            importlib.import_module(name)
        assert "opentelemetry" in str(ei.value)


def test_app_is_reachable_lazily_from_the_public_path():
    from dashscope.legacy_1x.cli.agentic_rl import app as cli_app

    for alias in PATHS:
        assert importlib.import_module(alias).app is cli_app


# -- the submit POST ------------------------------------------------------


def test_submit_post_keeps_the_published_wire_format(post_calls):
    _stub(V2AgenticRL).submit_job(job_name="计算任务")

    assert len(post_calls) == 1
    call = post_calls[0]
    assert call["url"] == f"{DASHSCOPE_HTTP_BASE_URL}/fine-tunes"
    assert call["timeout"] == _submit.SUBMIT_TIMEOUT

    headers = call["headers"]
    # Insertion order is part of the contract: it is the order 1.x's
    # CreateMixin produced, and the backend sees the same bytes.
    assert list(headers) == [
        "Content-Type",
        "user-agent",
        "x-dashscope-sdk-client",
        "x-dashscope-sdk-session-id",
        "Authorization",
        "Accept",
    ]
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Accept"] == headers["Content-Type"]
    assert headers["Authorization"] == "Bearer sk-submit"

    # Unmarked module tag: the v2 transport's ``finetune2`` belongs to
    # client.rl.jobs, not to the submit path.
    assert headers["x-dashscope-sdk-client"] == (
        f"python-sdk/{__version__}/finetune"
    )

    user_agent = headers["user-agent"]
    assert user_agent.startswith("dashscope/")
    assert f"python/{platform.python_version()}" in user_agent

    body = call["data"]
    assert isinstance(body, bytes)
    payload = json.loads(body.decode("utf-8"))
    # ensure_ascii=False: a non-ASCII job name must not be escaped.
    assert "计算任务".encode("utf-8") in body
    assert payload["job_name"].startswith("计算任务-")
    assert payload["model"] == "qwen-test"
    assert payload["training_type"] == str(TrainingType.TRAINING_TYPE)
    assert "teacher_model" not in payload


def test_cli_sdk_client_override_reaches_the_submit_tag(post_calls):
    # ``set_sdk_client`` is the CLI's own entry point into the mutable
    # global; the ``post_calls`` fixture pins and restores it.
    set_sdk_client("python-cli")

    _stub(V2AgenticRL).submit_job()

    assert post_calls[0]["headers"]["x-dashscope-sdk-client"] == (
        f"python-cli/{__version__}/finetune"
    )


def test_sdk_header_kill_switch_still_applies(monkeypatch, post_calls):
    monkeypatch.setenv("DASHSCOPE_DISABLE_SDK_HEADERS", "1")

    _stub(V2AgenticRL).submit_job()

    headers = post_calls[0]["headers"]
    assert "x-dashscope-sdk-client" not in headers
    assert "x-dashscope-sdk-session-id" not in headers
    assert headers["Authorization"] == "Bearer sk-submit"


def test_session_id_is_one_value_across_both_stacks():
    from dashscope.legacy_1x.common.utils import get_sdk_headers
    from dashscope.transport._ident import sdk_headers

    legacy = get_sdk_headers("finetune")["x-dashscope-sdk-session-id"]
    v2 = sdk_headers("finetune")["x-dashscope-sdk-session-id"]
    assert legacy == v2


# -- api key resolution ---------------------------------------------------


def test_submit_ignores_the_1x_root_api_key_global(monkeypatch, post_calls):
    monkeypatch.setattr(dashscope, "api_key", "sk-root-global", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env")

    rl = _stub(V2AgenticRL, api_key="sk-submit")
    assert rl._api_key == "sk-submit"
    rl.submit_job()

    assert post_calls[0]["headers"]["Authorization"] == "Bearer sk-submit"


def test_submit_falls_back_to_the_environment(monkeypatch, post_calls):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env")

    rl = _stub(V2AgenticRL, api_key=None)
    assert rl._api_key is None
    rl.submit_job()

    assert post_calls[0]["headers"]["Authorization"] == "Bearer sk-env"


# -- the two return types -------------------------------------------------


def test_current_submit_returns_submit_result_and_feeds_to_job():
    with patch(
        "dashscope.applications.finetune._submit.post_submit",
        return_value=dict(ENVELOPE),
    ):
        result = _stub(V2AgenticRL).submit_job()

    assert isinstance(result, SubmitResult)
    assert isinstance(result.output, SubmitOutput)
    assert result.output.job_id == "ft-1"
    assert result.request_id == "req-1"
    assert result.headers == {"x-request-id": "h-1"}

    job = _rl_mount._to_job(result)
    assert job.job_id == "ft-1"
    assert job.status == "RUNNING"
    # The v2 model's aliases absorb the submit's 1.x-era field names.
    assert job.created_at == "2026-01-02 03:04:05"
    assert job.finetuned_model == "qwen-ft"
    assert job.request_id == "req-1"


def test_legacy_submit_returns_the_finetune_dataclass():
    from dashscope.legacy_1x.finetune.customize_types import FineTune

    with patch(
        "dashscope.applications.finetune._submit.post_submit",
        return_value=dict(ENVELOPE),
    ):
        result = _stub(LegacyAgenticRL).submit_job()

    assert isinstance(result, FineTune)
    assert result.status_code == 200
    assert result.output.job_id == "ft-1"
    assert result.output.finetuned_output == "qwen-ft"


def test_a_failed_submit_is_an_envelope_not_an_exception():
    failure = {
        "status_code": 400,
        "request_id": "req-2",
        "code": "InvalidParameter",
        "message": "bad job",
        "output": None,
        "usage": None,
        "headers": {},
    }
    with patch(
        "dashscope.applications.finetune._submit.post_submit",
        return_value=dict(failure),
    ):
        result = _stub(V2AgenticRL).submit_job()

    assert result.status_code == 400
    assert result.code == "InvalidParameter"
    assert result.output is None


# -- single implementation ------------------------------------------------


def test_application_package_contains_the_unique_implementation():
    public = importlib.import_module(V2)
    package_dir = pathlib.Path(public.__file__).parent
    assert package_dir.name == "reinforcement"
    assert package_dir.parent.name == "finetune"
    assert len(list(package_dir.rglob("*.py"))) > 1
    assert importlib.util.find_spec("dashscope.reinforcement") is None
    assert not (package_dir.parent.parent / "agentic_rl.py").exists()


def test_the_v2_mount_points_at_the_v2_workflow_modules():
    assert _rl_mount._agentic_rl is importlib.import_module(
        "dashscope.applications.finetune.agentic_rl",
    )
    assert _rl_mount._reinforcement is importlib.import_module(V2)


def test_the_two_rl_error_modules_are_versioned():
    legacy_errors = importlib.import_module(f"{PUBLIC}.common.errors")
    canonical_errors = importlib.import_module(f"{V2}.common.errors")
    assert legacy_errors is not canonical_errors
    assert legacy_errors.ValueErrorWithCode is not ValueErrorWithCode
    assert canonical_errors.ValueErrorWithCode is ValueErrorWithCode


@pytest.mark.parametrize(
    ("catalog_name", "legacy_code"),
    _LEGACY_ERROR_CODES.items(),
)
def test_rl_errors_have_native_catalog_and_legacy_code_defaults(
    catalog_name,
    legacy_code,
):
    errors = importlib.import_module(f"{V2}.common.errors")
    cls_name = catalog_name.rsplit(".", 1)[1]
    cls = AgenticRLError if cls_name == "Error" else getattr(errors, cls_name)
    error = cls("failure")

    assert isinstance(error, SdkError)
    assert error.code == catalog_name
    assert error.error_code == legacy_code
    assert error.status_code is None


def test_explicit_legacy_code_overrides_the_class_default():
    error = ValueErrorWithCode("duplicate names", error_code=3004)

    assert error.code == "sdk.agentic_rl.ValueErrorWithCode"
    assert error.error_code == 3004
    assert error.message == "duplicate names"


def test_condition_only_catalog_entries_do_not_need_new_classes():
    error = AgenticRLError.from_catalog(
        "sdk.agentic_rl.DuplicateFunctionNames",
        error_code=3004,
        variables={"names": "rollout"},
    )

    assert type(error) is AgenticRLError
    assert error.code == "sdk.agentic_rl.DuplicateFunctionNames"
    assert error.message == "Duplicate function names detected: rollout."
    assert error.error_code == 3004
    assert error.allow_retry is False
    assert "unique name" in error.solution


def test_legacy_builtin_exception_inheritance_is_retained():
    # 1.x contract: these stayed catchable as the builtins they wrap, so
    # ``except RuntimeError`` / ``except ValueError`` keep working, while the
    # v2 unification means they are still ``AgenticRLError`` too.
    assert issubclass(RuntimeErrorWithCode, RuntimeError)
    assert issubclass(ValueErrorWithCode, ValueError)
    assert issubclass(RuntimeErrorWithCode, AgenticRLError)
    assert issubclass(ValueErrorWithCode, AgenticRLError)

    with pytest.raises(ValueError):
        raise ValueErrorWithCode("bad value")
    with pytest.raises(RuntimeError):
        raise RuntimeErrorWithCode("bad state")
