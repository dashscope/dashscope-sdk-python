# -*- coding: utf-8 -*-
"""Declared Python support must match python_requires, CI and real usage.

Until 1.27.5 this package declared classifiers only through 3.11 while 3.12
accounted for ~80% of downloads. A missing classifier never fails an install,
so nothing downstream surfaced the gap for six months. Each check below pins
one seam that can drift silently.

The helpers live here rather than in a standalone script so the guard runs as
part of the ordinary unit-test job, with no separate CI entry to keep in sync.
"""

import ast
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[3]
SETUP_PY = ROOT / "setup.py"
WORKFLOW = ROOT / ".github" / "workflows" / "unit_test.yml"
PYPISTATS_URL = (
    "https://pypistats.org/api/packages/{pkg}/python_minor?period=day"
)
USER_AGENT = "dashscope-sdk-python/packaging-check"
PREFIX = "Programming Language :: Python :: "
VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")
RECENT_DAYS = 30
THRESHOLD = 1.0

CURRENT = [(3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14)]

# Asking pypistats from all four matrix jobs at once invites a 429 that would
# skip the very check it is meant to run, so only one interpreter asks.
LIVE_CHECK_PYTHON = (3, 12)


def version_key(value: str) -> Tuple[int, int]:
    """Sort "3.9" before "3.10", which float() would get backwards."""
    parts = value.split(".")
    return int(parts[0]), int(parts[1] if len(parts) > 1 else 0)


def setup_kwargs() -> Dict[str, Any]:
    """Read literal setup() keyword args without importing setuptools."""
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    literal = (ast.List, ast.Tuple, ast.Dict, ast.Constant)
    for node in ast.walk(tree):
        func = node.func if isinstance(node, ast.Call) else None
        if not isinstance(func, ast.Attribute) or func.attr != "setup":
            continue
        return {
            kw.arg: ast.literal_eval(kw.value)
            for kw in node.keywords
            if kw.arg is not None and isinstance(kw.value, literal)
        }
    raise AssertionError("no setup() call found in setup.py")


def declared_versions(classifiers: List[str]) -> List[Tuple[int, int]]:
    found = []
    for item in classifiers:
        if not isinstance(item, str) or not item.startswith(PREFIX):
            continue
        match = VERSION_RE.match(item[len(PREFIX) :].strip())
        if match:
            found.append((int(match.group(1)), int(match.group(2))))
    return sorted(set(found))


def requires_floor(spec: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"(\d+)\.(\d+)", spec or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def ci_versions() -> Optional[List[str]]:
    """Collect python-version entries from every CI job matrix."""
    try:
        import yaml  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")) or {}
    found: List[str] = []
    for job in (data.get("jobs") or {}).values():
        matrix = ((job or {}).get("strategy") or {}).get("matrix") or {}
        for key, values in matrix.items():
            if "python" in str(key) and isinstance(values, list):
                found.extend(str(value) for value in values)
    return found


def download_shares(package: str) -> Optional[Dict[str, float]]:
    """Each Python minor's share of downloads over RECENT_DAYS.

    Returns None rather than raising when pypistats is unreachable or answers
    with something unexpected, because a rate limit is not a packaging defect
    and must not be able to turn CI red.
    """
    request = urllib.request.Request(
        PYPISTATS_URL.format(pkg=package),
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            rows = json.load(response)["data"]
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        KeyError,
        ValueError,
    ):
        return None

    recent = set(sorted({row["date"] for row in rows})[-RECENT_DAYS:])
    counts: Dict[str, int] = {}
    total = 0
    for row in rows:
        if row["date"] not in recent:
            continue
        total += row["downloads"]
        category = row["category"]
        counts[category] = counts.get(category, 0) + row["downloads"]
    if not total:
        return None
    return {name: 100.0 * hits / total for name, hits in counts.items()}


def ranked_shares(shares: Dict[str, float]) -> List[Tuple[str, float]]:
    """Download share per Python minor, highest first."""
    return sorted(
        (
            (name, share)
            for name, share in shares.items()
            if VERSION_RE.match(name)
        ),
        key=lambda pair: -pair[1],
    )


def audit(
    declared: List[Tuple[int, int]],
    floor: Optional[Tuple[int, int]],
    tested: Optional[List[str]],
    shares: Optional[Dict[str, float]],
    threshold: float,
) -> List[str]:
    """Return one message per way the support metadata has drifted."""
    if not declared:
        return [f"no '{PREFIX}X.Y' classifiers declared"]

    named = {f"{major}.{minor}" for major, minor in declared}
    violations: List[str] = []

    if floor and floor != declared[0]:
        violations.append(
            f"python_requires floor {floor[0]}.{floor[1]} != lowest "
            f"classifier {declared[0][0]}.{declared[0][1]}",
        )

    for previous, current in zip(declared, declared[1:]):
        if current[0] != previous[0] or current[1] != previous[1] + 1:
            violations.append(
                f"classifier gap between {previous[0]}.{previous[1]} "
                f"and {current[0]}.{current[1]}",
            )

    for version in sorted(set(tested or []), key=version_key):
        if version not in named:
            violations.append(
                f"CI tests Python {version} but no classifier declares it",
            )

    for name, share in ranked_shares(shares or {}):
        if share >= threshold and name not in named:
            violations.append(
                f"Python {name} is {share:.2f}% of downloads "
                f"(>= {threshold:g}%) but has no classifier",
            )
    return violations


def repo_audit(shares: Optional[Dict[str, float]] = None) -> List[str]:
    """Audit this repository's own setup.py against its own CI matrix."""
    kwargs = setup_kwargs()
    return audit(
        declared=declared_versions(list(kwargs.get("classifiers") or [])),
        floor=requires_floor(str(kwargs.get("python_requires", ""))),
        tested=ci_versions(),
        shares=shares,
        threshold=THRESHOLD,
    )


def test_version_key_sorts_minors_numerically():
    # float("3.10") == 3.1 < float("3.9"), so sorting by float inverts these
    ordered = sorted(["3.10", "3.9", "3.14", "3.2"], key=version_key)
    assert ordered == ["3.2", "3.9", "3.10", "3.14"]


def test_declared_versions_ignores_other_classifiers():
    found = declared_versions(
        [
            "Development Status :: 4 - Beta",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.9",
        ],
    )
    assert found == [(3, 9), (3, 12)]


def test_requires_floor_handles_micro_versions():
    assert requires_floor(">=3.9") == (3, 9)
    assert requires_floor(">=3.8.0") == (3, 8)
    assert requires_floor("") is None


def test_audit_flags_undeclared_version_carrying_real_traffic():
    # dashscope <=1.27.4 declared 3.8-3.11 while 3.12 drove most downloads
    violations = audit(
        declared=[(3, 8), (3, 9), (3, 10), (3, 11)],
        floor=(3, 8),
        tested=["3.9", "3.11"],
        shares={"3.11": 24.0, "3.12": 62.0, "null": 0.4},
        threshold=1.0,
    )
    assert any("3.12 is 62.00%" in item for item in violations)


def test_audit_flags_gap_between_declared_versions():
    violations = audit(
        declared=[(3, 9), (3, 11)],
        floor=(3, 9),
        tested=None,
        shares=None,
        threshold=1.0,
    )
    assert any("gap" in item for item in violations)


def test_audit_flags_ci_testing_an_undeclared_version():
    violations = audit(
        declared=[(3, 9), (3, 10)],
        floor=(3, 9),
        tested=["3.9", "3.12"],
        shares=None,
        threshold=1.0,
    )
    assert any("CI tests Python 3.12" in item for item in violations)


def test_audit_flags_floor_above_lowest_classifier():
    violations = audit(
        declared=[(3, 8), (3, 9)],
        floor=(3, 9),
        tested=None,
        shares=None,
        threshold=1.0,
    )
    assert any("python_requires floor" in item for item in violations)


def test_audit_accepts_consistent_metadata():
    violations = audit(
        declared=CURRENT,
        floor=(3, 9),
        tested=["3.9", "3.12", "3.13", "3.14"],
        shares={"3.12": 61.0, "3.15": 0.4, "3.8": 0.09},
        threshold=1.0,
    )
    assert not violations


def test_repo_metadata_passes_offline_checks():
    assert not repo_audit()


@pytest.mark.skipif(
    sys.version_info[:2] != LIVE_CHECK_PYTHON,
    reason="only one matrix job queries pypistats",
)
def test_repo_classifiers_cover_every_python_carrying_real_traffic():
    shares = download_shares("dashscope")
    if shares is None:
        pytest.skip("pypistats unreachable")
    assert not repo_audit(shares)


def test_python_39_config_reader_is_a_core_dependency():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert 'tomli>=1.1.0; python_version < "3.11"' in requirements


@pytest.fixture
def docs_workflow():
    yaml = pytest.importorskip("yaml")
    path = ROOT / ".github" / "workflows" / "docs.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def docs_expression(expression, event_name, ref):
    """Evaluate the trusted workflow's small Python-compatible subset."""
    expression = expression.strip().removeprefix("${{").removesuffix("}}")
    expression = expression.replace("github.event_name", repr(event_name))
    expression = expression.replace("github.ref", repr(ref))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return eval(
        " ".join(expression.split()),
        {
            "__builtins__": {},
            "format": lambda pattern, value: pattern.format(value),
        },
    )


def test_docs_release_dispatches_main_with_scoped_permissions(docs_workflow):
    # PyYAML's YAML 1.1 loader may interpret the GitHub Actions "on" as True.
    triggers = docs_workflow.get("on", docs_workflow.get(True))
    assert "workflow_dispatch" in triggers
    assert triggers["release"]["types"] == ["published"]
    dispatch = docs_workflow["jobs"]["dispatch"]
    assert dispatch["if"] == "github.event_name == 'release'"
    assert dispatch["permissions"] == {"actions": "write"}
    step = dispatch["steps"][0]
    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert step["run"] == (
        'gh workflow run docs.yml --repo "$GITHUB_REPOSITORY" --ref main'
    )
    assert docs_workflow["permissions"] == {"contents": "read"}


@pytest.mark.parametrize(
    "event_name,ref,build_expected,deploy_expected",
    [
        ("release", "refs/tags/v2.1.2", False, False),
        ("push", "refs/heads/main", True, True),
        ("workflow_dispatch", "refs/heads/main", True, True),
        ("workflow_dispatch", "refs/heads/dev/v2.1.2-2", True, False),
        ("pull_request", "refs/pull/242/merge", True, False),
    ],
)
def test_docs_deploys_only_main_pushes_and_dispatches(
    docs_workflow, event_name, ref, build_expected, deploy_expected
):
    jobs = docs_workflow["jobs"]
    assert docs_expression(jobs["build"]["if"], event_name, ref) is (
        build_expected
    )
    upload = next(
        step
        for step in jobs["build"]["steps"]
        if step.get("uses", "").startswith("actions/upload-pages-artifact@")
    )
    for condition in (upload["if"], jobs["deploy"]["if"]):
        assert docs_expression(condition, event_name, ref) is deploy_expected


def test_docs_deployments_share_concurrency_without_cancelling_dispatcher(
    docs_workflow,
):
    concurrency = docs_workflow["concurrency"]
    assert concurrency["cancel-in-progress"] is True
    expression = concurrency["group"]
    push = docs_expression(expression, "push", "refs/heads/main")
    dispatched = docs_expression(
        expression, "workflow_dispatch", "refs/heads/main"
    )
    release = docs_expression(expression, "release", "refs/tags/v2.1.2")
    first_pr = docs_expression(
        expression, "pull_request", "refs/pull/242/merge"
    )
    second_pr = docs_expression(
        expression, "pull_request", "refs/pull/243/merge"
    )
    deploy_condition = docs_workflow["jobs"]["deploy"]["if"]
    for event_name, ref in (
        ("push", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/main"),
        ("release", "refs/tags/v2.1.2"),
        ("release", "refs/tags/v2.1.3"),
    ):
        if docs_expression(deploy_condition, event_name, ref):
            assert docs_expression(expression, event_name, ref) == push
    assert push == dispatched
    assert len({push, release, first_pr, second_pr}) == 4


def test_build_py_removes_stale_deleted_packages(tmp_path):
    """Incremental wheel builds must not resurrect pre-migration modules.

    A dirty ``build/lib`` can otherwise contain both ``dashscope/client.py``
    and the deleted ``dashscope/client/__init__.py`` package. Python chooses
    the stale package and ``from dashscope.client import Client`` then fails.
    """
    build_lib = tmp_path / "build-lib"
    stale_files = (
        build_lib / "dashscope" / "client" / "__init__.py",
        build_lib / "dashscope" / "aigc" / "__init__.py",
        build_lib / "dashscope" / "resources" / "qwen.tiktoken",
    )
    for stale in stale_files:
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("# stale pre-2.0 build artifact\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build_py",
            "--build-lib",
            str(build_lib),
            "--force",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build_lib / "dashscope" / "client.py").is_file()
    for stale in stale_files:
        assert not stale.exists(), stale
