# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Self-diagnostic errors for the dashscope 2.0 SDK: local rule table.

A failed call gets a structured, human-friendly diagnosis matched
locally from ``(status_code, error code, message)``. No network, no
model invocation -- the DashScope error surface is finite and
enumerable, so a static rule table covers the high-frequency cases with
zero latency and zero privacy exposure.

The diagnosis is advisory data, never printed by the SDK itself: it
rides on ``DashScopeError.diagnosis`` (and on the non-raising response
envelope's ``.diagnosis``), and the CLI or :func:`install_excepthook`
renders it to stderr.

Rules are bilingual: ``DASHSCOPE_DIAGNOSIS_LANG=en`` selects English,
default is Chinese. Every rendered block ends with an escalation hint to
the built-in AI expert (``dashscope expert "<error>"``).

This module is part of :mod:`dashscope.devkit`.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

LANG_ENV = "DASHSCOPE_DIAGNOSIS_LANG"


@dataclass(frozen=True)
class Matcher:
    """All non-empty dimensions must match (AND); empty means wildcard."""

    status_codes: Tuple[int, ...] = ()
    code_keywords: Tuple[str, ...] = ()
    message_patterns: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Diagnosis:
    """A matched diagnosis: title, (cause, fix) pairs, optional command."""

    title: str
    fixes: Tuple[Tuple[str, str], ...]
    command: str = ""


@dataclass(frozen=True)
class Rule:
    name: str
    matchers: Tuple[Matcher, ...]
    diagnosis_zh: Diagnosis = field(compare=False)
    diagnosis_en: Diagnosis = field(compare=False)


RULES: Tuple[Rule, ...] = (
    Rule(
        name="auth",
        matchers=(
            Matcher(status_codes=(401,)),
            Matcher(code_keywords=("invalidapikey", "unauthorized")),
            Matcher(message_patterns=(r"no api key provided",)),
        ),
        diagnosis_zh=Diagnosis(
            title="鉴权失败：API Key 缺失或无效",
            fixes=(
                (
                    "未配置 API Key，或环境变量未生效",
                    "设置环境变量 DASHSCOPE_API_KEY，"
                    + "或在代码中 dashscope.Client(api_key='sk-...')",
                ),
                (
                    "Key 拼写错误或已被吊销",
                    "到百炼控制台「API-KEY 管理」页重新生成并替换",
                ),
                (
                    "Key 与目标 Workspace 不匹配",
                    "确认 Key 属于目标工作空间，或改用 "
                    + "DASHSCOPE_WORKSPACE / Client(workspace=...) 指定",
                ),
            ),
            command="echo $DASHSCOPE_API_KEY  # 确认环境变量已生效",
        ),
        diagnosis_en=Diagnosis(
            title="Authentication failed: API key missing or invalid",
            fixes=(
                (
                    "API key not configured, or the environment "
                    + "variable did not take effect",
                    "set DASHSCOPE_API_KEY, or pass "
                    + "dashscope.Client(api_key='sk-...')",
                ),
                (
                    "Key misspelled or revoked",
                    "regenerate it on the DashScope console "
                    + "(API-KEY management) and replace it",
                ),
                (
                    "Key does not belong to the target workspace",
                    "confirm the key's workspace, or set "
                    + "DASHSCOPE_WORKSPACE / Client(workspace=...)",
                ),
            ),
            command="echo $DASHSCOPE_API_KEY  # verify the env var is set",
        ),
    ),
    Rule(
        name="workspace",
        matchers=(
            Matcher(status_codes=(400,), message_patterns=(r"workspace",)),
            Matcher(code_keywords=("workspace",)),
        ),
        diagnosis_zh=Diagnosis(
            title="请求未匹配到有效的百炼工作空间（Workspace）",
            fixes=(
                (
                    "地域（Region）不匹配：当前走默认北京地域，" + "资源可能在新加坡等其他地域",
                    "构造客户端时指定 " + "dashscope.Client(region='ap-southeast-1')",
                ),
                (
                    "多租户模式下缺少 Workspace 标识",
                    "设置环境变量 DASHSCOPE_WORKSPACE，"
                    + "或在 Client(workspace=...) / 调用时传入",
                ),
            ),
            command="echo $DASHSCOPE_WORKSPACE  # 确认工作空间已配置",
        ),
        diagnosis_en=Diagnosis(
            title="Request did not match a valid workspace",
            fixes=(
                (
                    "Region mismatch: the request went to the default "
                    + "Beijing region while your resources live "
                    + "elsewhere (e.g. Singapore)",
                    "build the client with "
                    + "dashscope.Client(region='ap-southeast-1')",
                ),
                (
                    "Missing workspace id in multi-tenant mode",
                    "set DASHSCOPE_WORKSPACE or pass "
                    + "Client(workspace=...) / the workspace argument",
                ),
            ),
            command="echo $DASHSCOPE_WORKSPACE",
        ),
    ),
    Rule(
        name="throttling",
        matchers=(
            Matcher(status_codes=(429,)),
            Matcher(code_keywords=("throttling", "ratelimit", "quota")),
        ),
        diagnosis_zh=Diagnosis(
            title="触发限流或配额耗尽",
            fixes=(
                (
                    "请求速率超过账号 QPS 上限",
                    "降低并发，并为客户端增加指数退避重试",
                ),
                (
                    "账户额度（Quota）用尽",
                    "到百炼控制台检查额度与欠费状态",
                ),
            ),
        ),
        diagnosis_en=Diagnosis(
            title="Throttled or out of quota",
            fixes=(
                (
                    "Request rate exceeded the account QPS limit",
                    "lower concurrency and retry with "
                    + "exponential backoff",
                ),
                (
                    "Account quota exhausted",
                    "check quota and billing status on the "
                    + "DashScope console",
                ),
            ),
        ),
    ),
    Rule(
        name="model_not_found",
        matchers=(
            Matcher(code_keywords=("model.notexist", "invalidmodel")),
            Matcher(message_patterns=(r"model.*not exist",)),
        ),
        diagnosis_zh=Diagnosis(
            title="模型不存在或名称有误",
            fixes=(
                (
                    "模型名拼写错误",
                    "对照百炼控制台「模型列表」核对拼写"
                    + "（注意版本后缀，如 qwen-max 与 qwen-max-latest）",
                ),
                (
                    "模型在当前地域未上线",
                    "确认目标地域提供该模型，必要时切换 region",
                ),
            ),
        ),
        diagnosis_en=Diagnosis(
            title="Model does not exist or the name is wrong",
            fixes=(
                (
                    "Model name misspelled",
                    "check the console model list (mind version "
                    + "suffixes, e.g. qwen-max vs qwen-max-latest)",
                ),
                (
                    "Model not available in the current region",
                    "confirm the region offers this model; "
                    + "switch region if needed",
                ),
            ),
        ),
    ),
    Rule(
        name="model_unauthorized",
        matchers=(
            Matcher(status_codes=(403,)),
            Matcher(code_keywords=("unauthorized", "accessdenied")),
            Matcher(message_patterns=(r"not authorized", r"permission")),
        ),
        diagnosis_zh=Diagnosis(
            title="无权限访问该模型或资源",
            fixes=(
                (
                    "模型未在当前工作空间开通",
                    "到百炼控制台为目标 Workspace 开通该模型",
                ),
                (
                    "子账号缺少授权",
                    "联系主账号在 RAM 中授予百炼调用权限",
                ),
            ),
        ),
        diagnosis_en=Diagnosis(
            title="No permission for this model or resource",
            fixes=(
                (
                    "Model not activated for the current workspace",
                    "activate it on the DashScope console for "
                    + "the target workspace",
                ),
                (
                    "Sub-account lacks authorization",
                    "ask the primary account to grant DashScope "
                    + "access in RAM",
                ),
            ),
        ),
    ),
    Rule(
        name="content_moderation",
        matchers=(
            Matcher(
                code_keywords=("datainspection", "contentmoderation"),
            ),
            Matcher(
                message_patterns=(
                    r"inappropriate content",
                    r"data inspection",
                ),
            ),
        ),
        diagnosis_zh=Diagnosis(
            title="输入或输出触发了内容安全审核",
            fixes=(
                (
                    "prompt 或生成内容命中审核策略",
                    "检查输入文本；若属误判，缩小输入范围定位触发片段",
                ),
            ),
        ),
        diagnosis_en=Diagnosis(
            title="Input or output triggered content moderation",
            fixes=(
                (
                    "prompt or generated content hit the "
                    + "moderation policy",
                    "review the input; if it looks like a false "
                    + "positive, bisect the input to locate the "
                    + "triggering fragment",
                ),
            ),
        ),
    ),
    Rule(
        name="invalid_parameter",
        matchers=(
            Matcher(status_codes=(400,)),
            Matcher(code_keywords=("invalidparameter", "invalidinput")),
        ),
        diagnosis_zh=Diagnosis(
            title="请求参数不合法",
            fixes=(
                (
                    "参数取值越界或类型错误",
                    "对照 API 文档检查取值范围"
                    + "（如 temperature ∈ [0, 2)、top_p ∈ (0, 1]）",
                ),
                (
                    "输入结构不符合协议",
                    "检查 messages/prompt 结构；" + "多模态输入确认 URL 或本地路径可读",
                ),
            ),
        ),
        diagnosis_en=Diagnosis(
            title="Invalid request parameters",
            fixes=(
                (
                    "Parameter value out of range or wrong type",
                    "check ranges in the API docs (e.g. "
                    + "temperature in [0, 2), top_p in (0, 1])",
                ),
                (
                    "Input structure does not match the protocol",
                    "check the messages/prompt structure; for "
                    + "multimodal input confirm the URL or local "
                    + "path is readable",
                ),
            ),
        ),
    ),
    Rule(
        name="network",
        matchers=(
            Matcher(code_keywords=("gatewaytimeout",)),
            Matcher(
                message_patterns=(
                    r"timed? ?out",
                    r"connection",
                    r"max retries exceeded",
                ),
            ),
        ),
        diagnosis_zh=Diagnosis(
            title="网络连通性问题",
            fixes=(
                (
                    "到 dashscope.aliyuncs.com 的链路不通或被代理拦截",
                    "检查代理/防火墙；海外地域资源确认已用 " + "Client(region=...) 指定地域",
                ),
                (
                    "服务端响应过慢导致读取超时",
                    "减小 max_tokens 或改用 stream=True 流式读取",
                ),
            ),
        ),
        diagnosis_en=Diagnosis(
            title="Network connectivity problem",
            fixes=(
                (
                    "Link to dashscope.aliyuncs.com is blocked or "
                    + "proxied",
                    "check proxy/firewall; for overseas regions "
                    + "build the client with Client(region=...)",
                ),
                (
                    "Server too slow; read timed out",
                    "lower max_tokens or switch to stream=True",
                ),
            ),
        ),
    ),
    Rule(
        name="internal_error",
        matchers=(
            Matcher(status_codes=(500, 502, 503)),
            Matcher(code_keywords=("internalerror", "serviceunavailable")),
        ),
        diagnosis_zh=Diagnosis(
            title="服务端内部错误",
            fixes=(
                (
                    "平台侧临时故障",
                    "指数退避重试；持续失败则到阿里云控制台查看" + "服务健康状态或提工单",
                ),
            ),
        ),
        diagnosis_en=Diagnosis(
            title="Server-side internal error",
            fixes=(
                (
                    "Transient platform fault",
                    "retry with exponential backoff; if it "
                    + "persists, check service health on the "
                    + "Alibaba Cloud console or file a ticket",
                ),
            ),
        ),
    ),
    Rule(
        name="arrears",
        matchers=(Matcher(code_keywords=("arrearage",)),),
        diagnosis_zh=Diagnosis(
            title="账户欠费，调用已被停用",
            fixes=(
                (
                    "账户余额不足或已欠费",
                    "到阿里云 / 百炼控制台充值，" + "确认账号状态恢复正常后重试",
                ),
            ),
        ),
        diagnosis_en=Diagnosis(
            title="Account in arrears; calls suspended",
            fixes=(
                (
                    "Insufficient balance or overdue account",
                    "top up on the Alibaba Cloud / DashScope console "
                    + "and confirm the account is in good standing",
                ),
            ),
        ),
    ),
)


def _lang() -> str:
    return (
        "en" if os.environ.get(LANG_ENV, "").lower().startswith("en") else "zh"
    )


def _match_one(
    matcher: Matcher,
    status_code: Optional[int],
    code_lower: str,
    message: str,
) -> bool:
    if not (
        matcher.status_codes
        or matcher.code_keywords
        or matcher.message_patterns
    ):
        return False
    if matcher.status_codes and status_code not in matcher.status_codes:
        return False
    if matcher.code_keywords and not any(
        kw in code_lower for kw in matcher.code_keywords
    ):
        return False
    return not matcher.message_patterns or any(
        re.search(p, message, re.IGNORECASE) for p in matcher.message_patterns
    )


def diagnose(
    status_code: Optional[int] = None,
    code: Optional[str] = None,
    message: Optional[str] = None,
) -> Optional[Diagnosis]:
    """Match a failure against the local rule table.

    First matching rule wins; rules are ordered most-specific first.
    The returned Diagnosis is in the language selected by
    ``DASHSCOPE_DIAGNOSIS_LANG`` (default zh).
    """
    code_lower = (code or "").lower()
    msg = str(message or "")
    for rule in RULES:
        if any(
            _match_one(m, status_code, code_lower, msg) for m in rule.matchers
        ):
            if _lang() == "en":
                return rule.diagnosis_en
            return rule.diagnosis_zh
    return None


_WIDTH = 80

_EXPERT_HINT = {
    "zh": "🤖 仍未解决？运行 dashscope 进入交互提问，" '或一次性提问：dashscope expert "<贴入错误信息>"',
    "en": "🤖 Still stuck? Run dashscope for interactive Q&A, "
    'or ask once: dashscope expert "<paste the error>"',
}


def render(diagnosis: Diagnosis, lang: Optional[str] = None) -> str:
    """Render a diagnosis as a terminal block.

    Ends with an escalation hint to the built-in AI expert.
    """
    lang = lang or _lang()
    if lang == "zh":
        title_line = f"💡 [DashScope 诊断建议] {diagnosis.title}"
        causes_line = "🔍 可能的原因与修复方法："
    else:
        title_line = f"💡 [DashScope Diagnosis] {diagnosis.title}"
        causes_line = "🔍 Possible causes and fixes:"
    lines = ["=" * _WIDTH, title_line, "", causes_line]
    for i, (cause, fix) in enumerate(diagnosis.fixes, 1):
        lines.append(f"  {i}. {cause}")
        lines.append(f"     👉 {fix}")
    if diagnosis.command:
        label = "🛠️ 快捷命令：" if lang == "zh" else "🛠️ Quick check: "
        lines += ["", f"{label}{diagnosis.command}"]
    lines += ["", _EXPERT_HINT[lang]]
    lines.append("=" * _WIDTH)
    return "\n".join(lines)


_installed = False


def install_excepthook() -> None:
    """Append the diagnosis block to uncaught tracebacks (stderr only).

    The normal traceback still prints first; the diagnosis is advisory
    output appended after it. Idempotent.

    Chains to whatever ``sys.excepthook`` was installed before this call
    (not ``sys.__excepthook__``), so a third-party hook already in place
    -- pytest, IPython, an APM agent -- keeps running instead of being
    silently dropped.
    """
    global _installed
    if _installed:
        return
    _installed = True

    previous = sys.excepthook

    def _hook(exc_type, exc, tb):
        previous(exc_type, exc, tb)
        diag = getattr(exc, "diagnosis", None)
        if diag is not None:
            print(render(diag), file=sys.stderr)

    sys.excepthook = _hook
