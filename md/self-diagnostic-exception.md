# 智能诊断异常（Self-Diagnostic Exception）— dashscope 2.0 设计

> 状态：已落地（v2，Tier 0）。对应实现：`dashscope/v2/devkit/diagnosis.py`，挂载点 `dashscope/v2/errors.py`（`DashScopeError.diagnosis`）与 `dashscope/v2/types/common.py`（`ResponseBase.diagnosis`）。
> 来源：1.x 提案 `self-diagnostic-exception`，针对 2.0 错误体系重写。

## 1. 需求与目标

传统报错只抛冷冰冰的 HTTP 状态码（400/429），开发者要复制错误去查文档或问 AI。目标：捕获失败响应时，SDK 在本地匹配出结构化、「有人情味」的修复建议，零延迟、零成本、离线可用、无隐私外发。

终端体感（新加坡地域未配 Workspace 导致报错）：

```text
================================================================================
💡 [DashScope 智能诊断] 请求未匹配到有效的百炼工作空间（Workspace）

🔍 可能的原因与修复方法：
  1. 地域（Region）不匹配：当前走默认北京地域，资源可能在新加坡等其他地域
     👉 构造客户端时指定 dashscope.Client(region='ap-southeast-1')
  2. 多租户模式下缺少 Workspace 标识
     👉 设置环境变量 DASHSCOPE_WORKSPACE_ID，或在 Client(workspace=...) / 调用时传入

🛠️ 快捷命令：echo $DASHSCOPE_WORKSPACE_ID  # 确认工作空间已配置

🤖 仍未解决？运行 dashscope 进入交互提问，或一次性提问：dashscope expert "<贴入错误信息>"
================================================================================
```

## 2. 可行性结论

**可行，且默认路径不需要额外大模型调用。** DashScope 错误码有限可枚举，一张本地规则表（`(status_code, error code, message pattern)` → 结构化修复建议）即可覆盖绝大多数高频场景。业界先例全是静态规则（Python 3.12 "Did you mean"、Azure CLI 命令建议、Stripe 错误里的文档链接），没有一家在异常路径上同步调大模型。

## 3. 接入点：2.0 与 1.x 的差异

1.x 挂在 `DashScopeException` 与 `DashScopeAPIResponse` 两处（分别是抛出路径与返回路径）。

2.0 是双层错误模型，对应两个挂载点：

- **抛出路径**（`raise_on_error=True`，默认）：所有错误都是 `dashscope.v2.errors.DashScopeError` 子类，携带 `.status_code` / `.code` / `.message`。在基类加 `.diagnosis` 属性，全部子类（含 `SdkError`、`APIStatusError` 家族、网络类 `APIConnectionError`/`APITimeoutError`）自动继承。
- **返回路径**（`raise_on_error=False`）：失败以错误信封返回，即 `ResponseBase`（pydantic）带 `code`/`message`/`status_code`/`is_error`。在 `ResponseBase` 加 `.diagnosis` 属性，仅当 `is_error` 为真时匹配，成功响应返回 `None`、零开销。

两处都**懒导入** `diagnosis.diagnose`，成功路径不加载 devkit、不引入循环依赖（`diagnosis.py` 不反向依赖 errors/types）。

## 4. 工程约束（必须避开的坑）

1. **SDK 不直接 print**：库直接打终端会污染 stdout、破坏可组合性。诊断以结构化数据挂在对象上（`err.diagnosis` / `resp.diagnosis`），由 CLI 或 `install_excepthook()` 渲染，且只写 stderr。
2. **429 成本放大器**：限流在重试循环里高频爆发，每次触发一次付费诊断既烧钱又加剧限流。本地规则表天然无此问题。
3. **隐私合规**：错误消息常带请求片段；本地匹配不外发，天然规避。

## 5. 分层设计

- **Tier 0（默认，已落地）**：本地规则库。按 `(status_code, code keyword, message pattern)` 匹配，规则「最specific 优先、首个命中即返回」，输出原因列表 + 修复片段 + 快捷命令 + AI expert 升级提示。已覆盖 Top 高频错误：鉴权、workspace/region、限流、模型不存在、越权、内容审核、参数非法、网络、服务端内部错误。
- **Tier 1（未做，opt-in 兜底）**：规则未命中且用户显式开启时才调轻量诊断模型；对 429 类高频错误必须禁用或强去重。当前统一引导 `dashscope expert "<error>"`。

## 6. 双语

规则表中英双语。`DASHSCOPE_DIAGNOSIS_LANG=en` 选英文，默认中文。`render(diagnosis, lang=None)` 输出终端块，结尾附 AI expert 升级提示。

## 7. 公开 API

```python
from dashscope.v2.devkit import diagnose, render, install_excepthook, Diagnosis

diag = diagnose(status_code=401, code="InvalidApiKey", message="no api key provided")
print(render(diag))            # 终端块（中英由 DASHSCOPE_DIAGNOSIS_LANG 决定）

install_excepthook()           # 未捕获异常 traceback 后追加诊断块（仅 stderr，幂等）

# 挂载点：
err.diagnosis                  # DashScopeError 及全部子类
resp.diagnosis                 # raise_on_error=False 的错误信封（成功时为 None）
```

## 8. 测试

`tests/v2/test_devkit_diagnosis.py`：按 status/code/message 三种维度匹配、未命中返回 None、env 切英文、render 含标题与 expert 提示与编号修复、`DashScopeError.diagnosis`、`SdkError` 继承、错误信封 `.diagnosis`（业务错误命中 / 成功为 None）、`install_excepthook` 渲染到 stderr。
