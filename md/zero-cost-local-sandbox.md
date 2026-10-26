# 零成本本地沙箱（Zero-Cost Local Sandbox）— dashscope 2.0 设计

> 状态：已落地（v2）。对应实现：`dashscope/v2/devkit/mock.py`，挂载点 `dashscope/v2/transport/http.py`。
> 来源：1.x 提案 `zero-cost-local-sandbox`（方式 A），针对 2.0 资源式架构重写，非简单迁移。

## 1. 需求与目标

开发者用大模型 Agent 辅助写代码时，Agent 会反复跑单测、调试循环逻辑。每次都打云端 API 带来三个问题：账单激增、触发限流、受外网时延拖累。目标：调试阶段一行开启 mock，零云端 token 消耗，业务代码零改动。

```python
import dashscope.v2 as v2

v2.enable_dev_mock(mode="static")           # 或环境变量 DASHSCOPE_DEV_MOCK=static
client = v2.Client(api_key="sk-xxx")
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "写一首关于阿里云百炼的诗"}],
)
print(resp.output_text)                      # "[MOCK] static response for model=qwen-plus"
```

## 2. 拦截点：2.0 与 1.x 的关键差异

1.x 的唯一 choke point 是 `BaseApi.call`，mock 在此短路并直接返回**已解析的** `DashScopeAPIResponse`。

2.0 没有 `BaseApi.call`：每个资源（`chat.completions` / `multimodal.chat` / `embeddings` / …）各自构造请求，统一经 **transport 层** 的四个方法出口：

```
SyncTransport.request / open_stream
AsyncTransport.request / open_stream
```

transport 返回的是**原始 `httpx.Response`**（wire 字节），协议解析（native / openai / anthropic 适配器 + 类型化模型）在其后的资源层完成。

因此 v2 mock 的正确落点是 transport，且**合成的是 wire 响应而非成品对象**——真实解析管线原样跑在 mock 数据上。这比 1.x 更忠实：dev loop 验证的是与生产完全一致的解码路径（协议适配器、SSE 分帧、typed model 构造、流式聚合）。

**生产零成本挂载**：transport 每次发送前调 `_dev_mock_requested()`——一次 `DASHSCOPE_DEV_MOCK` 环境变量读取 + 一次 `sys.modules.get("dashscope.v2.devkit")` 探测；只有旗标为真时才 `from dashscope.v2 import devkit`。未开启时不导入 devkit、成功路径无额外依赖。真实发送逻辑抽成 `_*_impl`，mock 分支用一个 `real` 闭包（`functools.partial`）持有它，供 record 模式透传。

## 3. 三模式语义

- **static**：零网络。按**请求 path 推断协议**（与适配器同源信号），合成该协议能正确解析的 wire body：
  - native：`/services/aigc/...generation` → `output.choices[].message` 形态；
  - openai：`/chat/completions` → `choices[].message` + `usage`；
  - anthropic：`/messages` → `content[]` blocks + `stop_reason`。
  文本打上抹不掉的 `[MOCK]` 前缀，`request_id="mock-*"`，usage 给确定性估算。
- **record**：透传真实调用，把 (请求指纹 → 原始 wire 字节) 追加写入 JSONL fixture；**headers/Authorization 一律不落盘**（只存 status_code / content_type / body 文本）。流式用 tee 字节流边转发边捕获，保持直播不缓冲。
- **replay**：按请求指纹（method + path + body + params 的 sha256，**排除 headers**）查 fixture 回放，重建 `httpx.Response`；**未命中 = 硬报错** `DevMockMiss`（继承 `DashScopeError`），绝不静默走网络。record/replay 与协议无关，覆盖**所有**资源。

## 4. 覆盖边界：宁可loud，不可静默错配

static 只建模 chat / multimodal 三个协议的 wire 形态。对非 chat 端点（embeddings / images / files / …），static **抛 `DevMockMiss`** 明确提示改用 record/replay，而不是返回一个会被该端点 typed parser 静默误读的 body。这正是 1.x 提案「别让 Agent 在静默错配里空转」原则的 v2 落地。

## 5. 流式

static 按 `stream=True` 产出增量 SSE 帧序列（native 多帧末片 `finish_reason=stop` 带 usage；openai 追加 `data: [DONE]`；anthropic 走 `message_start → content_block_delta… → message_delta → message_stop`），使 `for chunk in stream` / `async for` 的消费与聚合路径与真实 SSE 完全一致——不接受「只 mock 非流式」。

## 6. 明确不做

- websocket 通路（realtime 语音 / 多模态对话）的 mock；
- 方式 B（环境变量桥接本地 Ollama）的 native↔OpenAI 翻译 shim——新代码走 OpenAI client 指本地即可，存量 native 翻译单独立项；
- SDK 侧影子流量。

## 7. 公开 API

```python
dashscope.v2.enable_dev_mock(mode="static" | "replay" | "record", fixture_path=None)
dashscope.v2.disable_dev_mock()
dashscope.v2.dev_mock_active() -> bool
# 等价物：dashscope.v2.devkit.{enable,disable,active,intercept,aintercept}
```

环境变量：`DASHSCOPE_DEV_MOCK=static|replay|record`、`DASHSCOPE_DEV_MOCK_FIXTURE=<path>`（replay/record 默认 `./dashscope_mock.jsonl`）。代码调用优先于环境变量。

## 8. 测试

`tests/v2/test_devkit_mock.py`：static 非流式/流式、三协议、无网络证明（指向 dead host）、多模态 path、非 chat 端点 loud miss、env 激活、非法 mode 报错、disable 恢复透传、record 落盘不含凭证、record→replay 往返（同步+异步、流式+非流式）、replay 未命中抛 `DevMockMiss`。
