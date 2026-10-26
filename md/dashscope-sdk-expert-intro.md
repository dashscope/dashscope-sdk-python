# DashScope SDK Expert 简介

DashScope SDK Expert 是一个面向 **DashScope Python SDK / Java SDK** 与 **阿里云百炼（Model Studio）** 的智能开发助手。它可以提供可运行的代码示例、解释 API 参数与错误码、阅读你的代码定位问题，并直接执行命令验证 SDK 行为——目标是让你**不翻文档、不手写样板代码**，只描述需求就能完成 SDK 开发与问题排查。

## 快速开始

1. **安装 / 升级 SDK**：DashScope SDK Expert 内置于 DashScope Python SDK，先安装或升级到最新版：

   ```bash
   pip install -U dashscope
   ```

2. **启动 CLI**：在终端执行 `dashscope` 命令，进入交互式会话。
3. **查看帮助**：在会话中输入 `/help`，即可查看当前可用的全部命令与技能（skill）列表。
4. **按需提问**：直接用自然语言提问即可，例如：
   - “用 qwen-plus 写一个流式对话示例”
   - “`InvalidApiKey` 这个错误怎么解决？”
   - “帮我看下 `my_app.py` 为什么调用失败”

> 提示：Expert 会交叉校验本机已安装的 dashscope 版本——版本过旧时先执行 `pip install -U dashscope` 升级，即可获得与最新 SDK 一致的答案。除 `/help` 外，还可通过 `/skill <名称>` 单行触发专项技能（如 `/skill text-generation`、`/skill error-code <错误码>`），具体可用列表以 `/help` 输出为准。

## 覆盖能力一览

| 领域 | 说明 |
|---|---|
| 文本生成 | Generation、OpenAI 兼容接口 |
| 多模态 | MultiModalConversation、图像合成、视频合成 |
| 语音 | 语音合成（TTS）、录音文件转写（ASR） |
| 检索 | 文本/批量/多模态向量、Rerank、RAG |
| 微调与部署 | SFT/CPT/DPO、独立推理端点、强化学习 |
| Agent | 托管 Agent API、应用调用、Assistants、插件与 MCP |
| CLI | dashscope 命令行全部命令组 |

---

## 最佳实践一：只提需求，不读文档，直接产出可运行代码

**痛点**：DashScope SDK 接口多、版本迭代快，开发者经常要翻文档确认参数名、输入格式和模型列表。

**做法**：把需求直接用自然语言描述给 DashScope SDK Expert，它会：

1. 匹配内置的 SDK 知识库（模型列表、参数、输入输出结构、错误码）；
2. **交叉校验本机已安装的 SDK 版本**，确保给出的签名与用法与你环境一致（版本不一致时会主动提醒）；
3. 直接生成可运行代码，并用 `run_command` 实际执行展示输出。

**示例对话**：

> 用户：写一个 Python 示例，用 qwen-plus 流式输出，统计 token 消耗。

助手直接给出完整可运行脚本（messages 格式 + `stream=True` + usage 解析），无需你查任何文档。

**小技巧**：

- 指定语言或框架：“用 Java SDK 实现”“封装成 FastAPI 接口”；
- 指定版本：“我们用的是 dashscope 1.20.x，这个参数支持吗？”——助手会核对安装版本后回答；
- 追问细节：“这个参数的取值范围是什么？”“还有哪些可选模型？”

---

## 最佳实践二：排查问题时，让它自动定位并修复 Bug

**痛点**：调用报错时，错误码含义不明、报错堆栈指向 SDK 内部，难以判断是代码问题还是参数问题。

**做法**：按以下任一路径交给 DashScope SDK Expert：

1. **贴错误码/报错信息**：如 `Throttling.RateQuota`、`InvalidParameter`，助手会解释含义并给出可执行的修复建议（如调整并发、修改参数格式、申请配额）。
2. **给出出问题的代码文件路径**：助手会自动读取文件，诊断 SDK 调用问题（常见根因：混用旧版 prompt 格式、messages 结构错误、api_key/workspace 未配置、异步任务未正确轮询等）。
3. **自动修复闭环**：定位根因后，助手会直接修改文件，并运行测试/脚本验证修复效果，形成 **读取 → 诊断 → 修复 → 验证** 的完整流程。

**示例对话**：

> 用户：运行 `gen.py` 报 `InvalidParameter: url error`，帮我修一下。

助手读取 `gen.py` → 发现传入了本地路径而非可访问的 URL → 修改为正确的输入方式 → 重新运行确认通过。

**小技巧**：

- 报错时附上 `request_id`，便于与服务端日志对应；
- 复杂问题可先让助手“解释这段代码逻辑”，确认理解一致后再修复；
- 遇到 SDK 本身的缺陷或未满足的需求，助手会引导你到官方仓库提 Issue：
  - Python SDK：https://github.com/dashscope/dashscope-sdk-python/issues
  - Java SDK：https://github.com/dashscope/dashscope-sdk-java/issues

---

## 小结

| 场景 | 一句话用法 |
|---|---|
| 开始使用 | 执行 `dashscope`，输入 `/help` 查看全部能力 |
| 开发 | 直接描述需求 → 拿到校验过版本的可运行代码 |
| 排障 | 贴报错或给文件路径 → 自动诊断、修复并验证 |
