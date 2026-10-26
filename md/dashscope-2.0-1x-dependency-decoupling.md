# DashScope 2.0 ↔ 1.x 依赖清单与解耦方案

> 目标仓库:`moe-dashscope-sdk-python`
> 作用:作为"2.0 上提为顶层、1.x 归入 legacy"这次大版本重构的**强制前置(Step 0)**。
> 背景:2.0(`dashscope/v2/`)并非自包含——它对 1.x 有 **8 处反向依赖**。任何 `git mv`
> 之前必须先把这 8 处解掉或归类,否则搬动 1.x 会直接打断 2.0 的运行时导入。

## 1. 解耦原则

1. **共享基础设施**:若代码量不多,**直接在 2.0 内重新实现**,让 2.0 自己拥有,消除对 1.x 的依赖。
2. **功能级委托**:做**反向搬迁**——把实现从 1.x 移到 2.0,原 1.x 路径改为**反向引用 2.0**(留薄垫片)。
3. **例外(单一事实来源)**:凡"进程级唯一 / 发行版级唯一"的模块,**禁止复制到 2.0**,否则破坏其唯一性不变量。此类保持为 1.x/2.0 **共用的中立叶子**,两边都 import 它(下详)。

> 原则 3 是对原则 1 的必要修正:`version`、`_sdk_session` 表面上也是"共享基础设施且代码极少",
> 但它们的价值恰恰在于**全局只有一份**。复制进 2.0 = 两个版本号 / 两个 session id = bug。

## 2. 依赖清单总表

| # | 2.0 引用点 | 依赖的 1.x 模块 | 规模 | 类别 | 原则 | 动作 | 风险 |
|---|---|---|---|---|---|---|---|
| D1 | `transport/_ident.py:53`、`transport/http.py:125` | `dashscope.version.__version__` | 4 行 | 单一来源 | 3 | 保持共用叶子;重构后两边仍指向同一处 | 低 |
| D2 | `transport/_ident.py:24` | `dashscope.common._sdk_session.sdk_session_id` | 43 行 | 单一来源 | 3 | 保持共用叶子(模块注释已声明此为唯一来源) | 低 |
| D3 | `v2/__init__.py:19` | `dashscope.common.api_key.save_api_key` | 函数 ~12 行 | 小型基础设施 | 1 | 在 2.0 内重写 `save_api_key`(仅文件写入) | 低 |
| D4 | `resources/chat/completions.py:17` | `dashscope.utils.param_utils.ParamUtil` | 31 行 | 小型基础设施 | 1 | 在 2.0 内重写该纯静态方法 | 低 |
| D5 | `v2/__init__.py:20` | `dashscope.tokenizers.{get_tokenizer,list_tokenizers}` | 434 行(tiktoken 封装) | 中型基础设施 | 1 / 2 | 优先随 2.0 拥有(见 §3.3) | 中 |
| D6 | `resources/_oss.py:26`(5 处调用) | `dashscope.utils.oss_utils.check_and_upload_local` | 312 行模块,与 1.x 深缠 | 中型基础设施 | 1 | 把 OSS 签名+上传逻辑移植为 2.0 原生(httpx + v2 errors) | 中 |
| D7 | `client.py:147/361`、`resources/_agentstudio.py:49/50/53` | `dashscope.agentstudio`(Client/AsyncClient/exceptions/pagination) | 8533 行,完整子 SDK | 功能级委托 | 2 | 整包移入 2.0,`dashscope/agentstudio` 改薄垫片反向引用 | 中高 |
| D8 | `resources/reinforcement_learning.py:74-76` | `dashscope.finetune.{agentic_rl, reinforcement, reinforcement.common}` | `agentic_rl.py` 685 行 + `reinforcement/` 11270 行(不含 examples 2349 行) | 功能级委托 | 2 | 仅移 `reinforcement/` 子包 + `agentic_rl.py` 入 2.0,原路径反向引用;`finetune/finetunes.py`、`deployments.py` 留在 1.x | 高 |

## 3. 分类详解

### 3.1 单一事实来源(原则 3,禁止复制)—— D1、D2

- **D1 `dashscope/version.py`**:仅 `__version__ = "2.0.0"`,是**整个发行版**的版本号唯一来源。2.0 的
  `_ident.py` / `http.py` 用它拼 `User-Agent`。绝不能"在 2.0 内再写一份版本号"——两处会漂移。
  重构后无论 1.x/2.0 目录怎么挪,`version.py` 应留在**中立的发行版根**(或 `dashscope/_shared/`),两边共同 import。
- **D2 `dashscope/common/_sdk_session.py`**:进程级唯一 `x-dashscope-sdk-session-id`。模块自身注释已写明:
  *"一个进程可能同时经 1.x 与 2.0 发请求,2.0 还会把 rl/agentstudio 委托进 1.x;若各自 mint id,一个逻辑会话会裂成多个 id,服务端日志分组失效。"*
  这正是我们要解耦的委托场景本身。因此它**必须保持为两边共用的叶子**,复制到 2.0 会破坏"一进程一 id"不变量。

**动作**:D1/D2 不参与"移入 2.0",而是确立为**中立共享层**(建议 `dashscope/_shared/`,只依赖 stdlib,
不 import `dashscope/__init__`,避免环)。这一步同时是重构后避免循环导入的锚点。

### 3.2 小型共享基础设施(原则 1,直接在 2.0 重写)—— D3、D4

- **D3 `save_api_key`**:v2 只用"写 key 文件"这一支(不碰读 1.x 全局 `dashscope.api_key` 的 `get_default_api_key`)。
  在 2.0 内实现约 12 行(`os.makedirs` + 写文件),常量 `DEFAULT_DASHSCOPE_CACHE_PATH` /
  `DEFAULT_DASHSCOPE_API_KEY_FILE_PATH` 一并带入 2.0 的 config 常量表。随后 `v2/__init__.py` 改为本地导出。
- **D4 `ParamUtil.should_modify_incremental_output`**:31 行纯函数(按模型名判断是否关闭增量输出),无任何依赖。
  直接在 2.0 内落一个等价函数(如 `dashscope/v2/resources/chat/_params.py`),`completions.py` 改本地导入。

**动作**:两者代码量都很小、无外部依赖,是原则 1 的教科书案例,可各自独立成 PR,零回归面。

### 3.3 中型共享基础设施(原则 1,但非平凡)—— D5、D6

- **D5 `tokenizers`(434 行)**:是对 tiktoken 的自包含封装,2.0 仅通过 `v2/__init__.py` 再导出
  `get_tokenizer` / `list_tokenizers` 作便利入口。它**不与 1.x 业务耦合**,更像可整体归属 2.0 的叶子。
  建议:重构时把 `dashscope/tokenizers/` 归 2.0 拥有,原路径留再导出垫片(本质是原则 2 的轻量版);
  若短期不动,则至少确认它是纯叶子,可安全共享。**注意**:不要"复制"——tokenizer 加载/缓存应只有一份。
- **D6 `oss_utils.check_and_upload_local`(312 行模块)**:是本次**最重的基础设施移植**。该模块 import 了
  `DashScopeAPIResponse`、`client.base_api.GetMixin`、`common.error/logging/utils`、`requests`——与 1.x 客户端机制深缠。
  v2 在 `_oss.py` 有 5 处调用(多模态本地文件上传取 OSS 凭证并上传)。
  建议:只把**实际用到的上传路径**(`upload_file` + `check_and_upload_local` 的签名/上传逻辑,约百余行)
  移植为 **2.0 原生实现**(走 v2 的 httpx transport 取凭证、抛 `dashscope.v2.errors`),不引入 1.x 的
  `GetMixin` / `DashScopeAPIResponse`。这是"代码量不算少"的边界项,单列一个 PR,配 OSS 上传回归测试。

### 3.4 功能级委托(原则 2,反向搬迁 + 1.x 反向引用)—— D7、D8

共同前提:`agentstudio` 与 `finetune.reinforcement` **都未在 1.x 包根 `dashscope/__init__.py` 再导出**
(经核对),只通过深路径 `dashscope.agentstudio` / `dashscope.finetune.reinforcement` 访问。
因此反向引用垫片的兼容面**只是这两条深路径**,不涉及 1.x 顶层 65 个公开名,风险可控。

- **D7 `agentstudio`(8533 行)**:是一个完整子 SDK,且已是"2.0 风格"(自带 `transport` / `RequestOptions` /
  `APIResponse`)。2.0 侧 `resources/_agentstudio.py` 只是**透明错误对齐代理**——挂载 1.x 的
  `agentstudio.Client`,把其异常族翻译成 `dashscope.v2.errors`(设计注释明言"不改动任何 1.x 代码")。
  **动作(原则 2)**:把整个 `dashscope/agentstudio/` 迁到 2.0 归属(重构后即 2.0 顶层下的 `agentstudio/`);
  原 `dashscope/agentstudio/__init__.py` 改为**薄反向引用**(`from <2.0>.agentstudio import *` 或 `sys.modules` 别名),
  保证 `import dashscope.agentstudio` 与其深导入仍解析到同一份实现。2.0 的错误对齐代理可就地保留。
  **注意(实测补充)**:agentstudio 并非完全自包含——`client.py`/`exceptions.py`/`streaming.py`/`transport.py`
  还 import 1.x 的 `common.api_key.get_default_api_key`、`common.env.validate_region` /
  `validate_workspace_id`、`common.error.DashScopeException`(其异常基类)、`common.logging.logger`、
  `common.utils.get_sdk_headers`。迁入 2.0 后这些成为对 1.x `common/` 的新增依赖;搬迁前需把这几支
  归类为中立共享层(同 D1/D2)或一并处理,否则 §4 的递减 gate 会在 agentstudio 上卡住。
- **D8 `finetune.reinforcement` + `agentic_rl`(合计约 11955 行运行时)**:本次**最大搬迁**。
  2.0 的 `resources/reinforcement_learning.py:151` 经模块内 `_new_agentic_rl()` 构造
  `_agentic_rl.AgenticRL(api_key=...)`——只从 v2 client 取 `config.api_key` 字符串传入,
  不再持有 client 本体,耦合面比撰写本文时更小,业务归属仍在 2.0。
  **动作(原则 2,注意是"部分搬迁")**:只把 `dashscope/finetune/reinforcement/`(子包)与
  `dashscope/finetune/agentic_rl.py` 迁入 2.0 归属;**`finetune/finetunes.py`、`finetune/deployments.py`
  属于 1.x 根公开 API(`FineTunes` / `Deployments`),留在 1.x 不动**。原
  `dashscope.finetune.reinforcement` / `dashscope.finetune.agentic_rl` 改为反向引用垫片。
  `reinforcement/examples/`(2349 行)是样例,按仓库惯例处理(随包或移到 `examples/`),不计入运行时耦合。
  **注意(实测补充)**:`agentic_rl.py` 自身还拖带 1.x 依赖——`:10` `client.base_api.CreateMixin`、
  `:11` `finetune.customize_types`、`:17` `finetune.finetunes.FineTunes`(正是上文计划留在 1.x 的模块)。
  迁入 2.0 会因此新增 v2→1.x 依赖,同样会卡住 §4 的递减 gate;搬迁前需先处理这三支
  (CreateMixin/customize_types 归类或重写,对 `FineTunes` 的依赖解耦)。

### 3.5 共享产品面:`acli`(顶层中立,1.x/2.0 都支持)—— 不属于任何一侧

`dashscope/acli/`(约 **36614 行**)是内嵌的交互式 AI 助手(agentic CLI),经 `python -m dashscope.acli`
启动(`acli/__main__.py` → `dashscope.acli.cli:main`),并在 `setup.py` 拥有独立可选依赖组
(`acli` / `acli-anthropic` / `acli-openai` / `acli-voice` / `acli-camera` / `acli-all`)。

它的定位与 D1–D8 不同,**不是 2.0 对 1.x 的反向依赖**,而是一个**同时服务于 1.x 与 2.0 的顶层产品面**:

- **依赖方向(实测)**:`acli` → 1.x,仅 2 处真实导入——`acli/ui/tts.py:287` 用 `dashscope.audio.tts_v2`、
  `acli/ui/voice.py:61` 用 `dashscope.audio.asr.{Recognition,RecognitionCallback}`。
  `acli` **不 import `dashscope.v2`**;1.x 包根与 2.0 也**都不 import `acli`**(仅 `v2/transport/_ident.py:22`
  有一句注释提及)。即 `acli` 是 SDK 的**消费方 / 独立入口**,不是任何一侧 API 层的组成部分。

**动作(布局归属)**:重构后 `acli` **保持在顶层 `dashscope/acli/` 不动**,与 `_shared`(§3.1)同属
**中立层**——既不进 1.x legacy 桶,也不并入 2.0 实现桶。理由:它对两栈都成立,且是被安装的独立产品入口。

**关键约束(勿在解耦中改写 acli 内部)**:据 `dashscope/common/_sdk_session.py` 的模块注释,`acli` 是
**agenticCLI 仓库的单向同步副本(one-way sync copy)**,连 session id 都刻意自持一份——因为任何本地改动
都会被下次同步覆盖。因此:

1. **不要**把 `acli` 内部作为本次解耦的重构对象(改了也会被 sync 冲掉)。
2. 重构搬动 1.x 时,**必须保证 `acli` 依赖的两条 1.x 音频路径继续可解析**:`dashscope.audio.tts_v2`、
   `dashscope.audio.asr`。两种做法二选一——(a) 由 1.x→legacy 的兼容垫片让这两条深路径照旧解析(推荐,
   零改 acli);(b) 若确需改 `acli` 的 import,则**同步更新上游 sync 映射**,否则下次同步回退。
3. `acli` 未来若要改用 2.0 client(它"支持 2.0"),属于 **agenticCLI 上游**的演进,不在本仓库解耦范围内。

## 4. 建议执行顺序(小→大,每项独立可合并)

1. **D1/D2**:确立 `dashscope/_shared/` 中立层(version + sdk_session),两边改指此处。*(低风险,且是防环锚点)*
2. **D3**、**D4**:各自在 2.0 内重写。*(低风险,两个独立小 PR)*
3. **D5**:tokenizers 归属 2.0 + 垫片。*(中)*
4. **D6**:OSS 上传逻辑 2.0 原生化。*(中,需回归测试)*
5. **D7**:agentstudio 反向搬迁 + 垫片。*(中高)*
6. **D8**:reinforcement/agentic_rl 部分反向搬迁 + 垫片。*(高,最后做)*

原则:每一步完成后,`grep -rn "dashscope\.<非v2>" dashscope/v2/` 的命中应**单调递减**,直到只剩 D1/D2
指向 `_shared`。D3–D8 全部落地后,2.0 对 1.x **业务模块零依赖**,此时才具备"1.x 移入 legacy、2.0 上提顶层"
的物理搬迁条件(见 `dashscope-v2-refactor-tasks.md` / 顶层重构评估)。

## 5. 回归验证 gate(每步必过)

- `python -c "import dashscope; import dashscope.v2"` 无循环导入、无告警。
- `pytest tests/v2` 全绿;涉及 D6/D7/D8 时补对应回归(OSS 上传、agentstudio 挂载与错误翻译、RL 构造)。
- 反向引用垫片生效:老深路径 `import dashscope.agentstudio`、`from dashscope.finetune.reinforcement import ...`
  仍解析到迁移后的**同一份对象**(`is` 一致),不产生双实现。
- `x-dashscope-sdk-session-id` 在"同进程同时用 1.x 与 2.0"下**仍为同一 id**(守护 D2 不变量)。
- `python -m dashscope.acli --help` 可启动;`acli` 依赖的 `dashscope.audio.tts_v2`、`dashscope.audio.asr`
  两条深路径在重构后**仍可解析**(守护 §3.5 中立层约束);`acli` 保持在顶层 `dashscope/acli/` 未被搬动。
- 全仓 secret 扫描通过,版本号仅一处(D1)。
