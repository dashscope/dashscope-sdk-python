# DashScope 2.0 根目录迁移与 `legacy_2x` 归档方案

> 状态：设计稿，尚未实施。
>
> 前置方案：`md/dashscope-2.0-agentstudio-agenticrl-migration.md`。
> 本文只描述后续目录迁移；不修改 `dashscope-error-registry`，不重复实施
> AgentStudio / AgenticRL 错误体系统一，也不调整已经完成的 `examples/` 迁移。

## 1. 目标

1. `dashscope/v2/` 不再作为实现目录，2.0 实现提升到 `dashscope/` 根目录；
   根目录公开 API（`dashscope.Client`、`dashscope.AsyncClient` 等）与实现位置一致。
2. AgentStudio、AgenticRL / reinforcement 遵循前置方案已经确认的结论：
   - 单一实现；
   - 原生使用统一错误体系；
   - 保留既有 1.x API 行为和需要兼容的导入路径；
   - `dashscope.reinforcement` 是 reinforcement 的公开入口，
     `dashscope.finetune.reinforcement` 继续可用。
3. `dashscope/acli/` 保留在根目录，不归档。
4. 其余属于 2.x 阶段旧公开面的目录和模块迁入 `dashscope/legacy_2x/`，
   明确标记为后续下线，不在本次迁移中直接删除。
5. `tests/v2/` 提升为 `tests/` 根目录；其余测试套件迁入
   `tests/legacy_1x/`。测试发现、CI、coverage、pre-commit、测试内部导入和
   路径计算同步修正。
6. `examples/` 已完成，本次不再搬迁或改写。

## 2. 范围界定与不可混淆的目录

“除 acli 外其他 2.x 目录”按本文 §4 的显式清单执行，不能解释为“移动除
acli 外的所有目录”。以下目录有特殊职责，不纳入 `legacy_2x`：

| 目录 | 处理 |
|---|---|
| `dashscope/acli/` | 当前 CLI，保留根目录 |
| `dashscope/_shared/` | 跨层共享基础设施，保留根目录 |
| `dashscope/legacy_1x/` | 既有 1.x 业务实现，不是本次新建的归档目录 |
| `dashscope/agentstudio/` | 提升为 2.0 实现目录，遵循前置方案 |
| `dashscope/reinforcement/` | 提升为 reinforcement 实现目录和公开入口 |
| `dashscope/finetune/` | 保留 1.x finetune API 与 AgenticRL 兼容面，按 §3.3 合并 |
| `dashscope/tokenizers/` | 合并 2.0 tokenizer 实现与既有兼容导出，不整体归档 |
| `dashscope/v2/` | 内容迁出后删除该实现命名空间 |

`legacy_2x` 是 2.x 阶段旧 API / 兼容面的过渡归档位置；`legacy_1x` 是其下层
历史实现的现有位置。本次不把两者合并，也不借目录迁移制造第二份实现。

`dashscope.v2` 当前定位为内部实现命名空间，不作为需要长期保留的公开兼容面。
迁移时所有内部 import、patch target、动态模块字符串和测试统一改为根目录新路径；
最终不保留 `dashscope/v2/` 实现副本。

## 3. `dashscope/` 迁移设计

### 3.1 目标结构（关键部分）

```text
dashscope/
├── __init__.py                 # 合并导出，不用 v2/__init__.py 覆盖
├── client.py                   # 当前 Client / AsyncClient 实现
├── errors.py                   # 当前统一错误体系
├── _error_registry.py          # vendored catalog，来源和 pin 不变
├── _config.py
├── _logging.py
├── _quickstart.py
├── agentic_rl.py               # 当前 AgenticRL 实现，见 §3.3
├── agentstudio/                # 当前 AgentStudio 实现
├── reinforcement/              # 当前 reinforcement 实现及公开入口
├── finetune/
│   ├── _submit.py              # 当前 submit transport
│   ├── agentic_rl.py           # 既有 1.x AgenticRL 行为和返回类型
│   └── reinforcement/          # 旧 reinforcement 导入路径的 alias
├── protocols/
├── resources/
├── transport/
├── types/
├── tokenizers/                 # 合并后的实现与资源
├── acli/
├── _shared/
├── legacy_1x/
└── legacy_2x/                  # 过渡归档，后续另行下线
```

### 3.2 常规 v2 文件映射

| 当前位置 | 目标位置 | 说明 |
|---|---|---|
| `dashscope/v2/client.py` | `dashscope/client.py` | 现有 `dashscope/client/` 先按 §4 归档，消除文件/包冲突 |
| `dashscope/v2/errors.py` | `dashscope/errors.py` | catalog 映射和错误类不变 |
| `dashscope/v2/_error_registry.py` | `dashscope/_error_registry.py` | 不改 registry 上游或 vendored 内容 |
| `dashscope/v2/_config.py` | `dashscope/_config.py` | 仅调整内部导入 |
| `dashscope/v2/_logging.py` | `dashscope/_logging.py` | 仅调整内部导入 |
| `dashscope/v2/_quickstart.py` | `dashscope/_quickstart.py` | 仅调整内部导入 |
| `dashscope/v2/protocols/` | `dashscope/protocols/` | 整目录提升 |
| `dashscope/v2/resources/` | `dashscope/resources/` | 整目录提升 |
| `dashscope/v2/transport/` | `dashscope/transport/` | 整目录提升 |
| `dashscope/v2/types/` | `dashscope/types/` | 整目录提升 |
| `dashscope/v2/tokenizers/` | `dashscope/tokenizers/` | 合并，不覆盖既有兼容导出 |
| `dashscope/v2/__init__.py` | 合并进 `dashscope/__init__.py` | 根导出仍以现有 `dashscope/__init__.py` 为基线 |
| `dashscope/v2/__init__.pyi`（若后续存在） | 合并进 `dashscope/__init__.pyi` | 类型导出与运行时一致 |

仅存在 `__pycache__` 的 `devkit/` 不属于 tracked source，不纳入迁移清单；
清理生成物不作为源码迁移提交。

### 3.3 AgentStudio / AgenticRL / reinforcement 特殊映射

这些路径不能直接按相对目录覆盖，否则会破坏前置方案已经确认的兼容契约。

#### AgentStudio

- `dashscope/v2/agentstudio/*` → `dashscope/agentstudio/*`，由 shim 位置变为唯一实现。
- 现有 `dashscope.agentstudio` 导入路径不变，公开类、helper、分页和 SSE 行为不变。
- 异常继续原生使用根目录统一错误体系；旧异常别名继续保留。
- 原 `tests/v2/test_agentstudio_shim.py` 中“1.x 路径与 v2 路径是同一模块”的断言
  不能机械替换为自身比较，应改为验证公开导出、deep import、错误别名和可选依赖行为。
- `dashscope/v2/agentstudio/` 随 v2 目录退出，不保留第二份实现。

#### AgenticRL

- `dashscope/v2/finetune/agentic_rl.py` → `dashscope/agentic_rl.py`，避免覆盖
  现有 1.x `dashscope/finetune/agentic_rl.py`。
- `dashscope/v2/finetune/_submit.py` → `dashscope/finetune/_submit.py`，submit
  wire format、header 顺序和返回 envelope 不变。
- 根 `Client(...).rl` 及其 mount 改为从 `dashscope.agentic_rl` 和
  `dashscope.reinforcement` 导入。
- `dashscope.finetune.agentic_rl.AgenticRL` 继续保留 1.x 子类、五个任务管理动词、
  `FineTunes` 委托和 `FineTune` 返回类型；当前实现继续返回 `SubmitResult`。
- 不因目录迁移把 1.x 动词合并进当前实现，也不让两个路径返回类型悄悄一致。

#### reinforcement

- `dashscope/v2/finetune/reinforcement/*` → `dashscope/reinforcement/*`，
  由公开 shim 位置变为唯一实现。
- `dashscope.finetune.reinforcement` 保留为 alias，deep import 仍解析到同一模块对象。
- `dashscope.reinforcement` 保持公开 FC / uvicorn / `python -m` dotted path：
  `dashscope.reinforcement.component.server.server`。
- 懒加载约束不变：`component.server.*` 和
  `component.observability.genai.*` 不得因包迁移变成 eager import，也不得把
  fastapi / opentelemetry 提升为根包硬依赖。
- `dashscope.finetune.agentic_rl` 的 re-export 和旧 CLI 继续可用。
- 数字 `error_code` 兼容、catalog code / solution / allow_retry、root cause 行为不变。

### 3.4 根导出与冲突处理

1. `dashscope/__init__.py` 和 `.pyi` 采用合并方式，当前 2.0 导出成为规范来源。
2. 过渡期内仍需保留的根级旧导出可重定向到 `legacy_2x`，其最终删除与
   `legacy_2x` 下线一起评审；不能在目录迁移提交中静默扩大删除范围。
3. `dashscope/client.py` 与旧 `dashscope/client/` 不能共存为同一导入名。旧包迁入
   `legacy_2x` 后，旧的 `dashscope.client.*` 深层导入属于兼容性变化，必须在迁移
   指南和破坏性变更清单中单独列出。
4. tokenizer 的 `qwen.tiktoken` 随实现移动，运行时 loader 与 `MANIFEST.in`
   必须使用同一路径；不能手改生成的 `egg-info` 来伪装迁移完成。
5. 所有 import 修改同时覆盖字符串形式：monkeypatch、`importlib`、runpy、uvicorn、
   entry point、Sphinx module mapping 和规格文件，不能只改静态 `from/import`。

## 4. `dashscope/legacy_2x/` 归档与下线标记

### 4.1 本次归档清单

以下现有顶层旧公开面包目录迁入 `dashscope/legacy_2x/`，保持目录名：

```text
aigc/  api_entities/  app/  assistants/  audio/  cli/  client/  common/
embeddings/  io/  multimodal/  nlp/  protocol/  rerank/  threads/  utils/
```

以下顶层旧模块一并迁入：

```text
files.py  model.py  models.py
```

`agentstudio/`、`reinforcement/`、`finetune/`、`tokenizers/` 不在该清单中，
按 §3 的特殊规则处理。`acli/`、`_shared/`、`legacy_1x/` 也不归档。

### 4.2 归档后的行为

- `dashscope/legacy_2x/__init__.py` 和各入口 docstring 标注“过渡兼容，后续下线”。
- 在包入口使用带正确 `stacklevel` 的 `DeprecationWarning`，并通过测试固定其含义；
  不改为静默删除或立即抛出导入异常。
- 归档代码继续复用既有实现，不能复制出另一套业务逻辑。
- `setup.py` 的 console entry point 从 `dashscope.cli:main` 调整到归档后的实际
  位置，直至该旧 CLI 正式下线；acli 入口不受影响。
- `setuptools.find_packages()` 会继续打包 `dashscope.legacy_2x`，这是过渡期预期行为；
  正式下线时再单独调整打包与根导出。
- 本次不确定具体删除版本。下线前另行完成使用面评估、迁移指南、发布说明和删除 PR。

## 5. `tests/` 迁移设计

### 5.1 目录映射

| 当前位置 | 目标位置 |
|---|---|
| `tests/v2/test_*.py` | `tests/test_*.py` |
| `tests/v2/protocols/` | `tests/protocols/` |
| `tests/v2/smoke/` | `tests/smoke/` |
| `tests/unit/` | `tests/legacy_1x/unit/` |
| `tests/integration/` | `tests/legacy_1x/integration/` |
| `tests/legacy/` | `tests/legacy_1x/legacy/` |
| `tests/data/` | 保持在 `tests/data/` |

`tests/data/` 是共享 fixtures，不是测试套件，不归档，避免无意义地修改全部数据
路径。`__pycache__`、coverage 和 pytest cache 是生成物，不纳入迁移。

`tests/legacy_1x/` 是测试物理归档位置，不等于立即停止执行。过渡期内其中仍包含
acli、打包、共享层、AgenticRL 等有效架构守卫，必须通过独立 legacy CI job 继续运行；
后续可按测试实际保护对象再拆分，不在目录迁移 PR 中顺手重写全部测试策略。

### 5.2 测试代码需要同步的内容

1. 所有 `dashscope.v2...` 导入和 patch target 改为 §3 的根实现路径。
2. `tests.unit...` 改为 `tests.legacy_1x.unit...`；为新增包层级补齐必要的
   `__init__.py`，保证既有跨测试模块导入可解析。
3. `tests.legacy...` 等动态导入按新包路径调整；`tests/data` 引用保持不变。
4. 修正依赖 `__file__` 层级的 repository-root 计算：
   - 原 `tests/v2/` 文件上移一级；
   - 原 unit / integration / legacy 文件下移一级。
5. 重写依赖物理目录或模块所有权的架构测试，而不是仅替换字符串：
   - AgentStudio shim 身份断言；
   - reinforcement 三命名空间、模块身份和懒加载断言；
   - package root 所有权断言；
   - layer decoupling 的 v2 路径、registry 物理路径和中立包清单。
6. 测试文档中的命令统一改为当前套件、legacy 套件和全量套件三种形式。

## 6. 测试流程与配置修正

### 6.1 pytest 与 CI

建议固定以下命令，避免“移动目录后 CI 实际没有运行当前 2.0 测试”：

```bash
# 当前套件（根 tests，排除过渡归档）
python -m pytest tests --ignore=tests/legacy_1x -q

# 过渡 legacy 套件（过渡期继续执行）
python -m pytest tests/legacy_1x -q

# 全量回归
python -m pytest tests -q
```

- `pytest.ini` 的 `testpaths = tests` 可保留，但必须通过 `norecursedirs` 或默认命令
  明确当前套件是否包含 `legacy_2x`；配置和 CI 的语义必须一致。
- `.github/workflows/unit_test.yml` 当前只运行 `tests/unit`，需改为当前根测试的
  coverage job，并增加 `tests/legacy_1x` 的独立 job；不能只把字符串改成
  `tests/legacy_1x/unit` 后宣称 2.0 已验收。
- coverage 继续以实际运行的测试为准；若增加 source / omit 配置，根实现与
  `legacy_2x` 的覆盖口径要分别说明。
- `requirements-test.txt`、`setup.py`、`.gitignore` 中关于 `tests/unit` 的说明同步更新。

### 6.2 pre-commit 与静态检查

- `.pre-commit-config.yaml` 中所有 `tests/legacy` exclude 改为
  `tests/legacy_1x/legacy`，逐个核对正则锚点，避免旧排除失效后突然对归档代码启用
  新规则。
- 对当前根测试保持现有 Black / lint 规则；归档测试沿用迁移前的排除策略。
- 路径迁移后运行 import / patch target 搜索，禁止残留指向已删除的
  `dashscope/v2` 或 `tests/v2` 的有效配置。

### 6.3 打包、规格和文档

- `MANIFEST.in`：`dashscope/v2/tokenizers/qwen.tiktoken` 改为新根路径。
- `setup.py`：核对 `find_packages()`、package data、console entry point；
  重新生成构建元数据，不手改 `dashscope.egg-info/SOURCES.txt`。
- `specs/chat.yaml` 的 v2 module mapping 改为根模块。
- 根 README、迁移指南和仍有效的架构文档更新规范导入路径；历史设计稿可保留当时
  背景，但若作为操作手册使用则必须注明新路径。
- `docs/source` 的历史 API 文档如继续发布，导入目标需指向保留的兼容路径或在
  `legacy_2x` 下线时同步下线。

## 7. 破坏性变化与兼容边界

| 项目 | 本次处理 |
|---|---|
| `dashscope.v2` 内部实现路径 | 退出，内部引用全部迁移；不保留第二实现 |
| 根公开 API（`dashscope.Client` 等） | 名称和行为保持，改为根实现直接提供 |
| `dashscope.agentstudio` | 路径和公开行为保持，shim 位置变为实现 |
| `dashscope.finetune.agentic_rl` | 保留 1.x 子类、动词和 `FineTune` 返回 |
| 当前 AgenticRL 实现 | 新实现位置为 `dashscope.agentic_rl`，仍返回 `SubmitResult` |
| `dashscope.reinforcement` | 公开路径保持，变为实现位置 |
| `dashscope.finetune.reinforcement` | 保留 alias 和模块身份 |
| 旧 `dashscope.client.*` deep import | 随旧包进入 `legacy_2x`，列入迁移指南 |
| 其他归档旧 API | 过渡期从 `legacy_2x` 可用；根级保留项按 §3.4 单独审查 |
| `tests/data` | 不变 |
| `examples` | 已完成，本次不改 |

## 8. 分阶段实施建议

1. **准备与基线**：提交前记录完整测试基线；确认工作区中的 AgentStudio /
   reinforcement 修改已纳入基线，避免迁移覆盖未提交工作。
2. **包迁移**：先处理 `legacy_2x` 和目标冲突，再移动常规 v2 文件，最后处理
   AgentStudio / AgenticRL / reinforcement 特殊映射与根导出。
3. **引用迁移**：修改静态和动态导入、entry point、package data、specs；
   搜索确认没有有效实现引用旧 v2 路径。
4. **测试迁移**：提升当前测试、归档 legacy 测试、修复包导入和路径深度，
   重写身份 / 分层架构断言。
5. **流程迁移**：修正 pytest、CI、coverage、pre-commit 和测试文档。
6. **验收与发布说明**：执行 §9；单独列出 `legacy_2x` 过渡政策和后续下线事项。

每个阶段保持可审查的小提交；目录移动与大规模行为改写不混在同一提交中。
本次只做位置、引用和必要兼容调整，不顺便重构无关 API。

## 9. 验收标准

### 9.1 当前实现

```bash
python -m pytest tests --ignore=tests/legacy_1x -q
coverage run -m pytest tests --ignore=tests/legacy_1x
coverage report -m
```

至少额外固定：

- 根 `Client` / `AsyncClient`、配置、错误、transport、types、protocols 可导入；
- AgentStudio 公开导入、deep import、SSE 和错误别名行为；
- 两类 AgenticRL 返回类型和 1.x 动词；
- reinforcement 两条公开路径的模块身份、runpy、uvicorn dotted path 和可选依赖懒加载；
- tokenizer 资源能从源码树和 wheel 读取。

### 9.2 过渡 legacy 与全量

```bash
python -m pytest tests/legacy_1x -q
python -m pytest tests -q
pre-commit run --all-files
git diff --check
```

并验证：

- CI 当前 job 确实执行根测试，legacy job 确实执行 `tests/legacy_1x`；
- 默认 pytest 的包含 / 排除语义与文档一致；
- 构建 sdist / wheel，检查 `legacy_2x` 过渡包、tokenizer 数据和 entry point；
- 在干净环境安装 wheel 后执行当前 API 与保留兼容路径的导入冒烟；
- 搜索无残留的 `dashscope/v2`、`tests/v2` 有效路径引用；
- `examples/` 不再产生迁移 diff，既有示例冒烟保持通过。

## 10. 非目标

- 不修改 error registry 或其 vendored 错误内容。
- 不重新设计 AgentStudio / AgenticRL 错误体系。
- 不在本次删除 `legacy_1x` 或 `legacy_2x`。
- 不把 acli 移入 `legacy_2x`。
- 不移动 `tests/data`。
- 不重复修改已经完成的 `examples/`。
- 不在目录迁移中增加无关功能、重构或清理生成物。
