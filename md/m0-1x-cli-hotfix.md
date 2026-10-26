# M0 — 1.x CLI 快修清单

> 从 `md/dashscope-2.0-design.md` 拆出：M0 是对 **1.x** 的修复，在 1.x 维护分支单独实施，**不在 V2 分支**。
> 背景依据：`md/cli-source-level-comparison-2026-08.md` §5"立即修"清单。
> 完成后 1.x 进入仅维护期（只修 bug/安全补丁，不再有任何功能性改动；新能力一律进 2.0）。

---

## 任务清单（每项 ≈ 一次 PR）

| # | 任务 | 涉及文件 | 验收 |
| --- | --- | --- | --- |
| 0.1 | 修流式分支出错仍 exit 0 的 bug：错误后 `raise typer.Exit(1)` | `cli/generation.py:172-189`（同类流式命令逐一排查） | 流式中途注入错误，退出码=1 |
| 0.2 | 开启 shell 补全：`add_completion=True` + README 安装说明 | `cli/__init__.py:244` | bash/zsh/fish 补全可用 |
| 0.3 | 新增 `dashscope auth whoami/login/logout`，whoami 以退出码表达 key 有效性 | 新 `cli/auth.py` | 有效 key 退出 0，无效退出 2 |
| 0.4 | 流式输出改增量纯文本（TTY）；非 TTY 输出 JSON 行 | `cli/generation.py` 等生成类命令 | 管道 `| jq` 可解析 |
| 0.5 | 退出码分级（0 成功 / 1 服务端 / 2 认证 / 3 参数 / 4 限流）落到 `cli/common.py` | `cli/common.py` | CI 用例覆盖各错误类别 |

## 说明

- **为什么先做**：五项均为"小改动、高回报"，不依赖 2.0 架构；其中 0.1 是 CI 会误判成功的真实缺陷（对比调研实锤）。
- **边界**：只做修复与体验补齐，**不引入新 API 能力**（遵循"1.x 仅维护"策略）；0.3/0.5 属契约修复而非新功能。
- **与 2.0 的关系**：2.0 方案（`dashscope-2.0-design.md`）里程碑从 M1 开始；M0 的产出（auth/config 习惯、退出码约定）会被 CLI 2.0（M4）继承。
- **风险提示**：0.4 改变了默认输出格式（dict repr → 纯文本/JSON），可能影响解析旧输出的存量脚本；建议加 `DASHSCOPE_CLI_LEGACY_OUTPUT=1` 回退开关并提前公告。
