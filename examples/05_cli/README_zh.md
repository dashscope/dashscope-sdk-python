# 05 命令行

> [English](README.md) | **中文**

其他各层都是从 Python 里驱动 SDK，这一层是从 **shell** 里驱动它：`dashscope` 命令是同一批能力的另一个契约面，而「可脚本化」正是这个契约带来的。三档输出格式、分层退出码、固定的一行 stderr 错误、四层配置及其明确的优先级、以及在无终端的 CI 作业里遇到破坏性命令时拒绝替你猜。下面每一条保证都是构建脚本可以依赖的承诺——而如果你只读过 Python 示例，这些保证一条都看不见。

| 文件 | 演示内容 | 联网 |
| --- | --- | --- |
| [`cli_output_and_errors.py`](./cli_output_and_errors.py) | 从内部看契约：一个假 client，用 `typer.testing.CliRunner` 在进程内驱动，把 table / json / yaml 三档、NDJSON 流式、`@file` / stdin / `--messages` 三种输入形态，以及四类失败档位的 stdout / stderr / 退出码并排打印出来 | 否 |
| [`cli_ci_harness.py`](./cli_ci_harness.py) | 从外部看契约：用标准库在临时端口起一个 mock 网关，驱动**真实子进程里的真实 CLI**——解析 `-o json`、按退出码分支与重试、逐行消费 NDJSON 流、按作业隔离配置，并闯过破坏性命令的确认关 | 否（默认离线 mock；`CLI_HARNESS_LIVE=1` 打真实网关，仅只读场景） |

建议按这个顺序读。第一个演示每个通道**吐出什么**，第二个演示**流水线该怎么消费它**。

---

## 安装与第一次调用

包里注册了 console script，所以 `pip install -U dashscope` 之后 `PATH` 上就有一个 `dashscope` 命令：

```bash
dashscope --help
dashscope version
```

在没装 console script 的源码检出里，模块入口完全等价：

```bash
python3 -m dashscope.cli2 --help
python3 -m dashscope.cli2 version
```

密钥配一次，两种方式都行：

```bash
export DASHSCOPE_API_KEY="sk-xxx"
dashscope chat create -m qwen-plus --prompt "用一句话介绍杭州"
```

**全局选项必须放在子命令前面。**它们属于 app，不属于 `chat create`：

```bash
dashscope -k sk-xxx --output json chat create -m qwen-plus --prompt "hi"   # 正确
dashscope chat create -m qwen-plus --prompt "hi" --output json             # exit 2，usage
```

最后这条不是小怪癖——它是这个 CLI 最容易让人一头雾水的失败方式，`cli_output_and_errors.py` 专门演示了它。

---

## 命令地图

| 命令 | 用途 |
| --- | --- |
| `dashscope version` | 打印已安装的 SDK 版本。什么都不碰，所以是你手上最便宜的冒烟测试 |
| `dashscope chat create` | 一次 chat 补全：`--prompt` / `--messages`、`--stream`、`--tools`、采样参数，其余走 `--parameters` |
| `dashscope ft create` | 起一个微调作业；`--wait` 带 spinner 阻塞到终态，`--timeout` 限定等待时长 |
| `dashscope ft get` | 查看单个作业 |
| `dashscope ft list` | 分页列出作业：`--page`、`--size` |
| `dashscope ft stream` | 跟随服务端事件流，直到服务端关闭它 |
| `dashscope ft cancel` | 取消运行中的作业。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope ft delete` | 删除作业及其产出模型。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope config set` | 往 `~/.dashscope/config.toml` 写一个键 |
| `dashscope config get` | 读回一个键（api_key 默认打码，除非 `--reveal`） |
| `dashscope config list` | 每个键、它的生效值，以及这个值来自哪一层 |

同一个入口仍然能跑 1.x 命令，所以老的 shell 脚本不会断。`main.py` 把 argv 分成三类：上面这棵 v2 命令树、一组**已废弃**命令（`generation`、`application`、`image-synthesis`、`multimodal-conversation`、`speech-synthesis`、`transcription`、`video-synthesis`、`image-generation`、`multimodal-embedding`、`agentic-rl`，委派前先打印提示），以及一组**等待 v2 替代**的命令（`auth`、`deployments`、`embeddings`、`files`、`models`、`oss`、`rerank`、`rl`、`tokenization`，静默委派）。设 `DASHSCOPE_CLI_NO_DEPRECATION=1` 可以在构建日志里静音这条提示。不带子命令的裸 `dashscope` 会交互式打开内置的 `expert` agent。

---

## 输出格式

`-o table`（默认）给人看，`-o json` 和 `-o yaml` 给机器看。这个切分是刻意的，并且在三个地方被强制执行：

| 行为 | table | json | yaml |
| --- | --- | --- | --- |
| 渲染方式 | Rich，会折行会上色 | stdout 上 `json.dumps(indent=2, ensure_ascii=False)` | stdout 上 `yaml.safe_dump(allow_unicode=True, sort_keys=False)` |
| 回答文本 | stdout | 在文档里 | 在文档里 |
| `reasoning_content`、`usage:`、分页信息 | **只走 stderr** | 是文档的一部分 | 是文档的一部分 |
| 流式 | 增量文本写 stdout | **NDJSON** | **NDJSON**（不是 YAML 文档） |
| 传输层 `headers` | 永不出现 | 剥掉，除非 `-v` | 剥掉，除非 `-v` |

有两个后果值得内化：

1. table 模式下 `dashscope chat create ... > answer.txt` 只会捕获回答本身。推理过程和 token 计数走 stderr，所以重定向永远不会捞到装饰性内容。
2. `--stream -o yaml` 是个阅读陷阱：它每行吐一个 **JSON** 对象，和 `--stream -o json` 一模一样。流式输出天然是按行的，而多文档 YAML 流对 `while read` 极不友好。

管道配方：

```bash
dashscope -o json chat create -m qwen-plus --prompt "hi" | jq -r '.choices[0].message.content'
dashscope -o json chat create -m qwen-plus --prompt "hi" | jq -r '.usage.total_tokens'
dashscope -o json ft list --size 50 | jq -r '.jobs[] | "\(.job_id)\t\(.status)"'
dashscope -o json chat create -m qwen-plus --prompt "hi" --stream \
  | jq -j '.choices[0].delta.content // empty'
```

需要原始 HTTP 响应头（提工单用）和失败堆栈时加 `-v` / `--verbose`。流水线里别加：`headers` 是传输元数据，不属于结果。

---

## 退出码与错误格式

退出码才是契约，不要去 grep 错误文本。

| 退出码 | 含义 | 可重试？ |
| --- | --- | --- |
| `0` | 成功 | — |
| `1` | `server`、`ratelimit`、`network`——服务或链路失败 | 可以，带退避。`ratelimit` 要尊重 `Retry-After` |
| `2` | `usage`——你的命令行、你的输入文件，或缺了 `--yes` | 不行，改调用方式 |
| `3` | `auth`——凭据被拒 | 不行，改密钥 |
| `4` | `notfound`——这个 id 不存在 | 不行，改 id |

每次失败都会在 stderr 上打印一行固定形状的内容，后面跟可选的 dim 上下文：

```text
Error [ratelimit]: Rate limit exceeded.
  request_id: 7574ee8f-38a3-4b1e-9280-11c33ab46e51
  solution: Reduce your request frequency.
```

`Error [category]: message` 是稳定的；两行 dim 只在有内容可说时才出现。`request_id` 是提工单需要的东西，`solution` 是注册表自带的指引——单独一行，这样你永远能分清哪些字来自网关。

HTTP 状态码如何变成 category（`cli2/errors.py` 的 `_STATUS_CATEGORY`）：

| 状态码 | category | 退出码 |
| --- | --- | --- |
| 400、413 | `usage` | 2 |
| 401、403 | `auth` | 3 |
| 404 | `notfound` | 4 |
| 429 | `ratelimit` | 1 |
| 500、503、504 | `server` | 1 |

SDK 侧的失败没有 HTTP 状态码，所以按注册表 code 归类：`sdk.ConnectionError` 和 `sdk.Timeout` → `network`（1），`sdk.InvalidApiKey` → `auth`（3），`sdk.IoError` → `usage`（2，因为「你指名的文件读不了」是你的错）。无法归因的一律落到 `server`。

shell 里的重试封装：

```bash
attempt=1
until dashscope -o json chat create -m qwen-plus --prompt "hi" > out.json; do
  code=$?
  case $code in
    1) [ $attempt -ge 5 ] && exit 1
       sleep $(( attempt * 2 )); attempt=$(( attempt + 1 )) ;;
    *) exit $code ;;   # 2 / 3 / 4 不会自己好
  esac
done
```

`cli_ci_harness.py` 的场景 2 是同一段逻辑的 Python 版，配一个「第一次限流、第二次放行」的 mock 网关，让你亲眼看它恢复。

---

## 配置

优先级，从高到低：

```text
命令行 flag  >  -c KEY=VALUE  >  环境变量  >  配置文件  >  内置默认值
```

所以 `-k sk-a -c api_key=sk-b` 用的是 `sk-a`：专用 flag 是更具体的写法，永远赢。

配置文件是 `~/.dashscope/config.toml`。用 `DASHSCOPE_CONFIG` 重定向它——这就是给单个 CI 作业一份独立配置、又不碰任何人 home 目录的办法：

```bash
DASHSCOPE_CONFIG=/tmp/job-42/config.toml dashscope config set default_model qwen-plus
```

顶层键都是 SDK 键，原样传给 `Client(...)`：`api_key`、`protocol`、`base_url`、`workspace`、`region`、`timeout`、`max_retries`、`http_max_connections`、`http_max_keepalive_connections`、`http_keepalive_expiry`。`[cli]` 表里只放两个 CLI 专用默认值：`output` 和 `default_model`。写成 `cli.output` / `cli.default_model` 可以，直接写也行——`config set output json` 知道该放哪。

```toml
api_key = "sk-xxx"
protocol = "native"

[cli]
output = "json"
default_model = "qwen-plus"
```

三个实践上真正要紧的细节：

- `config set api_key` **绝不回显值**，并且密钥落盘的那一刻文件就被 chmod 成 `0600`。终端录制和构建日志都抓不到你刚敲进去的东西。
- `config get api_key` 会打码：前 4 位 + 星号 + 后 2 位（`sk-l*******mo`）。`--reveal` 打印完整值，留给你确实需要核对的那一次。
- `config get` 读的是**文件本身**，`config list` 报告的是**生效值**加上赢的那一层。两者可能不一致——而这个不一致*就是*优先级这一课，`cli_ci_harness.py` 的场景 4 会把它讲明白，而不是让它看起来像个 bug。

`-c` 会被立刻校验，在任何命令上都一样，包括根本不联网的命令：

```bash
dashscope -c defualt_model=qwen-max version
# Error [usage]: unknown configuration key 'defualt_model'; known keys: api_key, base_url, ...
# exit 2
```

这是刻意的。流水线里的一个拼写错误应该在最便宜的命令上就暴露出来，而不是三个阶段之后变成一个「模型莫名其妙不对」的问题。

---

## 输入约定

每个文本选项都接受三种形态：

| 形态 | 含义 |
| --- | --- |
| `--prompt "text"` | 字面值 |
| `--prompt @path/to/file.txt` | 读文件（会展开 `~`）。`@` 形态能让长 payload 和工具 schema 不进 shell history、不出现在 `ps` 输出里 |
| `--prompt -` | 读 stdin 到 EOF |

三者都没给时，**管道进来的** stdin 会被自动拾取，所以下面这样是可行的：

```bash
cat question.txt | dashscope chat create -m qwen-plus
```

会话来源优先级：`--messages` > `--prompt`（+ `--system`）> 管道 stdin。三者都拿不到东西时，你会得到 exit 2 和 `no input given: pass --prompt, --messages, or pipe text on stdin`。

`--messages` 接受三种形态，并自己判断你给的是哪一种：

```bash
# 一个消息对象的 JSON 数组
dashscope chat create -m qwen-plus \
  --messages '[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"},{"role":"user","content":"继续"}]'

# JSONL——每行一个消息对象，用循环拼最省事
dashscope chat create -m qwen-plus --messages @turns.jsonl

# 一个带 messages 键的对象，也就是把抓下来的请求体原样回放
dashscope chat create -m qwen-plus --messages @last_request.json
```

JSON 写错会报位置，不会耸肩：`--messages: invalid JSON (line 1 column 23)`。

`--tools` 和 `--parameters` 同样是 JSON，同样接受 `@file`。`--tools` 必须是 **JSON 数组**，给它一个对象是 `usage` 错误（exit 2）。消息里的本地图片和视频路径会原样传给 SDK，由 SDK 自己上传——你不需要预先上传。

---

## 配方

可直接复制。标 🔑 的需要真实的 `DASHSCOPE_API_KEY`，其余完全离线。

**一次性问答存进变量** 🔑

```bash
answer=$(dashscope -o json chat create -m qwen-plus --prompt "用一句话介绍杭州" \
         | jq -r '.choices[0].message.content')
```

**边到边流式打到终端** 🔑

```bash
dashscope chat create -m qwen-plus --prompt "写一首关于春天的短诗" --stream
```

**长 prompt 走文件，绝不走 argv** 🔑

```bash
dashscope chat create -m qwen-plus --prompt @brief.md --system "你是资深编辑"
```

**工具定义走文件** 🔑

```bash
dashscope chat create -m qwen-plus --prompt @question.txt --tools @tools.json -o json
```

**循环里拼 JSONL 做多轮** 🔑

```bash
: > turns.jsonl
while read -r line; do
  printf '{"role":"user","content":"%s"}\n' "$line" >> turns.jsonl
done < questions.txt
dashscope chat create -m qwen-plus --messages @turns.jsonl
```

**只重试有可能自己好的** 🔑 ——见上面退出码一节的 shell 封装。

**微调并阻塞到尘埃落定** 🔑

```bash
dashscope ft create -m qwen-plus -t "$FILE_ID" --n-epochs 2 --wait --timeout 1800
```

`ft create -m` **不会**回落到 `default_model`。这是有意为之：chat 的默认模型不是微调基座，而静默微调错模型比报个错糟糕得多。

**跟随作业事件** 🔑

```bash
dashscope ft stream "$JOB_ID"
```

**CI：破坏性命令需要 `--yes`** 🔑

```bash
dashscope ft delete "$JOB_ID" --yes
```

没有 `--yes` 又没有终端时，它以 exit 2 拒绝，而不是替你猜。

**按作业隔离配置** 🔑

```bash
export DASHSCOPE_CONFIG="$CI_JOB_DIR/config.toml"
dashscope config set default_model qwen-plus
dashscope config set api_key "$DASHSCOPE_API_KEY"
dashscope chat create --prompt "hi"     # 不传 -m：来自 default_model
```

**干净的日志**——不要颜色、不要废弃提示：

```bash
export NO_COLOR=1 DASHSCOPE_CLI_NO_DEPRECATION=1
```

`NO_COLOR=1` 和 `--no-color` 是一回事。Rich 本来就能识别管道，所以这一条主要是给那种「先分配伪终端再把它倒进日志」的 runner 用的。

---

## 运行示例

两个都离线、确定性，都不需要密钥。在仓库根目录、且检出可导入的情况下：

```bash
PYTHONPATH=. python3 examples/05_cli/cli_output_and_errors.py
PYTHONPATH=. python3 examples/05_cli/cli_ci_harness.py
```

（只有包没被 pip 安装时才需要 `PYTHONPATH=.`。装了的话 `python3 examples/05_cli/cli_output_and_errors.py` 就够了。harness 会为自己的子进程设好这个变量，所以两种情况下都能跑。）

两个都会在最后打印一段小结，并且只有全部检查通过才 exit `0`——`cli_output_and_errors.py` 会断言它跑的每条命令的退出码，`cli_ci_harness.py` 会断言全部七个场景。两者都不读也不写你真实的 `~/.dashscope/config.toml`：harness 把 `DASHSCOPE_CONFIG` 指向一个临时目录，退出时删掉。

要让 harness 的只读场景打真实网关：

```bash
CLI_HARNESS_LIVE=1 DASHSCOPE_API_KEY=sk-xxx python3 examples/05_cli/cli_ci_harness.py
```

Live 模式跳过所有故障注入场景，并且**绝不**运行 `ft cancel` 或 `ft delete`。只有你同时把 `DASHSCOPE_JOB_ID` 设成一个你自己拥有的作业时，它才会跑 `ft stream`——对真实账户猜一个 id，正是「示例删掉了重要东西」的标准死法。

---

## 不读源码一定会踩的坑

1. **全局选项不能后置。**`dashscope chat create --prompt "hi" -o json` 得到的是 exit 2，不是格式化好的回答。顶层 `--help` 里的那些选项，位置是相对于 app 的。
2. **`--stream -o yaml` 是 NDJSON，不是 YAML。**每行一个 JSON 对象，和 `-o json` 一样。YAML 解析器读不了它。
3. **破坏性命令在 CI 里会拒绝执行。**没有终端 + 没有 `--yes` = exit 2。你的流水线需要在 `ft cancel` 和 `ft delete` 上加 `--yes`，而且必须是显式加——因为重点就在于这个命令不肯替你做决定。
4. **`ft create -m` 永不回落到 `default_model`。**只有 `chat create` 会。源码注释写了原因：chat 的默认模型不是微调基座。
5. **`config get` 和 `config list` 可能互相矛盾。**`get` 读文件，`list` 解析各层。两者不一致时，`list` 的 `source` 列在告诉你是哪一层赢了——通常是你忘了自己 export 过的环境变量。
6. **table 模式下有用的元数据在 stderr 上。**`usage:`、推理文本和 `page x/y, total n` 是刻意不放 stdout 的，这样重定向才干净。想把它们当数据拿，就用 `-o json`，别去捕获 stderr。

另见 [`../README_zh.md`](../README_zh.md) 了解这一层在整体中的位置，以及 [`../02_advanced_engineering/local_mock_debugging.py`](../02_advanced_engineering/local_mock_debugging.py)——那是这里的 mock 网关在 SDK 侧的对应物。
