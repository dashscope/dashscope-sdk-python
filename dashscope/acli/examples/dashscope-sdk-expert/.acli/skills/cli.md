---
name: cli
description: Quick reference for all dashscope CLI command groups, subcommands, options, and typical usage
---

# dashscope CLI

This file is authoritative for answering `dashscope` CLI usage questions (based on the dashscope/cli/ source code); do not invent options. For details, have the user run `dashscope <command group> --help` to verify.

## Command overview

| Command group | Subcommands | Purpose | Corresponding SDK class |
| --- | --- | --- | --- |
| generation | create | Text generation (including streaming and multi-turn messages) | dashscope.Generation |
| ft (hidden alias fine-tunes) | create/get/list/stream/cancel/delete | Fine-tuning job management | dashscope.FineTunes |
| files | upload/get/list/delete | Training file upload and management | dashscope.Files |
| deployments | create/get/list/scale/delete | Model deployment and scaling | dashscope.Deployments |
| oss | upload | Upload a file to OSS to obtain a URL | dashscope.utils.oss_utils.OssUtils |
| rerank | create | Document reranking | dashscope.TextReRank |
| embeddings | create | Text embeddings | dashscope.TextEmbedding |
| tokenization | create | Tokenization/token counting | dashscope.Tokenization |
| models | list/get | Query available models | dashscope.Models |
| application | create | Call a Bailian application | dashscope.Application |
| image-synthesis | create | Text-to-image (async task) | dashscope.ImageSynthesis |
| image-generation | create | Image generation (supports reference images) | dashscope.ImageGeneration |
| video-synthesis | create/fetch/wait/cancel/list | Full video synthesis task lifecycle | dashscope.VideoSynthesis |
| multimodal-conversation | create | Multimodal conversation | dashscope.MultiModalConversation |
| multimodal-embedding | create | Multimodal embeddings (text/image/audio) | dashscope.MultiModalEmbedding |
| transcription | create | Audio file transcription (async task) | dashscope.Transcription |
| speech-synthesis | create | Speech synthesis TTS | dashscope.HttpSpeechSynthesizer |
| rl (hidden alias agentic-rl) | run/get/cancel/logs/list/register_functions/test_functions/upload_data | Agentic RL fine-tuning | dashscope.finetune.agentic_rl.AgenticRL |

Note: code-generation and understanding still exist in the source but are deprecated/discontinued and are not registered in the CLI.

## Common usage

Prerequisite: `export DASHSCOPE_API_KEY=sk-xxx` (or pass the global `-k/--api-key`).

```bash
dashscope generation create -m qwen-plus -p "Hello" -s --temperature 0.7
dashscope ft create -m qwen-turbo -t file-xxx -v file-yyy -e 2 -b 16 -l 1e-5
dashscope files upload -f train.jsonl -p fine-tune -d "training set"
dashscope deployments create -m qwen-turbo -s demo -c 1
dashscope oss upload -m qwen-plus -f ./pic.png
dashscope rerank create -m gte-rerank -q "query" -d doc1 -d doc2 -n 3
dashscope embeddings create -m text-embedding-v3 -i "text1" -i "text2"
dashscope tokenization create -m qwen-turbo -p "text to count"
dashscope models list -p 1 -s 20
dashscope application create -a APP_ID -p "question" --has-thoughts
dashscope image-synthesis create -m wanx-v1 -p "a cat" -n 2 --size 1024*1024
dashscope image-generation create -m qwen-image -t "a cat" --image ref.png
dashscope video-synthesis create -m wanx2.1-t2v-turbo -p "ocean waves"
dashscope video-synthesis wait TASK_ID   # fetch/cancel/list take task_id the same way
dashscope multimodal-conversation create -m qwen-vl-plus -t "describe the image" --image a.jpg
dashscope multimodal-embedding create -m multimodal-embedding-v1 --text "cat" --image c.png
dashscope transcription create -m paraformer-v2 --file-url https://.../a.wav
dashscope speech-synthesis create -m cosyvoice-v1 -t "Hello" --voice longxiaochun
dashscope rl run -c rl_config.yaml -o json -v
```

Key option notes:
- generation create: `-p/--prompt`, `--messages` (JSON string), `-s/--stream`, `--temperature/--top-p/--top-k/--max-tokens/--seed/--stop/--repetition-penalty/--presence-penalty/--enable-search/--n/--result-format`
- ft create: `-t/--training-file-ids` (repeatable), `-v/--validation-file-ids`, `--mode`, `-e/--n-epochs`, `-b/--batch-size`, `-l/--learning-rate`, `-p/--prompt-loss`; get/stream/cancel/delete all take job_id as a positional argument; list uses `-p/--page`, `-s/--size`
- files: upload `-f/--file`, `-p/--purpose` (default fine-tune), `-d/--description`; all commands support `-u/--base-url`
- deployments: create `-m/-s/--suffix/-c/--capacity/--plan/--template-id`; `scale DEPLOYED_MODEL -c 2`; after create it blocks by default (timeout 3600 seconds)
- oss upload: `-f/--file`, `-m/--model`, `-k/--api-key` (reads DASHSCOPE_API_KEY), `-u/--base-url`
- rerank create: `-q/--query`, `-d/--document` (repeatable), `--return-documents/--no-return-documents`, `-n/--top-n`, `-i/--instruct`
- video-synthesis create: `--img-url/--first-frame-url/--last-frame-url/--duration/--resolution/--ratio/--seed/--prompt-extend/--watermark`; list supports `--start-time/--end-time/--model-name/--status/--region`
- speech-synthesis create: `--voice`, `--audio-format` (default wav), `--sample-rate` (default 24000), `--volume/--rate/--pitch`, `--url`
- rl: register_functions `--rollout-classpaths/--reward-classpaths/--group-reward-classpaths` (file.py:ClassName); test_functions `INSTANCE_ID -t ROLLOUT|REWARD -i '<json or file path>'`; upload_data `--training-files/--validation-files`; common `--api-key`, `-o/--output-format table|json|yaml`
- Each group's `-w/--workspace` can specify a Bailian workspace id (except files/deployments/ft/rl)

## Global options

- `-k/--api-key`: global API Key; can be placed before or after the command group name. The oss and rl/agentic-rl groups have their own local `--api-key` (with envvar DASHSCOPE_API_KEY), which is parsed locally once the command name appears.
- `-h` is automatically converted to `--help`; use `dashscope --help` or `dashscope <group> --help` for the authoritative option list.
- Legacy syntax compatibility: `dashscope fine_tunes.call` -> `fine-tunes create`, `generation.call` -> `generation create`, etc.; underscore options like `--training_file_ids` and `--api_key` are automatically mapped to hyphenated forms.
- Bare `dashscope` (no subcommand; a change in this repo): enters the built-in agent "DashScope SDK Expert" interactive mode, TUI by default; `--cli` switches to a plain REPL, `--tui` forces TUI; requires `pip install dashscope[acli]`. When stdin is not a TTY, piped text is treated as a one-shot question. Unrecognized commands do not error; the whole line is forwarded to the agent as a question. On the first interactive run, if there is no ./.acli, it asks whether to download the dashscope-sdk-expert example configuration.

## Common errors and troubleshooting

- `Error: ... AuthenticationError` (exit code 1): no Key configured; set DASHSCOPE_API_KEY or add `-k`.
- `Option '--api-key' requires an argument.` (exit code 2): `-k/--api-key` is missing a value or the value starts with `-`.
- `--messages must be a valid JSON string`: generation's --messages must be a JSON array string; watch shell quoting/escaping.
- `File ... does not exist`: local paths (-f, --image, --audio, etc.) are validated locally first; `~` expansion is supported; URLs must include a scheme.
- `Failed request_id: ..., status_code: ..., code: ..., message: ...` (exit code 1): rejected by the server; troubleshoot by code (InvalidParameter usually means a wrong model name or parameter value); HTTP 200 with a non-empty business code is also treated as a failure.
- Bare dashscope shows `Install dashscope[acli] to enable the agent`: install `dashscope[acli]`; if it reports `Interactive mode requires a terminal.`, use `dashscope "your question"` for a one-shot question instead.
