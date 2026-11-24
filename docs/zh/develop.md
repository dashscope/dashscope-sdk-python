---
description: "开发者指南：仓库结构、测试、docstring 规范与发布流程。"
---
# 开发指南

## 1. 代码风格
我们采用 [PEP8](https://www.python.org/dev/peps/pep-0008/) 作为首选代码风格。

使用以下工具进行静态检查与格式化：
- [flake8](http://flake8.pycqa.org/en/latest/)：静态检查
- [yapf](https://github.com/google/yapf)：格式化
- [isort](https://github.com/timothycrosley/isort)：import 排序

yapf 与 isort 的风格配置见 [setup.cfg](https://github.com/dashscope/dashscope-sdk-python/blob/main/setup.cfg)。
我们使用 [pre-commit 钩子](https://pre-commit.com/)，在每次提交时自动执行 `flake8`、`yapf`、`seed-isort-config`、`isort`、
`trailing whitespaces` 检查与格式化，修复 `end-of-files`，并对 `requirements.txt` 排序。
pre-commit 钩子的配置保存在 [.pre-commit-config](https://github.com/dashscope/dashscope-sdk-python/blob/main/.pre-commit-config.yaml)。
克隆仓库后，需要安装并初始化 pre-commit 钩子：
```bash
pip install -r requirements-test.txt
```
在仓库目录下执行：
```bash
pre-commit install
```

此后每次提交都会强制执行代码检查与格式化。

如果想用 pre-commit 检查全部文件，可以运行：
```bash
pre-commit run --all-files
```

如果只想格式化并检查代码，可以运行：
```bash
make linter
```

## 2. 测试

分别运行当前主线实现、归档兼容套件或完整测试套件：

```bash
python -m pytest tests --ignore=tests/legacy_1x -q
python -m pytest tests/legacy_1x -q
python -m pytest tests -q
```

当前覆盖率统计使用 `coverage run -m pytest tests
--ignore=tests/legacy_1x`，随后执行 `coverage report -m`。