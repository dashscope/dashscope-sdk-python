---
description: "Developer guide: repository layout, tests, docstring conventions and release process."
---
# Develop

## 1. Code Style
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [setup.cfg](https://github.com/dashscope/dashscope-sdk-python/blob/main/setup.cfg).
We use a [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `seed-isort-config`, `isort`, `trailing whitespaces`,
fixes `end-of-files`, and sorts `requirements.txt` automatically on every commit.
The config for the pre-commit hook is stored in [.pre-commit-config](https://github.com/dashscope/dashscope-sdk-python/blob/main/.pre-commit-config.yaml).
After you clone the repository, you will need to install and initialize the pre-commit hook:
```bash
pip install -r requirements-test.txt
```
From the repository folder:
```bash
pre-commit install
```

After this, code linters and formatters will be enforced on every commit.

If you want to use pre-commit to check all the files, you can run:
```bash
pre-commit run --all-files
```

If you only want to format and lint your code, you can run:
```bash
make linter
```

## 2. Tests

Run the current root implementation, archived compatibility suites, or the
complete suite with:

```bash
python -m pytest tests --ignore=tests/legacy_1x -q
python -m pytest tests/legacy_1x -q
python -m pytest tests -q
```

Current coverage uses `coverage run -m pytest tests
--ignore=tests/legacy_1x`, followed by `coverage report -m`.