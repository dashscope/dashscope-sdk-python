# Develop

## 1. Code Style
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use [pre-commit](https://pre-commit.com/) to check and format code
automatically on every commit. The hook suite is defined in
[.pre-commit-config.yaml](.pre-commit-config.yaml) and includes:

- [black](https://github.com/psf/black) (line length 79): formatter
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [pylint](https://pylint.readthedocs.io/): linter
- [mypy](https://mypy.readthedocs.io/): type checker
- pre-commit-hooks: trailing whitespace, encoding pragma, YAML/JSON/TOML
  checks, private key detection, etc.

Flake8 configuration lives in [setup.cfg](setup.cfg).

### Setup (one time after clone)

Install the development/test dependencies, then install the git hooks:

```bash
pip install pre-commit
pip install -r requirements-test.txt
pre-commit install
```

After this, every `git commit` runs the checks automatically; commits that
fail the hooks are rejected (auto-fixing hooks such as black will rewrite
files — review and re-stage, then commit again).

To check the whole repository manually:

```bash
pre-commit run --all-files
```

### CI

`.github/workflows/pre-commit.yml` runs `pre-commit run --all-files` on
every push and pull request, so code that bypasses the local hooks is still
blocked in CI.
