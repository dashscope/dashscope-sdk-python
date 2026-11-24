## Documentation Maintenance Guide

> **English** | [中文](README_zh.md)

The documentation is a MkDocs Material project (the same stack as the FastAPI
docs): site configuration lives in `mkdocs.yml` at the repository root, page
sources live in `docs/`, covering the dashscope 2.0 API. The API reference is
generated automatically from source docstrings via mkdocstrings (`::: module.Class`
directives); guides and the quickstart are plain Markdown.

The site is bilingual: **English is the default language** (`docs/en/`,
published at the site root) and **Simplified Chinese** (`docs/zh/`, published
under `/zh/`), with a language selector in the header (provided by the
`mkdocs-static-i18n` plugin, the same approach FastAPI uses).

1. Multi-language directory conventions

    ```text
    docs/
    ├── en/          # default language (English), mirrors the same subtree
    │   ├── index.md / features.md / quickstart.md / cli.md
    │   ├── develop.md / about.md / release-notes.md
    │   ├── guides/  └ reference/
    ├── zh/          # Chinese translation, one-to-one with en/
    ├── overrides/   # Material template override (per-page action block)
    ├── hooks.py     # build hook: inject SDK version from dashscope/version.py
    └── assets/ + stylesheets/   # logo and custom CSS
    ```

    - New pages: write the English original in `docs/en/` first, then add the
      same-named translation in `docs/zh/`; untranslated pages automatically
      fall back to the English original.
    - Navigation titles are written in English in the `nav` section of
      `mkdocs.yml`; Chinese titles are mapped through the i18n plugin's
      `nav_translations`.
    - The `:::` directives on API reference pages are identical in both
      languages (docstrings themselves are English); the zh version
      translates headings and introductions, the API body stays English.
    - Cross-page auto-references (autorefs) are configured with
      `resolve_closest: true`, so references inside Chinese pages point to
      Chinese pages and English pages point to English pages.
    - `docs/overrides/main.html` overrides the Material template to render
      the per-page action block (report issue / contribute); the directory
      is excluded from the site via `exclude_docs`.

2. Local build & preview
    ```shell
    # From the repository root:
    python -m pip install -r requirements-doc.txt
    python -m pip install -e .
    mkdocs serve
    # Open http://127.0.0.1:8000 in a browser for live preview
    # Or build the static site only:
    mkdocs build   # output in site/
    ```
    `site/` is git-ignored and regenerated on every build.

3. CI publishing

    `.github/workflows/docs.yml` runs `mkdocs build --strict` as a check on
    every PR (any warning or broken reference fails the build)
    and on pushes to `main` that touch `docs/`, `dashscope/`, `mkdocs.yml` or
    `requirements-doc.txt`. Pushes to `main` additionally deploy the HTML to
    GitHub Pages (always showing the latest docs, independent of the
    tag-driven release workflow).

    One-time repository setup required: **Settings → Pages → Build and
    deployment → Source = "GitHub Actions"**. After the first successful
    deployment the site is available at `https://<owner>.github.io/<repo>/`.

4. Docstring format

    We use Google-style docstrings as the standard. References:
    1. Google Python style guide, docstrings [link](http://google.github.io/styleguide/pyguide.html#381-docstrings)
    2. Google docstring examples [link](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
    3. Example: torch.nn.modules.conv [link](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d)
    4. Using a load function as an example:

    ```python
    def load(file, file_format=None, **kwargs):
        """Load data from json/yaml/pickle files.

        This method provides a unified api for loading data from serialized files.

        Args:
            file (str or :obj:`Path` or file-like object): Filename or a file-like
                object.
            file_format (str, optional): If not specified, the file format will be
                inferred from the file extension, otherwise use the specified one.
                Currently supported formats include "json", "yaml/yml".

        Examples:
            >>> load('/path/of/your/file')  # file is stored in disk
            >>> load('https://path/of/your/file')  # file is stored on internet
            >>> load('oss://path/of/your/file')  # file is stored in petrel

        Returns:
            The content from the file.
        """
    ```