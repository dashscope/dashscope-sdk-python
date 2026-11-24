## 文档维护说明

> [English](README.md) | **中文**

文档是一个 MkDocs Material 工程（与 FastAPI 文档同款技术栈）：
站点配置在根目录 `mkdocs.yml`，页面源在 `docs/`，覆盖 dashscope 2.0 API。
API 参考通过 mkdocstrings 从源码 docstring 自动生成（`::: 模块.类名`
指令）；专题指南与快速开始为普通 Markdown。

站点为双语：**英文为默认语言**（`docs/en/`，发布在站点根路径），
**简体中文**（`docs/zh/`，发布在 `/zh/` 下），顶栏有语言切换器
（由 `mkdocs-static-i18n` 插件提供，FastAPI 同款方案）。

1. 多语言目录约定

    ```text
    docs/
    ├── en/          # 默认语言（英文），镜像同样的子目录结构
    │   ├── index.md / features.md / quickstart.md / cli.md
    │   ├── develop.md / about.md / release-notes.md
    │   ├── guides/  └ reference/
    ├── zh/          # 中文翻译，与 en/ 一一对应
    ├── overrides/   # Material 模板覆写（每页操作块）
    ├── hooks.py     # 构建 hook：从 dashscope/version.py 注入版本号
    └── assets/ + stylesheets/   # Logo 与自定义 CSS
    ```

    - 新增页面：先在 `docs/en/` 写英文原版，再在 `docs/zh/` 放同名翻译；
      未翻译的页面会自动回退显示英文原版。
    - 导航标题在 `mkdocs.yml` 的 `nav` 中用英文书写，中文标题通过
      i18n 插件的 `nav_translations` 映射。
    - API 参考页的 `:::` 指令两种语言完全相同（docstring 本身是英文），
      zh 版翻译标题与导语，API 正文保持英文。
    - 跨页自动引用（autorefs）已配置 `resolve_closest: true`，
      保证中文页内的引用指向中文页、英文页指向英文页。
    - `docs/overrides/main.html` 覆写 Material 模板，在每页底部渲染
      页面操作块（报告问题 / 贡献）；该目录经 `exclude_docs` 排除，
      不进入站点。

2. 本地构建与预览
    ```shell
    # 在仓库根目录：
    python -m pip install -r requirements-doc.txt
    python -m pip install -e .
    mkdocs serve
    # 浏览器打开 http://127.0.0.1:8000 实时预览
    # 或仅构建静态站点：
    mkdocs build   # 产物在 site/
    ```
    `site/` 已被 git 忽略，每次构建都会重新生成。

3. CI 自动发布

    `.github/workflows/docs.yml` 会在每个 PR 以及推送到 `main` 且
    改动涉及 `docs/`、`dashscope/`、`mkdocs.yml` 或
    `requirements-doc.txt` 时执行 `mkdocs build --strict` 校验
    （任何警告或坏引用都会使构建失败）。
    推送到 `main` 还会把 HTML 部署到 GitHub Pages（始终展示最新文档，
    与基于 tag 的发布流程相互独立）。

    仓库需要一次性设置：**Settings → Pages → Build and deployment →
    Source = "GitHub Actions"**。首次部署成功后，站点地址为
    `https://<owner>.github.io/<repo>/`。

4. docstring 格式

    我们采用 Google 风格的 docstring 作为标准，参考以下资料：
    1. Google Python 风格指南 docstring [链接](http://google.github.io/styleguide/pyguide.html#381-docstrings)
    2. Google docstring 示例 [链接](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
    3. 范例：torch.nn.modules.conv [链接](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d)
    4. 以 load 函数为例：

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