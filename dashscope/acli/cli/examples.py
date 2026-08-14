# -*- coding: utf-8 -*-
"""Example management functions."""
# pylint: disable=too-many-branches,too-many-return-statements
# pylint: disable=too-many-statements

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.console import Console

console = Console()

_DEFAULT_EXAMPLES_REPO = ""  # set via examples_repo in ~/.acli/config.toml
_DEFAULT_EXAMPLES_BRANCH = "main"

_KNOWN_EXAMPLES = [
    "basic-chat",
    "dashscope-sdk-expert",
]


def _get_examples_dir() -> Path:
    """Return a local examples checkout if present (dev mode only)."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    packaged = Path(__file__).resolve().parent.parent / "examples"
    for candidate in (
        project_root / "examples",
        packaged,
    ):
        if candidate.is_dir():
            return candidate
    return packaged


def _get_configured_repo() -> tuple[str, str]:
    try:
        from dashscope.acli.config import Config

        cfg = Config.load()
        return (
            cfg.examples_repo or _DEFAULT_EXAMPLES_REPO,
            cfg.examples_branch or _DEFAULT_EXAMPLES_BRANCH,
        )
    except Exception:
        return _DEFAULT_EXAMPLES_REPO, _DEFAULT_EXAMPLES_BRANCH


def _list_examples() -> list[str]:
    examples_dir = _get_examples_dir()
    if not examples_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in examples_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    )


def _clone_example_to_temp(
    name: str,
    repo: str,
    branch: str,
    tmp: Path,
) -> Path:
    """Sparse-clone <name> from the examples repo into tmp."""
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--no-checkout",
                "-b",
                branch,
                repo,
                str(tmp),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp),
                "sparse-checkout",
                "set",
                name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp), "checkout"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        raise RuntimeError(f"git operation failed: {err or e}") from e
    except FileNotFoundError:
        raise RuntimeError("git not found; please install git") from None

    src = tmp / name
    if not src.is_dir():
        raise FileNotFoundError(f"example {name} not found in the repo")
    return src


def _list_remote_examples(repo: str, branch: str) -> list[str]:
    """List top-level example dirs in the remote repo via a blob-less clone."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                "-b",
                branch,
                repo,
                tmp,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        out = subprocess.run(
            ["git", "-C", tmp, "ls-tree", "-d", "--name-only", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    return sorted(d for d in out.splitlines() if not d.startswith((".", "_")))


def _save_examples_repo(repo: str) -> None:
    """Write examples_repo into the global ~/.acli/config.toml (create or
    replace)."""
    from dashscope.acli.config import CONFIG_DIR, CONFIG_FILE
    from dashscope.acli.utils.paths import atomic_write_text
    from dashscope.acli.utils.toml import toml_str

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    text = (
        CONFIG_FILE.read_text(encoding="utf-8") if CONFIG_FILE.exists() else ""
    )
    line = f"examples_repo = {toml_str(repo)}"
    if re.search(r"(?m)^examples_repo\s*=", text):
        text = re.sub(r"(?m)^examples_repo\s*=.*$", line, text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += line + "\n"
    atomic_write_text(CONFIG_FILE, text)


def _example_files(src: Path) -> list[Path]:
    """All copyable files under an example dir (relative-safe, no junk)."""
    return [
        p
        for p in sorted(src.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and ".git" not in p.parts
        and p.suffix != ".pyc"
    ]


def _copy_example_flat(src: Path, dst: Path, *, force: bool) -> bool:
    """Merge the example into dst/.acli/, backing up conflicting files.

    The example's own .acli/ tree merges as-is; root-level files (README,
    build scripts) land in .acli/ too; the standalone-repo root .gitignore is
    skipped. Without --force: TTY lists conflicts and asks (default Yes);
    non-TTY aborts. Conflicting files are copied to .acli/backup/ first
    (single rolling backup, replaced on each download) so the last merge
    can be rolled back.
    """
    acli_dir = dst / ".acli"
    copies: list[
        tuple[Path, Path]
    ] = []  # (source file, path relative to dst/.acli)
    for p in _example_files(src):
        rel = p.relative_to(src)
        if rel.parts[0] == ".acli":
            rel = rel.relative_to(".acli")
        elif rel.name == ".gitignore" and len(rel.parts) == 1:
            continue
        copies.append((p, rel))

    conflicts = [(p, rel) for p, rel in copies if (acli_dir / rel).exists()]

    if conflicts and not force:
        listing = "\n".join(f"  .acli/{rel}" for _, rel in conflicts)
        console.print(
            f"[yellow]These files already exist; continuing will "
            f"overwrite them (originals are auto-backed up):[/yellow]\n"
            f"{listing}",
        )
        if not sys.stdin.isatty():
            console.print(
                "[yellow]Conflicts found in non-interactive mode; "
                "cancelled.[/yellow]",
            )
            console.print(
                "[dim]Pass --force to overwrite (conflict files are "
                "auto-backed up).[/dim]",
            )
            return False
        try:
            answer = input("Continue? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
        if answer in ("n", "no"):
            console.print("[dim]Cancelled.[/dim]")
            return False

    backup_dir: Path | None = None
    if conflicts:
        backup_dir = acli_dir / "backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)  # single rolling backup: latest only
        for _, rel in conflicts:
            old = acli_dir / rel
            bdst = backup_dir / rel
            bdst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, bdst)

    acli_dir.mkdir(parents=True, exist_ok=True)
    for p, rel in copies:
        target = acli_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)

    if backup_dir is not None:
        manifest = "\n".join(str(Path(".acli") / rel) for _, rel in copies)
        (backup_dir / ".manifest").write_text(
            manifest + "\n",
            encoding="utf-8",
        )

    console.print(f"[green]✓ Example copied to: {acli_dir}[/green]")
    if backup_dir is not None:
        console.print(
            f"[dim]Overwritten files backed up to: {backup_dir} "
            f"(undo with /example restore)[/dim]",
        )
    readmes = sorted(
        (
            acli_dir / rel
            for _, rel in copies
            if rel.name.lower().startswith("readme")
        ),
        key=lambda p: (len(p.parts), str(p)),
    )
    if readmes:
        console.print(f"[dim]See README for usage: {readmes[0]}[/dim]")
    console.print(
        "[dim]Edit config and skills under .acli/ "
        "to customize your Agent[/dim]",
    )
    console.print(
        "[dim]Run /setup to personalize, /help for all commands[/dim]",
    )
    return True


def _prune_empty_dirs(path: Path, stop: Path) -> None:
    while path != stop and path.is_dir():
        try:
            path.rmdir()
        except OSError:
            break
        path = path.parent


def _restore_example_backup(dst: Path) -> bool:
    """Restore the single .acli/backup: originals copied back, added files
    removed."""
    backup = dst / ".acli" / "backup"
    if not backup.is_dir():
        console.print(
            "[yellow]No restorable backup found (.acli/backup/).[/yellow]",
        )
        return False

    removed = 0
    manifest = backup / ".manifest"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            rel = line.strip()
            if not rel:
                continue
            backup_rel = Path(rel).relative_to(".acli")
            if (backup / backup_rel).exists():
                continue  # overwritten files are restored below
            target = dst / rel
            if target.is_file():
                target.unlink()
                removed += 1
                _prune_empty_dirs(target.parent, dst)

    restored = 0
    acli_dir = dst / ".acli"
    for p in sorted(backup.rglob("*")):
        if not p.is_file() or p.name == ".manifest":
            continue
        rel = p.relative_to(backup)
        target = acli_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        restored += 1

    shutil.rmtree(backup)
    _prune_empty_dirs(backup.parent, dst)
    console.print(
        f"[green]✓ Restored ({restored} original files restored, "
        f"{removed} example-added files removed)[/green]",
    )
    return True


def _print_example_list():
    examples_dir = _get_examples_dir()
    available = _list_examples()
    if not available:
        repo, branch = _get_configured_repo()
        if repo:
            try:
                available = _list_remote_examples(repo, branch)
            except Exception:
                available = []
    if not available:
        available = _KNOWN_EXAMPLES
    console.print("[bold]Available examples:[/bold]")
    for name in available:
        readme = examples_dir / name / "README.md"
        desc = ""
        if readme.exists():
            first_line = readme.read_text(encoding="utf-8").split("\n", 1)[0]
            desc = first_line.lstrip("# ").strip()
        console.print(f"  [cyan]{name}[/cyan]  — {desc}")
    console.print()
    console.print("[bold]Usage:[/bold]")
    console.print(
        "  acli example download <name>                 "
        "[dim]# merge into ./.acli/ (conflicts backed up)[/dim]",
    )
    console.print(
        "  acli example download <name> --target <dir>  "
        "[dim]# merge into the given directory[/dim]",
    )
    console.print(
        "  acli example download <name> --force         "
        "[dim]# overwrite without asking (still backed up)[/dim]",
    )
    console.print(
        "  acli example download <name> --repo <url>    "
        "[dim]# use this repo and save it to config[/dim]",
    )
    console.print(
        "  acli example restore                         "
        "[dim]# restore the latest backup (undo merge)[/dim]",
    )


def _handle_example_command(args):
    """Handle `acli example [list | download <name>]`."""
    examples_dir = _get_examples_dir()

    if not args:
        _print_example_list()
        return

    sub = args[0]

    if sub in ("list", "--list", "-l"):
        _print_example_list()
        return

    # restore [--target DIR]
    if sub == "restore":
        target_dir = Path.cwd()
        if "--target" in args or "-t" in args:
            idx = (
                args.index("--target")
                if "--target" in args
                else args.index("-t")
            )
            if idx + 1 < len(args):
                target_dir = Path(args[idx + 1])
        _restore_example_backup(target_dir)
        return

    # download <name> [--target DIR] [--force]
    if sub == "download":
        if len(args) < 2:
            console.print(
                "[yellow]Error: download requires an example name[/yellow]",
            )
            available = _list_examples() or _KNOWN_EXAMPLES
            console.print(f"[dim]Available: {', '.join(available)}[/dim]")
            return
        name = args[1]
        target_dir = Path.cwd()
        if "--target" in args or "-t" in args:
            idx = (
                args.index("--target")
                if "--target" in args
                else args.index("-t")
            )
            if idx + 1 < len(args):
                target_dir = Path(args[idx + 1])
        force = "--force" in args or "-f" in args
        provided = None
        if "--repo" in args:
            idx = args.index("--repo")
            if idx + 1 < len(args):
                provided = args[idx + 1]

        src = examples_dir / name
        if src.is_dir():
            _copy_example_flat(src, target_dir, force=force)
            return

        repo, branch = _get_configured_repo()
        if not repo and not provided and sys.stdin.isatty():
            console.print("[yellow]No examples repo configured.[/yellow]")
            try:
                provided = input(
                    "Enter the examples git repo URL (empty to cancel): ",
                ).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return
            if not provided:
                console.print("[dim]Cancelled.[/dim]")
                return
        repo = provided or repo
        if not repo:
            console.print(
                "[yellow]No examples repo configured; "
                "cannot download.[/yellow]",
            )
            console.print(
                "Pass one directly (auto-saved to ~/.acli/config.toml):",
            )
            console.print(
                f"  acli example download {name} --repo <git repo URL>",
            )
            console.print("Or add it manually to ~/.acli/config.toml:")
            console.print('  examples_repo = "<git repo URL>"')
            return
        if provided:
            _save_examples_repo(provided)
            console.print(
                "[dim]Saved examples_repo to ~/.acli/config.toml[/dim]",
            )
        console.print(
            f"[dim]Downloading example '{name}' from {repo}...[/dim]",
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fetched = _clone_example_to_temp(name, repo, branch, Path(tmp))
                _copy_example_flat(fetched, target_dir, force=force)
        except (RuntimeError, FileNotFoundError, OSError) as e:
            console.print(f"[yellow]Error: {e}[/yellow]")
            console.print(f"[dim]Repo: {repo} (branch: {branch})[/dim]")
            console.print(
                "[dim]You can set examples_repo in "
                "~/.acli/config.toml[/dim]",
            )
        return

    # Unknown subcommand — show hint
    console.print(f"[yellow]Unknown subcommand: '{sub}'[/yellow]")
    console.print("[dim]Usage:[/dim]")
    console.print(
        "  acli example                         [dim]# list examples[/dim]",
    )
    console.print(
        "  acli example download <name>         "
        "[dim]# download an example[/dim]",
    )
    console.print(
        "  acli example restore                 "
        "[dim]# restore the latest backup[/dim]",
    )
