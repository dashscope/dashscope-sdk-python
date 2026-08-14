# -*- coding: utf-8 -*-
"""Skill package lifecycle: install, link, update, uninstall, search,
publish."""
# pylint: disable=too-many-branches,too-many-statements

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from dashscope.acli.config import CONFIG_DIR, WORKSPACE_DIR
from dashscope.acli.skills.package import load_skill_package
from dashscope.acli.utils import loads_toml

MANIFEST_NAME = ".acli-skills.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_skills_dir(global_: bool = False) -> Path:
    """Return the target skills directory."""
    if global_:
        return CONFIG_DIR / "skills"
    return WORKSPACE_DIR / "skills"


def _manifest_path(skills_dir: Path) -> Path:
    return skills_dir / MANIFEST_NAME


def load_manifest(skills_dir: Path) -> dict[str, Any]:
    """Load the install manifest mapping package names to source metadata."""
    path = _manifest_path(skills_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_manifest(skills_dir: Path, manifest: dict[str, Any]) -> None:
    """Persist the install manifest."""
    skills_dir.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(skills_dir)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _is_git_url(source: str) -> bool:
    """Heuristically detect git remotes."""
    s = source.strip()
    if s.startswith("git+"):
        return True
    if s.startswith("git@"):
        return True
    if s.endswith(".git"):
        return True
    parsed = urlparse(s)
    if parsed.scheme in ("http", "https") and "/" in parsed.path:
        # Could still be a raw archive; treat .git suffix explicitly above.
        return False
    return False


def _git_available() -> bool:
    try:
        subprocess.run(
            ["git", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _run_git(
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def _package_name_from_source(source: str) -> str:
    """Infer a package name from a source string or path."""
    s = source.strip().rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    parsed = urlparse(s)
    if parsed.path:
        s = parsed.path
    return Path(s).name or "skill"


def _validate_skill_name(name: str, base_dir: Path) -> None:
    """Reject skill names that could escape the skills base directory."""
    if (
        not name
        or "/" in name
        or "\\" in name
        or ".." in name
        or Path(name).is_absolute()
    ):
        raise RuntimeError(f"Invalid skill name: {name}")
    base = base_dir.resolve()
    dest = (base / name).resolve()
    if dest != base and base not in dest.parents:
        raise RuntimeError(f"Invalid skill name: {name}")


def _remove_existing(path: Path) -> None:
    """Remove a file/symlink/dir; shutil.rmtree raises OSError on symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy_package(source_dir: Path, dest_dir: Path) -> None:
    """Copy a local skill package directory to the store."""
    if dest_dir.exists() or dest_dir.is_symlink():
        _remove_existing(dest_dir)
    shutil.copytree(
        source_dir,
        dest_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )


def _git_clone(source: str, dest_dir: Path) -> None:
    """Clone a git repository into the store."""
    if not _git_available():
        raise RuntimeError(
            "git is not installed; cannot install skill from a git URL",
        )
    if dest_dir.exists() or dest_dir.is_symlink():
        _remove_existing(dest_dir)
    # Strip git+ prefix if present.
    url = source[4:] if source.startswith("git+") else source
    _run_git("clone", url, str(dest_dir))


_MAX_ARCHIVE_BYTES = 50 * 1024 * 1024


def _is_archive_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def _download_and_extract(source: str, dest_dir: Path) -> None:
    """Download a publish()-style tar.gz and extract it into dest_dir.

    Extraction validates every member path (and link target) stays inside
    the temp dir — tarballs from a remote registry are untrusted input.
    """
    import io
    import urllib.request

    with urllib.request.urlopen(source, timeout=60) as resp:
        payload = resp.read(_MAX_ARCHIVE_BYTES + 1)
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"skill archive too large "
            f"(>{_MAX_ARCHIVE_BYTES // (1024 * 1024)}MB): {source}",
        )

    if dest_dir.exists() or dest_dir.is_symlink():
        _remove_existing(dest_dir)

    with tempfile.TemporaryDirectory(dir=dest_dir.parent) as tmp:
        tmp_real = os.path.realpath(tmp)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            for member in tar.getmembers():
                target = os.path.realpath(os.path.join(tmp_real, member.name))
                if target != tmp_real and not target.startswith(
                    tmp_real + os.sep,
                ):
                    raise RuntimeError(
                        f"tar member escapes target dir: {member.name}",
                    )
                if member.islnk() or member.issym():
                    link_real = os.path.realpath(
                        os.path.join(os.path.dirname(target), member.linkname),
                    )
                    if not link_real.startswith(tmp_real + os.sep):
                        raise RuntimeError(
                            f"tar link escapes target dir: {member.name}",
                        )
            tar.extractall(tmp)
        entries = list(Path(tmp).iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise RuntimeError(
                f"tar archive has unexpected structure (expected a "
                f"single top-level directory): {source}",
            )
        entries[0].rename(dest_dir)


def _fetch_registry_index(registry_url: str) -> list[dict[str, Any]]:
    """Fetch and parse a registry index file (TOML, YAML or JSON)."""
    if not registry_url:
        return []
    try:
        if urlparse(registry_url).scheme in ("http", "https"):
            import urllib.request

            with urllib.request.urlopen(registry_url, timeout=30) as resp:
                content = resp.read().decode("utf-8")
        else:
            content = Path(registry_url).read_text(encoding="utf-8")

        if registry_url.endswith(".json"):
            data = json.loads(content)
        elif registry_url.endswith(".toml"):
            data = loads_toml(content) or {}
        else:
            data = yaml.safe_load(content) or {}

        skills = data.get("skills", []) if isinstance(data, dict) else data
        return [s for s in skills if isinstance(s, dict)]
    except Exception as e:
        raise RuntimeError(f"Failed to load skill registry: {e}") from e


def _resolve_name_from_registry(
    name: str,
    registry_url: str,
) -> dict[str, Any] | None:
    """Look up a package by name (optionally with @version) in the registry."""
    if "@" in name:
        pkg_name, version = name.rsplit("@", 1)
    else:
        pkg_name, version = name, None

    index = _fetch_registry_index(registry_url)
    candidates = [s for s in index if s.get("name") == pkg_name]
    if version:
        candidates = [
            s for s in candidates if str(s.get("version", "")) == version
        ]
    if not candidates:
        return None
    # Prefer exact version match; otherwise latest.
    return max(candidates, key=lambda s: str(s.get("version", "")))


def _validate_package(path: Path) -> tuple[bool, str]:
    """Validate a skill package directory."""
    if not path.is_dir():
        return False, f"Not a directory: {path}"
    skill_toml = path / "skill.toml"
    if not skill_toml.exists():
        return False, f"Missing skill.toml: {path}"
    pkg = load_skill_package(path)
    if pkg is None:
        return False, f"Cannot parse skill.toml: {path}"
    return True, ""


def install(
    source: str,
    skills_dir: Path | None = None,
    registry_url: str = "",
    global_: bool = False,
) -> str:
    """Install a skill package from local path, git URL, or registry name.

    Returns the installed package name.
    """
    target_dir = skills_dir or get_skills_dir(global_)
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved_source = source
    source_type = "local"

    if _is_git_url(source):
        source_type = "git"
        name = _package_name_from_source(source)
    elif _is_archive_url(source):
        source_type = "url"
        name = _package_name_from_source(source)
    elif Path(source).is_dir():
        source_type = "local"
        name = Path(source).name
    elif registry_url:
        entry = _resolve_name_from_registry(source, registry_url)
        if entry is None:
            raise RuntimeError(f"Skill not found in registry: {source}")
        resolved_source = entry.get("source", "")
        if not resolved_source:
            raise RuntimeError(f"Registry entry missing source: {source}")
        name = entry.get("name", _package_name_from_source(resolved_source))
        if _is_git_url(resolved_source):
            source_type = "git"
        elif _is_archive_url(resolved_source):
            source_type = "url"
        else:
            source_type = "local"
    else:
        raise RuntimeError(
            f"Unrecognized skill source: {source} (need a local path, "
            f"git URL, or configured skill_registry)",
        )

    _validate_skill_name(name, target_dir)
    dest_dir = target_dir / name

    if source_type == "git":
        _git_clone(resolved_source, dest_dir)
    elif source_type == "url":
        _download_and_extract(resolved_source, dest_dir)
    else:
        _copy_package(Path(resolved_source), dest_dir)

    # Re-validate using the final directory name (skill.toml may declare
    # a different name).
    pkg = load_skill_package(dest_dir)
    if pkg is None:
        _remove_existing(dest_dir)
        raise RuntimeError(
            f"Failed to parse skill package after install: {dest_dir}",
        )
    final_name = pkg.name
    try:
        _validate_skill_name(final_name, target_dir)
    except RuntimeError:
        _remove_existing(dest_dir)
        raise
    if final_name != name:
        # Rename directory to match declared name.
        new_dest = target_dir / final_name
        if new_dest.exists() or new_dest.is_symlink():
            _remove_existing(new_dest)
        dest_dir.rename(new_dest)
        dest_dir = new_dest

    manifest = load_manifest(target_dir)
    manifest[final_name] = {
        "source": resolved_source,
        "source_type": source_type,
        "version": pkg.version,
        "installed_at": _now(),
        "updated_at": _now(),
    }
    save_manifest(target_dir, manifest)
    return final_name


def link(
    source_dir: str,
    skills_dir: Path | None = None,
    global_: bool = False,
) -> str:
    """Symlink a local skill package into the store for development."""
    target_dir = skills_dir or get_skills_dir(global_)
    src = Path(source_dir).expanduser().resolve()
    valid, error = _validate_package(src)
    if not valid:
        raise RuntimeError(error)

    pkg = load_skill_package(src)
    assert pkg is not None
    name = pkg.name
    _validate_skill_name(name, target_dir)
    dest_dir = target_dir / name

    if dest_dir.exists() or dest_dir.is_symlink():
        if dest_dir.is_symlink():
            dest_dir.unlink()
        else:
            shutil.rmtree(dest_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.symlink_to(src, target_is_directory=True)

    manifest = load_manifest(target_dir)
    manifest[name] = {
        "source": str(src),
        "source_type": "link",
        "version": pkg.version,
        "installed_at": _now(),
        "updated_at": _now(),
    }
    save_manifest(target_dir, manifest)
    return name


def uninstall(
    name: str,
    skills_dir: Path | None = None,
    global_: bool = False,
) -> bool:
    """Remove an installed skill package."""
    target_dir = skills_dir or get_skills_dir(global_)
    _validate_skill_name(name, target_dir)
    dest_dir = target_dir / name
    if not dest_dir.exists() and not dest_dir.is_symlink():
        return False

    if dest_dir.is_symlink():
        dest_dir.unlink()
    else:
        shutil.rmtree(dest_dir)

    manifest = load_manifest(target_dir)
    if name in manifest:
        del manifest[name]
        save_manifest(target_dir, manifest)
    return True


def update(
    name: str,
    skills_dir: Path | None = None,
    registry_url: str = "",  # pylint: disable=unused-argument
    global_: bool = False,
) -> str:
    """Update an installed skill package."""
    target_dir = skills_dir or get_skills_dir(global_)
    _validate_skill_name(name, target_dir)
    manifest = load_manifest(target_dir)
    entry = manifest.get(name)
    if entry is None:
        raise RuntimeError(f"Installed skill not found: {name}")

    source_type = entry.get("source_type", "local")
    source = entry.get("source", "")

    if source_type == "git":
        if not _git_available():
            raise RuntimeError("git is not installed; cannot update")
        dest_dir = target_dir / name
        _run_git("pull", cwd=dest_dir)
    elif source_type == "link":
        # For linked packages, just re-validate; changes are live.
        src = Path(source)
        valid, error = _validate_package(src)
        if not valid:
            raise RuntimeError(error)
    elif source_type == "url" and source:
        dest_dir = target_dir / name
        _download_and_extract(source, dest_dir)
    elif source_type == "local" and source:
        src = Path(source)
        if not src.exists():
            raise RuntimeError(f"Local source no longer exists: {source}")
        dest_dir = target_dir / name
        _copy_package(src, dest_dir)
    else:
        raise RuntimeError(f"Cannot update source type: {source_type}")

    entry["updated_at"] = _now()
    # Refresh version from package metadata.
    pkg = load_skill_package(target_dir / name)
    if pkg is not None:
        entry["version"] = pkg.version
    save_manifest(target_dir, manifest)
    return name


def update_all(
    skills_dir: Path | None = None,
    registry_url: str = "",
    global_: bool = False,
) -> list[tuple[str, str | None]]:
    """Update all installed packages. Returns list of (name, error_or_None)."""
    target_dir = skills_dir or get_skills_dir(global_)
    manifest = load_manifest(target_dir)
    results: list[tuple[str, str | None]] = []
    for name in list(manifest.keys()):
        try:
            update(name, target_dir, registry_url, global_)
            results.append((name, None))
        except Exception as e:
            results.append((name, str(e)))
    return results


def search(
    query: str,
    skills_dir: Path | None = None,
    registry_url: str = "",
    global_: bool = False,
) -> list[dict[str, Any]]:
    """Search installed packages and optionally a registry index."""
    target_dir = skills_dir or get_skills_dir(global_)
    q = query.lower()
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Local installed packages.
    if target_dir.is_dir():
        candidates = sorted(target_dir.iterdir())
    else:
        candidates = []
    for candidate in candidates:
        pkg = load_skill_package(candidate)
        if pkg is None:
            continue
        text = f"{pkg.name} {pkg.description} {' '.join(pkg.triggers)}".lower()
        if q in text:
            results.append(
                {
                    "name": pkg.name,
                    "version": pkg.version,
                    "description": pkg.description,
                    "author": pkg.author,
                    "installed": True,
                    "source": "local",
                },
            )
            seen.add(pkg.name)

    # Registry packages.
    if registry_url:
        try:
            for entry in _fetch_registry_index(registry_url):
                name = entry.get("name", "")
                if not name or name in seen:
                    continue
                text = (
                    f"{name} {entry.get('description', '')} "
                    f"{' '.join(entry.get('tags', []))}"
                ).lower()
                if q in text:
                    results.append(
                        {
                            "name": name,
                            "version": entry.get("version", ""),
                            "description": entry.get("description", ""),
                            "author": entry.get("author", ""),
                            "installed": False,
                            "source": entry.get("source", ""),
                        },
                    )
        except Exception:
            pass

    return results


def publish(source_dir: str, output_dir: Path | None = None) -> Path:
    """Validate a skill package and create a distributable tarball."""
    src = Path(source_dir).expanduser().resolve()
    valid, error = _validate_package(src)
    if not valid:
        raise RuntimeError(error)

    pkg = load_skill_package(src)
    assert pkg is not None
    name = pkg.name
    version = pkg.version

    out = output_dir or (src.parent / ".dist")
    _validate_skill_name(name, out)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / f"{name}-{version}.tar.gz"

    if archive_path.exists():
        archive_path.unlink()

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(src, arcname=name)

    return archive_path
