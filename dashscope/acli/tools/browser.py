# -*- coding: utf-8 -*-
"""Browser tools — Playwright-based web scraping for JS-rendered pages.

Security hardening:
- URL scheme validation (http/https only)
- SSRF protection against private/internal IPs (DNS-level + route-level
  anti-rebinding interception)
- Screenshot path traversal prevention
- CSS selector sanitisation
- Output bounded by text truncation in each tool

Performance hardening:
- Browser instance reuse within a session (thread-safe)
- atexit cleanup to avoid zombie chromium processes
"""

from __future__ import annotations

import asyncio
import atexit
import ipaddress
import os
import socket
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from dashscope.acli.tools.registry import PermissionLevel, tool

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_PLAYWRIGHT_INSTALL_HINT = (
    "Error: playwright is not installed. Run "
    "`pip install playwright && playwright install chromium`"
)

# Schemes we allow.
_ALLOWED_SCHEMES = {"http", "https"}

# Timeout bounds: minimum 1s, maximum 120s.
_MIN_TIMEOUT_MS = 1_000
_MAX_TIMEOUT_MS = 120_000

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


def _validate_url(url: str) -> str:
    """Validate and normalise a URL, raising ValueError on anything unsafe.

    Checks performed:
    - Scheme must be http or https (blocks file://, javascript:, data:, etc.)
    - Hostname must resolve to a non-private IP (SSRF protection against
      cloud metadata endpoints, localhost, LAN addresses, etc.)

    Note: This is a DNS-level check at validation time. DNS rebinding attacks
    (where the first resolution returns a public IP but the browser's actual
    request resolves to a private IP) are mitigated by the route-level
    interception in _launch_page.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed; "
            f"only http/https are supported",
        )
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL is missing a hostname")

    # Block raw IP addresses pointing to private ranges directly.
    _is_ip = False
    try:
        ip = ipaddress.ip_address(hostname)
        _is_ip = True
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
        ):
            raise ValueError(
                f"Access to internal network address {hostname} "
                f"is not allowed (SSRF protection)",
            )
        return url
    except ValueError:
        if _is_ip:
            raise
        # Not a raw IP — fall through to DNS resolution check.

    # Resolve hostname and check for private/reserved IPs.
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(hostname, port)
    except socket.gaierror as exc:
        raise ValueError(
            f"Cannot resolve hostname '{hostname}': {exc}",
        ) from exc

    for _family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
        ):
            raise ValueError(
                f"Access to internal network address {ip_str} "
                f"is not allowed (SSRF protection)",
            )
    return url


def _clamp_timeout(timeout: int) -> int:
    """Ensure timeout is within reasonable bounds."""
    return max(_MIN_TIMEOUT_MS, min(int(timeout), _MAX_TIMEOUT_MS))


def _validate_screenshot_path(output_path: str) -> str:
    """Ensure the screenshot path doesn't escape the current working
    directory."""
    resolved = os.path.realpath(output_path)
    cwd = os.path.realpath(".")
    if not resolved.startswith(cwd + os.sep) and resolved != cwd:
        raise ValueError(
            f"Screenshot path '{output_path}' is outside the current "
            f"working directory (path traversal protection)",
        )
    # Ensure parent directory exists.
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return resolved


# Reject selectors containing obvious injection vectors. Playwright treats
# the argument purely as a CSS selector, so the risk is breaking out of the
# selector context into JS — block backticks, semicolons, ${...} etc.
def _validate_selector(selector: str) -> str:
    """Basic sanitisation of CSS selectors to prevent injection.

    Blocks selectors containing characters like backticks, semicolons,
    or shell metacharacters that could be exploited for injection.
    """
    if not selector:
        return selector
    # Block obvious injection vectors
    dangerous = {"`", ";", "${", "javascript:", "<script", "eval(", "alert("}
    if any(d in selector for d in dangerous):
        raise ValueError(
            f"CSS selector contains suspicious content: "
            f"'{selector[:50]}...'",
        )
    return selector


# ---------------------------------------------------------------------------
# Browser lifecycle (thread-safe, session-scoped reuse)
# ---------------------------------------------------------------------------

_browser_lock: asyncio.Lock | None = None
_browser_instance = None  # type: ignore[var-annotated]
_playwright_instance = None  # type: ignore[var-annotated]
_browser_loop: asyncio.AbstractEventLoop | None = None


def _get_lock() -> asyncio.Lock:
    global _browser_lock
    if _browser_lock is None:
        _browser_lock = asyncio.Lock()
    return _browser_lock


async def _get_browser():
    """Return a reusable Chromium browser instance, creating one if needed.

    Async-safe. The browser is lazily created and reused across calls
    within the same process. Each call gets its own BrowserContext for
    isolation.
    """
    global _browser_instance, _playwright_instance, _browser_loop
    async with _get_lock():
        if _browser_instance is not None:
            try:
                if _browser_instance.is_connected():
                    return _browser_instance
            except Exception:
                pass
            # Browser is dead — clean up and recreate.
            try:
                await _browser_instance.close()
            except Exception:
                pass
            _browser_instance = None

        if _playwright_instance is not None:
            try:
                await _playwright_instance.stop()
            except Exception:
                pass
            _playwright_instance = None

        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ImportError(_PLAYWRIGHT_INSTALL_HINT) from exc

        _playwright_instance = await async_playwright().start()
        _browser_instance = await _playwright_instance.chromium.launch(
            headless=True,
        )
        _browser_loop = asyncio.get_running_loop()
        return _browser_instance


def _cleanup_browser():
    """Best-effort atexit cleanup.

    Playwright objects are bound to the event loop that created them, so
    this only works when that loop is still open at interpreter exit;
    otherwise the coroutines are discarded (closing them avoids the
    "coroutine never awaited" warning) and the OS reaps the chromium
    child on process exit.
    """
    global _browser_instance, _playwright_instance, _browser_loop
    loop = _browser_loop
    for obj, method in (
        (_browser_instance, "close"),
        (_playwright_instance, "stop"),
    ):
        if obj is None:
            continue
        coro = getattr(obj, method)()
        try:
            if loop is not None and not loop.is_closed():
                loop.run_until_complete(coro)
            else:
                coro.close()
        except Exception:
            coro.close()
    _browser_instance = None
    _playwright_instance = None
    _browser_loop = None


atexit.register(_cleanup_browser)


def _is_safe_response_ip(url: str) -> bool:
    """Check if a response URL points to a public IP (anti-rebinding)."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # Hostname not an IP — resolve it
        try:
            infos = socket.getaddrinfo(hostname, parsed.port or 80)
        except socket.gaierror:
            return False
        for _, _, _, _, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
            ):
                return False
        return True
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    )


@asynccontextmanager
async def _launch_page(timeout: int = 30000, viewport: dict | None = None):
    """Yield a ready-to-use Playwright page with an isolated context.

    Reuses the shared browser instance but creates a fresh BrowserContext
    for each call (cookies, storage, etc. are not shared).
    """
    timeout = _clamp_timeout(timeout)
    browser = await _get_browser()
    ctx_kwargs: dict = {"user_agent": _USER_AGENT}
    if viewport:
        ctx_kwargs["viewport"] = viewport
    context = await browser.new_context(**ctx_kwargs)
    page = await context.new_page()
    page.set_default_timeout(timeout)

    # Route-level interception for SSRF anti-rebinding.
    async def _handle_route(route):
        # Anti-rebinding: block requests to private IPs even if the
        # initial DNS check passed (DNS could have changed).
        if not _is_safe_response_ip(route.request.url):
            await route.abort("addressunreachable")
            return

        await route.continue_()

    await page.route("**/*", _handle_route)

    try:
        yield page
    finally:
        try:
            await page.unroute("**/*")
        except Exception:
            pass
        try:
            await context.close()
        except Exception:
            pass


async def _navigate_and_wait(
    page,
    url: str,
    wait_selector: str = "",
    timeout: int = 30000,
):
    """Navigate to *url* and wait for content to be ready."""
    timeout = _clamp_timeout(timeout)
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    if wait_selector:
        await page.wait_for_selector(wait_selector, timeout=timeout)
    else:
        await page.wait_for_load_state("networkidle", timeout=timeout)


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


@tool(
    name="scrape_web",
    description=(
        "Scrape web page content with a Playwright headless browser "
        "(supports JS-rendered SPA pages). Returns the page's plain "
        "text. Optional wait selector, timeout, etc."
    ),
    permission=PermissionLevel.AUTO,
)
async def scrape_web(
    url: str,
    wait_selector: str = "",
    timeout: int = 30000,
) -> str:
    """Scrape a web page using Playwright headless browser.

    Args:
        url: The URL to scrape (http/https only).
        wait_selector: Optional CSS selector to wait for before extracting
        content.
        timeout: Max wait time in milliseconds (default 30s, max 120s).
    """
    try:
        _validate_url(url)
        wait_selector = _validate_selector(wait_selector)
    except ValueError as e:
        return f"Error: {e}"

    try:
        async with _launch_page(timeout=timeout) as page:
            await _navigate_and_wait(page, url, wait_selector, timeout)
            content = await page.inner_text("body")
            title = await page.title() or ""
    except ImportError as e:
        return str(e)
    except Exception as e:
        return f"Error: web scraping failed — {e}"

    # Clean up excessive whitespace
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    text = "\n".join(lines)

    max_len = 50_000
    if len(text) > max_len:
        text = text[:max_len] + "\n...(content truncated)"

    result = ""
    if title:
        result += f"# {title}\n\n"
    result += f"URL: {url}\n\n{text}"
    return result


@tool(
    name="scrape_web_html",
    description=(
        "Scrape a web page's HTML source with a Playwright headless "
        "browser (supports JS-rendered pages). Returns the fully "
        "rendered HTML."
    ),
    permission=PermissionLevel.AUTO,
)
async def scrape_web_html(
    url: str,
    wait_selector: str = "",
    timeout: int = 30000,
) -> str:
    """Scrape a web page and return the rendered HTML source.

    Args:
        url: The URL to scrape (http/https only).
        wait_selector: Optional CSS selector to wait for before extracting
        HTML.
        timeout: Max wait time in milliseconds (default 30s, max 120s).
    """
    try:
        _validate_url(url)
        wait_selector = _validate_selector(wait_selector)
    except ValueError as e:
        return f"Error: {e}"

    try:
        async with _launch_page(timeout=timeout) as page:
            await _navigate_and_wait(page, url, wait_selector, timeout)
            html = await page.content()
    except ImportError as e:
        return str(e)
    except Exception as e:
        return f"Error: web scraping failed — {e}"

    max_len = 80_000
    if len(html) > max_len:
        html = html[:max_len] + "\n...(HTML truncated)"
    return html


@tool(
    name="scrape_web_screenshot",
    description=(
        "Take a screenshot of a web page with a Playwright headless "
        "browser and save it as an image file. Useful for inspecting "
        "visual layout."
    ),
    permission=PermissionLevel.AUTO,
)
async def scrape_web_screenshot(
    url: str,
    output_path: str = "screenshot.png",
    full_page: bool = True,
    wait_selector: str = "",
    timeout: int = 30000,
) -> str:
    """Take a screenshot of a web page using Playwright headless browser.

    Args:
        url: The URL to screenshot (http/https only).
        output_path: Where to save the screenshot image (must be within
        current directory).
        full_page: If True, capture the full scrollable page.
        wait_selector: Optional CSS selector to wait for before capturing.
        timeout: Max wait time in milliseconds (default 30s, max 120s).
    """
    try:
        _validate_url(url)
        wait_selector = _validate_selector(wait_selector)
        output_path = _validate_screenshot_path(output_path)
    except ValueError as e:
        return f"Error: {e}"

    try:
        async with _launch_page(
            timeout=timeout,
            viewport={"width": 1280, "height": 720},
        ) as page:
            await _navigate_and_wait(page, url, wait_selector, timeout)
            await page.screenshot(path=output_path, full_page=full_page)
    except ImportError as e:
        return str(e)
    except Exception as e:
        return f"Error: screenshot failed — {e}"

    return (
        f"Screenshot saved to {output_path}. Reference it with @path "
        f"to send the image to a vision model for analysis."
    )
