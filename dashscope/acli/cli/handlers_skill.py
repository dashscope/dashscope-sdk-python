"""Skill and cron command handlers."""

from __future__ import annotations

from rich.console import Console

from dashscope.acli.agent import Agent
from dashscope.acli.config import Config

console = Console()


async def _handle_skill_package_command(sub: str, args: list[str]):
    """Handle /skill package lifecycle subcommands."""
    from dashscope.acli.skills import get_skill_manager

    manager = get_skill_manager()

    if sub == "list":
        from dashscope.acli.cli.mcp import _mcp_clients
        from dashscope.acli.skills import list_skills

        connected = set(_mcp_clients.keys())
        console.print(list_skills(connected))
        _print_skill_packages()
        return None

    if sub == "reload":
        manager.reload()
        console.print("[green]✓ Skill 包已重新加载[/green]")
        _print_skill_packages()
        return None

    if sub in ("enable", "disable"):
        if not args:
            console.print(f"[red]用法: /skill {sub} <package-name>[/red]")
            return None
        name = args[0]
        if sub == "enable":
            if manager.enable(name):
                console.print(f"[green]✓ 已启用 Skill 包: {name}[/green]")
            else:
                console.print(f"[red]未找到 Skill 包: {name}[/red]")
        else:
            if manager.disable(name):
                console.print(f"[green]✓ 已禁用 Skill 包: {name}[/green]")
            else:
                console.print(f"[red]未找到 Skill 包: {name}[/red]")
        return None

    def _strip_at(value: str) -> str:
        return value[1:] if value.startswith("@") else value

    if sub == "install":
        if not args:
            console.print(
                "[red]用法: /skill install <local-dir|git-url|name[@version]>[/red]"
            )
            return None
        source = _strip_at(args[0])
        try:
            name = manager.install(source)
            console.print(f"[green]✓ 已安装 Skill 包: {name}[/green]")
        except Exception as e:
            console.print(f"[red]安装失败: {e}[/red]")
        return None

    if sub == "link":
        if not args:
            console.print("[red]用法: /skill link <local-dir>[/red]")
            return None
        source_dir = _strip_at(args[0])
        try:
            name = manager.link(source_dir)
            console.print(f"[green]✓ 已链接 Skill 包: {name}[/green]")
        except Exception as e:
            console.print(f"[red]链接失败: {e}[/red]")
        return None

    if sub == "update":
        if args and args[0] == "--all":
            results = manager.update_all()
            ok = [n for n, err in results if err is None]
            failed = [(n, err) for n, err in results if err is not None]
            if ok:
                console.print(
                    f"[green]✓ 已更新 {len(ok)} 个 Skill 包: {', '.join(ok)}[/green]"
                )
            if failed:
                console.print("[red]更新失败:[/red]")
                for n, err in failed:
                    console.print(f"  - {n}: {err}")
            if not ok and not failed:
                console.print("[dim]没有需要更新的 Skill 包[/dim]")
            return None
        if not args:
            console.print(
                "[red]用法: /skill update <package-name> 或 /skill update --all[/red]"
            )
            return None
        name = args[0]
        try:
            manager.update(name)
            console.print(f"[green]✓ 已更新 Skill 包: {name}[/green]")
        except Exception as e:
            console.print(f"[red]更新失败: {e}[/red]")
        return None

    if sub == "uninstall":
        if not args:
            console.print("[red]用法: /skill uninstall <package-name>[/red]")
            return None
        name = args[0]
        if manager.uninstall(name):
            console.print(f"[green]✓ 已卸载 Skill 包: {name}[/green]")
        else:
            console.print(f"[red]未找到 Skill 包: {name}[/red]")
        return None

    if sub == "search":
        query = args[0] if args else ""
        results = manager.search(query)
        if not results:
            console.print("[dim]未找到匹配的 Skill 包[/dim]")
            return None
        console.print(f"\n[bold]搜索结果 ({len(results)}):[/bold]")
        for r in results:
            status = "[green]已安装[/green]" if r["installed"] else "[dim]未安装[/dim]"
            console.print(f"  {r['name']} v{r['version']} {status}")
            if r.get("description"):
                console.print(f"    {r['description']}")
        return None

    if sub == "publish":
        if not args:
            console.print("[red]用法: /skill publish <package-dir>[/red]")
            return None
        source_dir = _strip_at(args[0])
        try:
            url = manager.publish(source_dir)
            console.print(f"[green]✓ 已发布 Skill 包: {url}[/green]")
        except Exception as e:
            console.print(f"[red]发布失败: {e}[/red]")
        return None

    console.print(f"[red]未知的 /skill 子命令: {sub}[/red]")
    return None


def _print_skill_packages():
    """Display installed skill packages."""
    from dashscope.acli.skills import get_skill_manager

    manager = get_skill_manager()
    packages = manager.list()
    if not packages:
        console.print("[dim]未安装任何 Skill 包[/dim]")
        return

    console.print(f"\n[bold]已安装 Skill 包 ({len(packages)}):[/bold]")
    for pkg in packages:
        status = "[green]常驻[/green]" if pkg["always_active"] else "[dim]按需[/dim]"
        desc = f" — {pkg['description']}" if pkg.get("description") else ""
        console.print(f"  {pkg['name']} v{pkg['version']} {status}{desc}")


async def _handle_skill_command(
    cmd: str, config: Config, agent: Agent | None = None
) -> str | None:
    """Handle /skill command. Returns rendered prompt to feed to agent, or None if handled internally."""
    # Ensure skills are loaded every time
    from dashscope.acli.skills.base import load_skill_files

    load_skill_files()

    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2 or parts[1].strip() == "":
        # /skill — list all skills
        from dashscope.acli.cli.mcp import _mcp_clients
        from dashscope.acli.skills import list_skills

        connected = set(_mcp_clients.keys())
        console.print(list_skills(connected))
        _print_skill_packages()
        return None

    rest = parts[1].split()
    skill_name = rest[0]
    args = rest[1:]

    # Skill package management subcommands
    package_subs = {
        "add": "install",
        "remove": "uninstall",
    }
    if skill_name in package_subs:
        skill_name = package_subs[skill_name]
    if skill_name in (
        "list",
        "reload",
        "enable",
        "disable",
        "install",
        "link",
        "update",
        "uninstall",
        "search",
        "publish",
    ):
        return await _handle_skill_package_command(skill_name, args)

    from dashscope.acli.cli.mcp import _connect_mcp, _mcp_clients
    from dashscope.acli.skills import BUILTIN_SKILLS, render_skill

    skill = BUILTIN_SKILLS.get(skill_name)
    if not skill:
        console.print(f"[red]未知技能: {skill_name}[/red]")
        console.print("[dim]查看可用技能: /skill[/dim]")
        return None

    # Auto-connect required MCP service
    if skill.mcp_service and skill.mcp_service not in _mcp_clients:
        console.print(f"[dim]正在连接 MCP 服务: {skill.mcp_service}...[/dim]")
        err = await _connect_mcp(skill.mcp_service, config)
        if err:
            console.print(f"[red]连接失败: {err}[/red]")
            return None

    if not args:
        arg_hint = " ".join(f"<{a}>" for a in skill.arguments)
        console.print(f"[dim]用法: /skill {skill_name} {arg_hint}[/dim]")
        return None

    rendered = render_skill(skill, args)
    if not rendered:
        arg_hint = " ".join(f"<{a}>" for a in skill.arguments)
        console.print(f"[red]参数不足。用法: /skill {skill_name} {arg_hint}[/red]")
        return None

    if agent is not None:
        agent.note_skill_use(skill_name)
    return rendered


async def _handle_cron_command(cmd: str, config: Config, agent: Agent):
    """Handle /cron commands (async)."""
    import dashscope.acli.cli as _pkg

    if _pkg._scheduler is None:
        from dashscope.acli.scheduler import Scheduler

        _pkg._scheduler = Scheduler(config, agent)
        await _pkg._scheduler.load_and_start()

    _scheduler = _pkg._scheduler

    parts = cmd.strip().split(maxsplit=1)
    subcmd = parts[1].strip() if len(parts) > 1 else ""

    if not subcmd or subcmd == "list":
        _scheduler.print_jobs()
        return

    if subcmd.startswith("add"):
        args_str = subcmd[3:].strip()
        if not args_str:
            console.print("[dim]用法: /cron add every 5m skill weather 杭州[/dim]")
            console.print("[dim]      /cron add at 14:30 skill search AI新闻[/dim]")
            console.print(
                '[dim]      /cron add cron "0 9 * * 1-5" skill search 早报 condition "curl -sf http://health"[/dim]'
            )
            return
        from dashscope.acli.scheduler import parse_cron_add

        try:
            schedule, skills, condition, subagent = parse_cron_add(args_str)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return
        from dashscope.acli.skills import BUILTIN_SKILLS

        for inv in skills:
            if inv.name not in BUILTIN_SKILLS:
                console.print(f"[red]未知技能: {inv.name}[/red]")
                console.print("[dim]查看可用技能: /skill[/dim]")
                return
        job = _scheduler.add_job(schedule, skills, condition, subagent)
        console.print(f"[green]✓ 已创建定时任务:[/green] {job.id}")
        console.print(f"  调度: {schedule}")
        console.print(f"  技能: {', '.join(s.name for s in skills)}")
        if condition:
            console.print(f"  条件: {condition}")
        if schedule.kind == "at":
            console.print("  [dim]（一次性任务，执行后自动禁用）[/dim]")
        return

    if subcmd.startswith("remove"):
        job_id = subcmd[6:].strip()
        if not job_id:
            console.print("[red]用法: /cron remove <job-id>[/red]")
            return
        if _scheduler.remove_job(job_id):
            console.print(f"[green]✓ 已删除定时任务: {job_id}[/green]")
        else:
            console.print(f"[red]未找到定时任务: {job_id}[/red]")
        return

    if subcmd.startswith("pause"):
        job_id = subcmd[5:].strip()
        if not job_id:
            console.print("[red]用法: /cron pause <job-id>[/red]")
            return
        if _scheduler.pause_job(job_id):
            console.print(f"[green]✓ 已暂停定时任务: {job_id}[/green]")
        else:
            console.print(f"[red]未找到定时任务: {job_id}[/red]")
        return

    if subcmd.startswith("resume"):
        job_id = subcmd[6:].strip()
        if not job_id:
            console.print("[red]用法: /cron resume <job-id>[/red]")
            return
        if _scheduler.resume_job(job_id):
            console.print(f"[green]✓ 已恢复定时任务: {job_id}[/green]")
        else:
            console.print(f"[red]未找到定时任务: {job_id}[/red]")
        return

    console.print(f"[red]未知的 /cron 子命令: {subcmd}[/red]")
