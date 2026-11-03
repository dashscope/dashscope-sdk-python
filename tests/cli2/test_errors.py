# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

# pylint: disable=protected-access

"""Exit-code tiers, classification and the root error handler."""

import pytest
import typer
from typer.testing import CliRunner

from dashscope import _error_registry as registry
from dashscope import errors as sdk_errors
from dashscope.cli2 import errors as cli_errors
from dashscope.cli2.errors import (
    AUTH,
    EXIT_AUTH,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_SERVER,
    EXIT_USAGE,
    NETWORK,
    NOTFOUND,
    RATELIMIT,
    SERVER,
    USAGE,
    CliError,
    ErrorHandlingGroup,
    classify,
    exit_code_for,
    usage_error,
)
from tests.cli2.conftest import strip_ansi_codes


def _runner() -> CliRunner:
    """Separate stdout/stderr: the CLI promises clean JSON on stdout."""
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:  # click >= 8.2 always separates the streams
        return CliRunner()


class TestTierIsomorphism:
    def test_every_category_maps_to_a_tier(self):
        for category in cli_errors.CATEGORIES:
            assert exit_code_for(category) in (
                EXIT_SERVER,
                EXIT_USAGE,
                EXIT_AUTH,
                EXIT_NOT_FOUND,
            )

    def test_documented_tiers(self):
        assert (
            EXIT_OK,
            EXIT_SERVER,
            EXIT_USAGE,
            EXIT_AUTH,
            EXIT_NOT_FOUND,
        ) == (
            0,
            1,
            2,
            3,
            4,
        )
        assert exit_code_for(USAGE) == 2
        assert exit_code_for(AUTH) == 3
        assert exit_code_for(NOTFOUND) == 4
        for category in (SERVER, RATELIMIT, NETWORK):
            assert exit_code_for(category) == 1


class TestClassify:
    @pytest.mark.parametrize(
        "exc,category,exit_code",
        [
            (sdk_errors.AuthenticationError("nope"), AUTH, 3),
            (sdk_errors.PermissionDeniedError("nope"), AUTH, 3),
            (sdk_errors.RateLimitError("slow down"), RATELIMIT, 1),
            (sdk_errors.NotFoundError("gone"), NOTFOUND, 4),
            (sdk_errors.BadRequestError("bad"), USAGE, 2),
            (sdk_errors.RequestTooLargeError("big"), USAGE, 2),
            (sdk_errors.InternalServerError("boom"), SERVER, 1),
            (sdk_errors.ServiceUnavailableError("down"), SERVER, 1),
            (sdk_errors.GatewayTimeoutError("late"), SERVER, 1),
            (sdk_errors.APIStatusError("status"), SERVER, 1),
            (sdk_errors.APITimeoutError("timeout"), NETWORK, 1),
            (sdk_errors.APIConnectionError("conn"), NETWORK, 1),
            (sdk_errors.DashScopeError("sdk"), SERVER, 1),
            (sdk_errors.WsEventError("ws"), SERVER, 1),
            (ValueError("bad protocol"), USAGE, 2),
            (RuntimeError("unexpected"), SERVER, 1),
        ],
    )
    def test_categories(self, exc, category, exit_code):
        assert classify(exc)[0] == category
        assert exit_code_for(category) == exit_code

    def test_content_filter_inherits_bad_request_tier(self):
        """The catalog names ``BadRequestError``; V2 refines it by subclassing.

        ``ContentFilterError`` is therefore *not* a row of the derived
        table -- it has to reach ``usage`` through its parent, which is
        what keeps the refinement catalog-legal.
        """
        assert issubclass(
            sdk_errors.ContentFilterError,
            sdk_errors.BadRequestError,
        )
        assert classify(sdk_errors.ContentFilterError("blocked"))[0] == USAGE

    def test_timeout_is_network_via_its_catalog_code(self):
        # APITimeoutError subclasses APIConnectionError; both are
        # SdkErrors, categorised by their catalog code rather than by
        # class, so the subclassing is irrelevant to the tier.
        assert sdk_errors.APITimeoutError("t").code == "sdk.Timeout"
        assert isinstance(
            sdk_errors.APITimeoutError("t"),
            sdk_errors.APIConnectionError,
        )
        assert classify(sdk_errors.APITimeoutError("t"))[0] == NETWORK

    def test_exit_code_override_wins(self):
        assert CliError(USAGE, "x", exit_code=7).exit_code == 7

    def test_message_is_passed_through_verbatim(self):
        message = (
            "Model not exist [request_id=abc] <weird> & \"quoted\" 'single'"
        )
        exc = sdk_errors.NotFoundError(message, request_id="abc")
        diagnosis = classify(exc)
        assert diagnosis.category == NOTFOUND
        assert diagnosis.message == message
        assert diagnosis.request_id == "abc"

    def test_diagnosis_still_unpacks_as_a_tuple(self):
        category, message, solution, request_id = classify(
            sdk_errors.NotFoundError("gone", request_id="r"),
        )
        assert (category, message, solution, request_id) == (
            NOTFOUND,
            "gone",
            None,
            "r",
        )

    def test_cli_error_carries_its_own_category(self):
        diagnosis = classify(CliError(AUTH, "x"))
        assert (diagnosis.category, diagnosis.message) == (AUTH, "x")

    def test_cli_error_rejects_unknown_category(self):
        with pytest.raises(ValueError):
            _ = CliError("nope", "x")

    def test_usage_error_shorthand(self):
        exc = usage_error("bad flag")
        assert (exc.category, exc.message, exc.exit_code) == (
            USAGE,
            "bad flag",
            None,
        )


class TestCatalogDerivation:
    """The tables are read out of ``_error_registry.py``, not hand-listed.

    These are the drift guards: regenerate the catalog and one of them
    fails, pointing at the single human decision (a new status, or a new
    SDK code that needs a tier) rather than letting the CLI silently
    mis-slot it.
    """

    def test_status_table_covers_exactly_the_catalog_statuses(self):
        assert set(cli_errors._STATUS_CATEGORY) == {
            entry.status_code for entry in registry.PUBLIC_ERRORS
        }

    def test_every_status_maps_to_a_declared_category(self):
        for category in cli_errors._STATUS_CATEGORY.values():
            assert category in cli_errors.CATEGORIES

    def test_sdk_code_table_keys_are_real_catalog_names(self):
        assert set(cli_errors._SDK_CODE_CATEGORY) <= {
            entry.name for entry in registry.SDK_ERRORS
        }

    def test_public_table_names_every_catalog_class(self):
        assert {cls for cls, _ in cli_errors._public_table()} == (
            {
                getattr(sdk_errors, entry.error_code)
                for entry in registry.PUBLIC_ERRORS
            }
            | {sdk_errors.APIStatusError}
        )

    def test_public_table_has_one_row_per_class(self):
        rows = cli_errors._public_table()
        assert len({cls for cls, _ in rows}) == len(rows)

    def test_base_class_is_matched_last(self):
        # Every other row subclasses APIStatusError, so it must stay at
        # the end of the isinstance walk or it would swallow them all.
        assert cli_errors._public_table()[-1][0] is sdk_errors.APIStatusError

    @pytest.mark.parametrize(
        "entry",
        registry.PUBLIC_ERRORS,
        ids=lambda e: e.key,
    )
    def test_each_catalog_entry_lands_on_its_status_tier(self, entry):
        cls = getattr(sdk_errors, entry.error_code)
        assert classify(cls(entry.error_msg))[0] == (
            cli_errors._STATUS_CATEGORY[entry.status_code]
        )

    def test_undeclared_sdk_code_falls_to_server(self):
        # sdk.ws.* has no tier of its own: a protocol or state violation
        # is neither the caller's command line nor a credential problem.
        exc = sdk_errors.sdk_error("sdk.ws.UpstreamQueueFull")
        assert classify(exc)[0] == SERVER

    def test_sdk_invalid_api_key_is_auth_not_server(self):
        # A generic SdkError used to reach the DashScopeError fallback and
        # exit 1; the catalog name says plainly what this one is.
        assert classify(sdk_errors.sdk_error("sdk.InvalidApiKey"))[0] == AUTH
        assert exit_code_for(AUTH) == 3

    def test_local_io_failure_is_usage(self):
        # The entry's own solution reads "verify the file path exists and
        # is readable" -- the caller's mistake, so tier 2.
        exc = sdk_errors.sdk_error("sdk.IoError", {"detail": "no such file"})
        assert classify(exc)[0] == USAGE


class TestCatalogMessageFallback:
    def test_server_wording_always_wins(self):
        assert classify(sdk_errors.NotFoundError("gone")).message == "gone"

    def test_empty_message_falls_back_to_the_catalog(self):
        assert classify(sdk_errors.NotFoundError("")).message == (
            registry.RESOURCE_NOT_FOUND.error_msg
        )

    def test_placeholder_is_left_verbatim(self):
        # Nothing here knows the resource name; stripping the placeholder
        # would hide the field the catalog declares.
        assert "{resource}" in classify(sdk_errors.NotFoundError("")).message

    def test_ambiguous_entry_is_not_guessed(self):
        # Five catalog rows name BadRequestError with different wording,
        # so naming the class does not identify the entry.
        assert classify(sdk_errors.BadRequestError("")).message == ""

    def test_sdk_error_falls_back_to_its_own_entry(self):
        # _CatalogSdkError always formats its message from the entry, so
        # the fallback only fires on a bare SdkError built without one.
        exc = sdk_errors.SdkError("", code="sdk.ConnectionError")
        assert classify(exc).message == registry.SDK_CONNECTION_ERROR.message


class TestCatalogSolution:
    def test_sdk_errors_carry_their_entries_solution(self):
        exc = sdk_errors.APIConnectionError(detail="connection refused")
        assert classify(exc).solution == (
            registry.SDK_CONNECTION_ERROR.solution
        )

    def test_gateway_errors_have_no_solution_to_render(self):
        # PUBLIC_ERRORS carries no solution field: the gateway's own
        # message is the whole of what the CLI may say.
        assert classify(sdk_errors.NotFoundError("gone")).solution is None

    def test_solution_is_rendered_on_its_own_line(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")

        def raise_it():
            raise sdk_errors.APIConnectionError(detail="connection refused")

        result = _runner().invoke(_app(raise_it), ["run"])
        assert result.exit_code == 1
        lines = strip_ansi_codes(result.stderr).splitlines()
        assert (
            lines[0]
            == "Error [network]: Connection failed: connection refused."
        )
        assert lines[1].startswith("solution: Check network connectivity")
        # The service's wording and the registry's stay separable.
        assert "solution:" not in lines[0]


def _app(body):
    """An app with two commands.

    Typer collapses a one-command app into a plain ``TyperCommand`` and
    ignores ``cls=``, so a single-command fixture would silently test
    nothing. The real root app always has several subcommands.
    """
    app = typer.Typer(cls=ErrorHandlingGroup)

    @app.command()
    def run():
        body()

    @app.command()
    def other():
        pass

    return app


class TestErrorHandlingGroup:
    def test_success_exits_zero(self):
        result = _runner().invoke(_app(lambda: None), ["run"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (sdk_errors.AuthenticationError("bad key"), 3),
            (sdk_errors.NotFoundError("missing"), 4),
            (sdk_errors.RateLimitError("slow"), 1),
            (sdk_errors.InternalServerError("boom"), 1),
            (CliError(USAGE, "nope"), 2),
            (ValueError("bad protocol"), 2),
            (RuntimeError("unexpected"), 1),
        ],
    )
    def test_exit_code_by_category(self, exc, expected):
        def raise_it():
            raise exc

        result = _runner().invoke(_app(raise_it), ["run"])
        assert result.exit_code == expected

    def test_stderr_line_format_and_clean_stdout(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")

        def raise_it():
            raise sdk_errors.NotFoundError(
                "job xyz does not exist",
                request_id="req-1",
            )

        result = _runner().invoke(_app(raise_it), ["run"])
        assert result.exit_code == 4
        assert result.stdout == ""
        stderr = strip_ansi_codes(result.stderr)
        assert "Error [notfound]: job xyz does not exist" in stderr
        assert "request_id: req-1" in stderr

    def test_server_message_is_not_reworded(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        raw = "Throttling.RateQuota: Requests rate limit exceeded, please try again."

        def raise_it():
            raise sdk_errors.RateLimitError(raw)

        result = _runner().invoke(_app(raise_it), ["run"])
        assert raw in strip_ansi_codes(result.stderr)

    def test_long_message_is_not_wrapped_to_terminal_width(self, monkeypatch):
        """Rich wraps at 80 columns by default; a server message must not.

        Re-flowing the text would break anyone grepping stderr and would
        hand a support ticket a message with newlines the service never
        sent.
        """
        monkeypatch.setenv("NO_COLOR", "1")
        raw = (
            "Requests rate limit exceeded for model qwen-plus in "
            "workspace ws-1234567890, please try again later. " * 3
        ).strip()
        assert len(raw) > 80

        def raise_it():
            raise sdk_errors.RateLimitError(raw)

        result = _runner().invoke(_app(raise_it), ["run"])
        stderr = strip_ansi_codes(result.stderr)
        assert f"Error [ratelimit]: {raw}" in stderr

    def test_brackets_in_message_survive_markup_escaping(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")

        def raise_it():
            raise sdk_errors.BadRequestError("invalid [red] param")

        result = _runner().invoke(_app(raise_it), ["run"])
        assert "invalid [red] param" in strip_ansi_codes(result.stderr)

    def test_verbose_adds_traceback(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        app = typer.Typer(cls=ErrorHandlingGroup)

        @app.callback()
        def root(verbose: bool = False):
            pass

        @app.command()
        def run():
            raise RuntimeError("kaboom")

        quiet = _runner().invoke(app, ["run"])
        assert "Traceback" not in strip_ansi_codes(quiet.stderr)

        # The callback never ran to completion, so ctx.obj is unset and
        # the handler falls back to ctx.params -- pin that path.
        loud = _runner().invoke(app, ["--verbose", "run"])
        assert loud.exit_code == 1
        assert "Traceback" in strip_ansi_codes(loud.stderr)

    def test_typer_exit_is_not_converted(self):
        def abort():
            raise typer.Exit(9)

        result = _runner().invoke(_app(abort), ["run"])
        assert result.exit_code == 9

    def test_click_usage_error_passes_through(self):
        app = typer.Typer(cls=ErrorHandlingGroup)

        @app.command()
        def run(name: str):
            typer.echo(name)

        @app.command()
        def other():
            pass

        result = _runner().invoke(app, ["run"])
        assert result.exit_code == 2
        assert "Missing argument" in strip_ansi_codes(result.stderr)

    def test_keyboard_interrupt_is_not_an_error_line(self):
        def interrupt():
            raise KeyboardInterrupt

        result = _runner().invoke(_app(interrupt), ["run"])
        # click turns an interrupt into its own exit code; the point is
        # that the handler did not claim it as a CLI error line.
        assert result.exit_code != 0
        assert "Error [" not in strip_ansi_codes(result.stderr)

    def test_context_object_is_closed_on_failure(self):
        closed = []

        class Obj:
            verbose = False

            def close(self):
                closed.append(True)

        app = typer.Typer(cls=ErrorHandlingGroup)

        @app.callback()
        def root(ctx: typer.Context):
            ctx.obj = Obj()

        @app.command()
        def run():
            raise RuntimeError("kaboom")

        @app.command()
        def other():
            pass

        result = _runner().invoke(app, ["run"])
        assert result.exit_code == 1
        assert closed == [True]

    def test_cls_is_actually_respected(self):
        """The handler only works if typer really used our group class.

        A one-command app collapses to ``TyperCommand`` and drops
        ``cls=``, which is why this builds two commands.
        """
        app = _app(lambda: None)

        built = typer.main.get_command(app)
        assert isinstance(built, ErrorHandlingGroup)

    def test_single_command_app_would_not_be_a_group(self):
        """Pin the collapse behaviour the fixture above works around."""
        app = typer.Typer(cls=ErrorHandlingGroup)

        @app.command()
        def solo():
            pass

        assert not isinstance(typer.main.get_command(app), ErrorHandlingGroup)
