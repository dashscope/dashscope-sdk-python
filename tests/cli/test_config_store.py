# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The config.toml line editor behind ``dashscope config``."""

import os
import stat

import pytest

from dashscope.cli import config_store
from dashscope.cli.config_store import (
    CLI_SECTION,
    effective_values,
    get_value,
    known_keys,
    load_cli_section,
    load_raw,
    mask,
    parse_toml_value,
    render_toml_value,
    restrict_permissions,
    set_value,
)
from dashscope.cli.errors import CliError


@pytest.fixture
def config(clean_env):
    return clean_env["config"]


@pytest.fixture
def permissive_umask():
    """The ``umask 022`` default a world-readable temp file needs."""
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


class TestValueParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("30", 30),
            ("1.5", 1.5),
            ("true", True),
            ("false", False),
            ('"quoted"', "quoted"),
            ("qwen-plus", "qwen-plus"),
            ("[1, 2]", [1, 2]),
            # JSON object syntax is not TOML (inline tables use ``=``), so
            # this stays a string rather than silently becoming a dict.
            ('{"a": 1}', '{"a": 1}'),
        ],
    )
    def test_parse(self, raw, expected):
        assert parse_toml_value(raw) == expected

    @pytest.mark.parametrize(
        "value",
        ["plain", "with space", 'quote"inside', "back\\slash", "你好"],
    )
    def test_roundtrip(self, value):
        assert parse_toml_value(render_toml_value(value)) == value

    def test_numbers_roundtrip(self):
        assert parse_toml_value(render_toml_value(30)) == 30
        assert parse_toml_value(render_toml_value(1.5)) == 1.5
        assert parse_toml_value(render_toml_value(True)) is True


class TestKeyRouting:
    def test_sdk_keys_are_top_level(self):
        assert "api_key" in known_keys()
        assert "protocol" in known_keys()

    def test_cli_keys_routed_to_the_cli_section(self, config):
        set_value("output", "json", path=config)
        assert load_cli_section(config) == {"output": "json"}
        assert load_raw(config)[CLI_SECTION]["output"] == "json"

    def test_dotted_spelling_is_the_same_key(self, config):
        set_value("cli.default_model", "qwen-plus", path=config)
        set_value("default_model", "qwen-max", path=config)
        assert load_cli_section(config) == {"default_model": "qwen-max"}

    def test_unknown_key_is_a_usage_error(self, config):
        with pytest.raises(CliError) as info:
            set_value("nope", "x", path=config)
        assert info.value.category == "usage"
        assert "known keys" in info.value.message

    def test_unknown_section_is_a_usage_error(self, config):
        with pytest.raises(CliError) as info:
            set_value("other.output", "json", path=config)
        assert "unknown config section" in info.value.message


class TestValidation:
    @pytest.mark.parametrize(
        "key,value",
        [
            ("protocol", "bogus"),
            ("region", "mars-1"),
            ("output", "csv"),
            ("timeout", "0"),
            ("timeout", "-1"),
            ("timeout", "abc"),
            ("max_retries", "-1"),
            ("http_keepalive_expiry", "0"),
            ("http_max_connections", "0"),
            ("http_max_connections", "-1"),
            ("api_key", ""),
        ],
    )
    def test_rejected(self, key, value, config):
        with pytest.raises(CliError) as info:
            set_value(key, value, path=config)
        assert info.value.category == "usage"

    def test_the_pool_floor_matches_the_sdk_setters(self, config):
        """A config file is read on every later invocation, so writing a
        value `Settings` refuses would break the CLI permanently."""
        assert set_value("http_max_connections", "1", path=config) == (
            "http_max_connections",
            1,
        )
        assert (
            set_value("http_max_keepalive_connections", "0", path=config)[1]
            == 0
        )
        assert set_value("max_retries", "0", path=config)[1] == 0

    def test_timeout_becomes_a_number(self, config):
        assert set_value("timeout", "30", path=config) == ("timeout", 30)
        assert load_raw(config)["timeout"] == 30

    def test_valid_protocol_and_region(self, config):
        set_value("protocol", "openai", path=config)
        set_value("region", "cn-beijing", path=config)
        assert load_raw(config)["protocol"] == "openai"


class TestPreservation:
    def test_comments_and_unknown_keys_survive(self, config):
        config.write_text(
            "# my note\n"
            "some_future_key = 1\n"
            'api_key = "sk-old"\n'
            "\n"
            "# trailing note\n",
            encoding="utf-8",
        )
        set_value("api_key", "sk-new", path=config)
        text = config.read_text(encoding="utf-8")
        assert "# my note" in text
        assert "some_future_key = 1" in text
        assert "# trailing note" in text
        assert 'api_key = "sk-new"' in text
        assert "sk-old" not in text

    def test_replace_does_not_duplicate(self, config):
        set_value("workspace", "ws-1", path=config)
        set_value("workspace", "ws-2", path=config)
        set_value("workspace", "ws-3", path=config)
        text = config.read_text(encoding="utf-8")
        assert text.count("workspace") == 1
        assert load_raw(config)["workspace"] == "ws-3"

    def test_inline_comment_on_an_existing_line_is_replaced_whole(
        self,
        config,
    ):
        config.write_text(
            'api_key = "sk-old"  # keep out of git\n',
            encoding="utf-8",
        )
        set_value("api_key", "sk-new", path=config)
        assert load_raw(config)["api_key"] == "sk-new"

    def test_cli_key_lands_inside_its_own_section(self, config):
        config.write_text(
            'api_key = "sk-1"\n\n[cli]\ndefault_model = "qwen-plus"\n',
            encoding="utf-8",
        )
        set_value("output", "json", path=config)
        text = config.read_text(encoding="utf-8")
        assert text.index("[cli]") < text.index("output")
        assert load_cli_section(config) == {
            "default_model": "qwen-plus",
            "output": "json",
        }

    def test_new_cli_section_is_appended(self, config):
        config.write_text('api_key = "sk-1"\n', encoding="utf-8")
        set_value("output", "yaml", path=config)
        text = config.read_text(encoding="utf-8")
        assert text.rstrip().endswith('output = "yaml"')
        assert "[cli]" in text

    def test_top_level_key_precedes_existing_tables(self, config):
        """TOML would otherwise attribute the key to the last table."""
        config.write_text(
            '[cli]\ndefault_model = "qwen-plus"\n',
            encoding="utf-8",
        )
        set_value("api_key", "sk-1", path=config)
        data = load_raw(config)
        assert data["api_key"] == "sk-1"
        assert "api_key" not in data[CLI_SECTION]

    def test_missing_file_is_created(self, config):
        assert not config.exists()
        set_value("api_key", "sk-1", path=config)
        assert load_raw(config) == {"api_key": "sk-1"}

    def test_missing_directory_is_created(self, tmp_path):
        target = tmp_path / "nested" / "config.toml"
        set_value("api_key", "sk-1", path=target)
        assert load_raw(target) == {"api_key": "sk-1"}


class TestTomlSemantics:
    """The editor reads the file as TOML, not as text that looks like it."""

    def test_a_table_header_may_carry_a_comment(self, config):
        config.write_text(
            '[cli] # my settings\noutput = "json"\n',
            encoding="utf-8",
        )
        set_value("workspace", "ws-1", path=config)
        data = load_raw(config)
        # Top level is where the SDK reads it; inside [cli] it is ignored.
        assert data["workspace"] == "ws-1"
        assert data["cli"] == {"output": "json"}

    def test_a_key_inside_a_multiline_string_is_left_alone(self, config):
        config.write_text(
            'prompt = """\n[cli]\noutput = "yaml"\n"""\n',
            encoding="utf-8",
        )
        set_value("output", "json", path=config)
        data = load_raw(config)
        assert data["prompt"] == '[cli]\noutput = "yaml"\n'
        assert data["cli"]["output"] == "json"

    def test_a_literal_multiline_string_is_left_alone(self, config):
        config.write_text(
            "prompt = '''\napi_key = \"not mine\"\n'''\n",
            encoding="utf-8",
        )
        set_value("api_key", "sk-1", path=config)
        data = load_raw(config)
        assert data["prompt"] == 'api_key = "not mine"\n'
        assert data["api_key"] == "sk-1"

    def test_a_hash_inside_a_string_is_not_a_comment(self, config):
        config.write_text('base_url = "http://host#frag"\n', encoding="utf-8")
        set_value("base_url", "http://other#2", path=config)
        assert load_raw(config)["base_url"] == "http://other#2"

    def test_a_commented_out_key_is_not_revived(self, config):
        config.write_text('# api_key = "old"\n', encoding="utf-8")
        set_value("api_key", "sk-new", path=config)
        text = config.read_text(encoding="utf-8")
        assert text.startswith('# api_key = "old"')
        assert load_raw(config)["api_key"] == "sk-new"

    def test_a_quoted_table_header_is_recognised(self, config):
        config.write_text('[ "cli" ]\n', encoding="utf-8")
        set_value("output", "json", path=config)
        assert load_raw(config)["cli"]["output"] == "json"

    def test_a_top_level_key_stays_out_of_an_array_of_tables(self, config):
        config.write_text('[[models]]\nid = "a"\n', encoding="utf-8")
        set_value("workspace", "ws-1", path=config)
        assert load_raw(config)["workspace"] == "ws-1"

    def test_a_comment_after_a_header_does_not_leak_into_the_name(
        self,
        config,
    ):
        config.write_text("[cli] # see [docs]\n", encoding="utf-8")
        set_value("output", "json", path=config)
        set_value("workspace", "ws-1", path=config)
        data = load_raw(config)
        assert data["cli"]["output"] == "json"
        assert data["workspace"] == "ws-1"


class TestReadBack:
    """An edit that would not be read back must not report success."""

    def test_a_value_written_into_the_wrong_table_is_refused(
        self,
        config,
        monkeypatch,
    ):
        # Stands in for any future scanner bug: force the key into [cli]
        # and check the command fails instead of claiming success.
        monkeypatch.setattr(
            config_store,
            "_insert_key",
            lambda lines, section, rendered: lines.extend(
                ["[cli]", rendered],
            ),
        )
        with pytest.raises(CliError) as info:
            set_value("workspace", "ws-1", path=config)
        assert info.value.category == "usage"
        assert "did not take effect" in info.value.message
        assert not config.exists()


class TestSafety:
    def test_refuses_to_write_invalid_toml_and_leaves_the_original(
        self,
        config,
    ):
        # A dotted key already defines cli.output; appending the same key
        # under a [cli] header would be a duplicate, which is invalid TOML.
        original = 'cli.output = "json"\n'
        config.write_text(original, encoding="utf-8")
        with pytest.raises(CliError) as info:
            set_value("output", "yaml", path=config)
        assert info.value.category == "usage"
        assert "invalid TOML" in info.value.message
        assert config.read_text(encoding="utf-8") == original
        assert not list(config.parent.glob("*.tmp"))

    def test_existing_invalid_file_is_reported_not_overwritten_silently(
        self,
        config,
    ):
        config.write_text("api_key = \n", encoding="utf-8")
        with pytest.raises(CliError) as info:
            load_raw(config)
        assert "invalid TOML" in info.value.message

    def test_restrict_permissions(self, config):
        set_value("api_key", "sk-1", path=config)
        restrict_permissions(config)
        mode = stat.S_IMODE(os.stat(config).st_mode)
        assert mode == 0o600

    def test_file_mode_is_preserved_across_rewrites(self, config):
        config.write_text('api_key = "sk-1"\n', encoding="utf-8")
        os.chmod(config, 0o600)
        set_value("api_key", "sk-2", path=config)
        assert stat.S_IMODE(os.stat(config).st_mode) == 0o600

    def test_a_new_file_is_owner_only_under_a_permissive_umask(
        self,
        config,
        permissive_umask,
    ):
        assert not config.exists()
        set_value("api_key", "sk-secret", path=config)
        assert stat.S_IMODE(os.stat(config).st_mode) == 0o600

    def test_the_temp_file_already_holds_the_secret_at_0600(
        self,
        config,
        permissive_umask,
        monkeypatch,
    ):
        """There must be no window where the key is world-readable."""
        modes = []
        replace = os.replace

        def spy(source, destination):
            modes.append(stat.S_IMODE(os.stat(source).st_mode))
            return replace(source, destination)

        monkeypatch.setattr(os, "replace", spy)
        set_value("api_key", "sk-secret", path=config)
        assert modes == [0o600]

    def test_group_and_other_bits_are_dropped_on_rewrite(
        self,
        config,
        permissive_umask,
    ):
        config.write_text('api_key = "sk-1"\n', encoding="utf-8")
        os.chmod(config, 0o644)
        set_value("api_key", "sk-2", path=config)
        assert stat.S_IMODE(os.stat(config).st_mode) == 0o600
        assert load_raw(config)["api_key"] == "sk-2"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
    def test_a_planted_tmp_symlink_is_not_followed(self, config):
        """The old fixed name let anyone redirect the write."""
        victim = config.with_name("victim")
        victim.write_text("precious", encoding="utf-8")
        config.with_name(config.name + ".tmp").symlink_to(victim)
        set_value("api_key", "sk-1", path=config)
        assert victim.read_text(encoding="utf-8") == "precious"
        assert load_raw(config) == {"api_key": "sk-1"}

    def test_restrict_permissions_tolerates_a_missing_file(self, tmp_path):
        restrict_permissions(tmp_path / "absent.toml")


class TestMasking:
    def test_long_secret(self):
        assert mask("sk-1234abcd5678") == "sk-1*********78"

    def test_short_secret_is_fully_hidden(self):
        assert mask("abc123") == "***"
        assert mask("") == "***"

    def test_keeps_only_the_ends(self):
        masked = mask("sk-abcdefghij")
        assert masked.startswith("sk-a")
        assert masked.endswith("ij")
        assert "bcdefgh" not in masked


class TestEffectiveValues:
    def test_file_value(self, config):
        set_value("workspace", "ws-1", path=config)
        rows = {row["key"]: row for row in effective_values(path=config)}
        assert rows["workspace"] == {
            "key": "workspace",
            "value": "ws-1",
            "source": "config file",
        }

    def test_env_beats_file(self, config, monkeypatch):
        set_value("workspace", "ws-file", path=config)
        monkeypatch.setenv("DASHSCOPE_WORKSPACE", "ws-env")
        rows = {row["key"]: row for row in effective_values(path=config)}
        assert rows["workspace"]["value"] == "ws-env"
        assert rows["workspace"]["source"] == "env DASHSCOPE_WORKSPACE"

    def test_unset_reports_default(self, config):
        rows = {row["key"]: row for row in effective_values(path=config)}
        assert rows["region"]["source"] == "default"
        assert rows["region"]["value"] is None

    def test_api_key_is_masked_from_every_source(self, config, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-envkey123456")
        rows = {row["key"]: row for row in effective_values(path=config)}
        assert rows["api_key"]["value"] == mask("sk-envkey123456")
        assert rows["api_key"]["value"].startswith("sk-e")
        assert "envkey" not in rows["api_key"]["value"]

    def test_cli_keys_are_included(self, config):
        set_value("default_model", "qwen-plus", path=config)
        rows = {row["key"]: row for row in effective_values(path=config)}
        assert rows["cli.default_model"]["value"] == "qwen-plus"
        assert rows["cli.output"]["source"] == "default"

    def test_every_known_key_appears(self, config):
        keys = {row["key"] for row in effective_values(path=config)}
        for key in known_keys():
            expected = (
                f"{CLI_SECTION}.{key}"
                if key in ("output", "default_model")
                else key
            )
            assert expected in keys


class TestGetValue:
    def test_roundtrip(self, config):
        set_value("api_key", "sk-1", path=config)
        assert get_value("api_key", path=config) == "sk-1"

    def test_cli_key(self, config):
        set_value("output", "json", path=config)
        assert get_value("output", path=config) == "json"

    def test_unset_is_none(self, config):
        assert get_value("workspace", path=config) is None
