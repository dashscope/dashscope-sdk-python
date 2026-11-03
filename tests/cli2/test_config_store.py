# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The config.toml line editor behind ``dashscope config``."""

import os
import stat

import pytest

from dashscope.cli2.config_store import (
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
from dashscope.cli2.errors import CliError


@pytest.fixture
def config(clean_env):
    return clean_env["config"]


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
            ("api_key", ""),
        ],
    )
    def test_rejected(self, key, value, config):
        with pytest.raises(CliError) as info:
            set_value(key, value, path=config)
        assert info.value.category == "usage"

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
        assert not (config.parent / (config.name + ".tmp")).exists()

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
