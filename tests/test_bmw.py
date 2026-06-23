"""Tests for the BMW LLM API gateway integration — fully offline (no network/SSL).

The BMW intranet endpoints are unreachable from anywhere but a BMW PC, so every test here
stubs the token fetch, CA-cert download, and httpx client. Live end-to-end coverage has to
run on a BMW PC (see the plan / README).
"""

import httpx
import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel as FakeModel  # aliased so pytest won't collect it

from cad_gen import bmw
from cad_gen.bmw import BMWConfigError, _TokenCache, require_bmw_credentials, resolve_model
from cad_gen.models import RunConfig, _replace_legacy_model

# Every env var the BMW path or the toggle reads — cleared before each test for hermeticity.
_BMW_ENV_VARS = (
    "CAD_GEN_BMW",
    "LLM_API_PROD_KEY",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "LLM_ACCESS_TOKEN",
    "LLM_ACCESS_TOKEN_EXP",
    "BMW_LLM_BASE_URL",
    "BMW_CA_CERT_PATH",
    "BMW_CA_CERT_URL",
    "BMW_AUTH_ENDPOINT",
    "CAD_GEN_MODEL",
    "CAD_GEN_CRITIC_MODEL",
    "CAD_GEN_VIEW_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_bmw(monkeypatch):
    for var in _BMW_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Reset the per-process provider/token caches so model-building tests don't leak.
    monkeypatch.setattr(bmw, "_provider", None)
    monkeypatch.setattr(bmw, "_token_cache", _TokenCache())


def _stub_network(monkeypatch):
    """Let _shared_provider build without touching the BMW intranet or a real CA cert."""
    monkeypatch.setattr(bmw, "download_ca_cert", lambda *a, **k: "dummy.pem")
    monkeypatch.setattr(bmw, "_build_http_client", lambda cert_path, api_key: httpx.AsyncClient())
    monkeypatch.setattr(bmw, "fetch_access_token", lambda *a, **k: ("tok-fetched", 3600))


class TestResolveModel:
    def test_passes_through_non_bmw_string(self):
        assert resolve_model("google:gemini-3.5-flash") == "google:gemini-3.5-flash"

    def test_passes_through_model_object(self):
        m = FakeModel()
        assert resolve_model(m) is m

    def test_bmw_prefix_builds_openai_chat_model(self, monkeypatch):
        _stub_network(monkeypatch)
        monkeypatch.setenv("LLM_API_PROD_KEY", "key-abc")
        monkeypatch.setenv("LLM_ACCESS_TOKEN", "tok-direct")
        result = resolve_model("bmw:openai/gpt-5-mini")
        assert isinstance(result, OpenAIChatModel)
        assert result.model_name == "openai/gpt-5-mini"  # the bmw: prefix is stripped

    def test_base_url_defaults_to_eu_prod_v1(self, monkeypatch):
        _stub_network(monkeypatch)
        monkeypatch.setenv("LLM_API_PROD_KEY", "key-abc")
        monkeypatch.setenv("LLM_ACCESS_TOKEN", "tok-direct")
        resolve_model("bmw:openai/gpt-5-mini")
        assert str(bmw._shared_provider().base_url).rstrip("/") == (
            "https://api.gcp.cloud.bmw/llmapi/v1"
        )

    def test_base_url_override(self, monkeypatch):
        _stub_network(monkeypatch)
        monkeypatch.setenv("LLM_API_PROD_KEY", "key-abc")
        monkeypatch.setenv("LLM_ACCESS_TOKEN", "tok-direct")
        monkeypatch.setenv("BMW_LLM_BASE_URL", "https://api.int.gcp.cloud.bmw/llmapi")
        resolve_model("bmw:anthropic/claude-sonnet-4-5")
        assert str(bmw._shared_provider().base_url).rstrip("/") == (
            "https://api.int.gcp.cloud.bmw/llmapi/v1"
        )

    def test_missing_api_key_raises(self, monkeypatch):
        _stub_network(monkeypatch)
        monkeypatch.setenv("LLM_ACCESS_TOKEN", "tok-direct")  # have a token but no x-apikey
        with pytest.raises(BMWConfigError):
            resolve_model("bmw:openai/gpt-5-mini")


class TestLegacyFilterGuard:
    @pytest.mark.parametrize(
        "value",
        ["bmw:openai/gpt-4o", "bmw:anthropic/claude-sonnet-4-5", "bmw:openai/gpt-5-mini"],
    )
    def test_bmw_prefix_survives_legacy_filter(self, value):
        # Without the guard, the gpt-/openai legacy rewrite would clobber these.
        assert _replace_legacy_model(value, "google:gemini-3.5-flash") == value


class TestBmwToggle:
    def test_toggle_routes_all_agents_to_bmw(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_BMW", "1")
        cfg = RunConfig()
        assert cfg.model == "bmw:openai/gpt-5-mini"
        assert cfg.critic_model == "bmw:openai/gpt-5-mini"
        assert cfg.view_model == "bmw:openai/gpt-5-mini"

    def test_toggle_keeps_explicit_bmw_override(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_BMW", "1")
        monkeypatch.setenv("CAD_GEN_CRITIC_MODEL", "bmw:anthropic/claude-sonnet-4-5")
        cfg = RunConfig()
        assert cfg.model == "bmw:openai/gpt-5-mini"
        assert cfg.critic_model == "bmw:anthropic/claude-sonnet-4-5"  # per-agent override kept

    def test_toggle_coerces_non_bmw_model_string(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_BMW", "1")
        monkeypatch.setenv("CAD_GEN_MODEL", "google:gemini-3.5-flash")
        assert RunConfig().model == "bmw:openai/gpt-5-mini"

    def test_toggle_off_keeps_gemini_defaults(self):
        cfg = RunConfig()
        assert cfg.model == "google:gemini-3.5-flash"
        assert cfg.critic_model == "google:gemini-3.5-flash"

    def test_bmw_prefix_without_toggle(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_MODEL", "bmw:openai/gpt-4o")
        assert RunConfig().model == "bmw:openai/gpt-4o"


class TestCredentials:
    def test_missing_reports_api_key_and_token_source(self):
        missing = require_bmw_credentials()
        assert "LLM_API_PROD_KEY" in missing
        assert any("CLIENT_ID" in m for m in missing)

    def test_present_with_client_creds(self, monkeypatch):
        monkeypatch.setenv("LLM_API_PROD_KEY", "k")
        monkeypatch.setenv("CLIENT_ID", "c")
        monkeypatch.setenv("CLIENT_SECRET", "s")
        assert require_bmw_credentials() == []

    def test_present_with_direct_token(self, monkeypatch):
        monkeypatch.setenv("LLM_API_PROD_KEY", "k")
        monkeypatch.setenv("LLM_ACCESS_TOKEN", "tok")
        assert require_bmw_credentials() == []


class TestTokenCache:
    def test_uses_unexpired_env_token_without_fetch(self, monkeypatch):
        monkeypatch.setenv("LLM_ACCESS_TOKEN", "env-token")  # no expiry => assumed valid

        def _boom(*a, **k):
            raise AssertionError("fetch_access_token should not be called")

        monkeypatch.setattr(bmw, "fetch_access_token", _boom)
        assert _TokenCache().get_token() == "env-token"

    def test_fetches_when_no_env_token(self, monkeypatch):
        monkeypatch.setenv("CLIENT_ID", "c")
        monkeypatch.setenv("CLIENT_SECRET", "s")
        monkeypatch.setattr(bmw, "fetch_access_token", lambda *a, **k: ("fetched", 3600))
        assert _TokenCache().get_token() == "fetched"

    def test_expired_env_token_triggers_fetch(self, monkeypatch):
        monkeypatch.setenv("LLM_ACCESS_TOKEN", "old")
        monkeypatch.setenv("LLM_ACCESS_TOKEN_EXP", "2000-01-01T00:00:00")
        monkeypatch.setenv("CLIENT_ID", "c")
        monkeypatch.setenv("CLIENT_SECRET", "s")
        monkeypatch.setattr(bmw, "fetch_access_token", lambda *a, **k: ("fresh", 3600))
        assert _TokenCache().get_token() == "fresh"

    def test_no_credentials_raises(self):
        with pytest.raises(BMWConfigError):
            _TokenCache().get_token()
