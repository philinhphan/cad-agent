"""BMW LLM API gateway provider for cad-gen.

On BMW PCs the public OpenAI/Google/Anthropic endpoints are blocked — only the BMW LLM
gateway is reachable. The gateway is OpenAI-spec compatible, but a call needs four things a
plain PydanticAI ``"provider:model"`` string cannot express:

1. a region-specific base URL (e.g. ``https://api.gcp.cloud.bmw/llmapi/v1``),
2. a static ``x-apikey`` header (the LLM API key),
3. an OAuth2 Bearer access token (WebEAM machine2machine — it *expires*), and
4. SSL verification against BMW's CA bundle.

We express that as a PydanticAI ``OpenAIChatModel`` backed by an ``OpenAIProvider`` whose
``http_client`` is an ``httpx.AsyncClient`` carrying the header + CA cert, and whose bearer
token is refreshed per request via :class:`BMWAuth`. A model is routed here when its string
carries the ``bmw:`` prefix (``bmw:openai/gpt-5-mini``) — see :func:`resolve_model`, which
every agent builder calls. The global ``CAD_GEN_BMW=1`` toggle (handled in ``models.py``)
rewrites all model strings to ``bmw:`` so a locked-down BMW PC works with one switch.
"""

import os
from collections.abc import Generator
from datetime import datetime, timedelta

import httpx
from pydantic_ai.models import Model

# ── Model-string routing ──────────────────────────────────────────────────────
BMW_PREFIX = "bmw:"

# ── Environment-variable contract ─────────────────────────────────────────────
ENV_API_KEY = "LLM_API_PROD_KEY"  # the x-apikey header value
ENV_CLIENT_ID = "CLIENT_ID"  # WebEAM machine2machine client id
ENV_CLIENT_SECRET = "CLIENT_SECRET"  # WebEAM machine2machine client secret
ENV_ACCESS_TOKEN = "LLM_ACCESS_TOKEN"  # optional pre-fetched bearer token
ENV_ACCESS_TOKEN_EXP = "LLM_ACCESS_TOKEN_EXP"  # ISO timestamp the pre-fetched token expires
ENV_BASE_URL = "BMW_LLM_BASE_URL"  # gateway base (without the /v1 suffix)
ENV_CA_CERT_PATH = "BMW_CA_CERT_PATH"  # local CA bundle path
ENV_CA_CERT_URL = "BMW_CA_CERT_URL"  # where to download the CA bundle from
ENV_AUTH_ENDPOINT = "BMW_AUTH_ENDPOINT"  # WebEAM access-token endpoint

# Defaults target EU PROD. EU QA: https://api.int.gcp.cloud.bmw/llmapi ;
# US variants append /us (e.g. https://api.gcp.cloud.bmw/llmapi/us).
DEFAULT_BASE_URL = "https://api.gcp.cloud.bmw/llmapi"
DEFAULT_CA_CERT_PATH = "BMW_Trusted_Certificates_Latest.pem"
DEFAULT_CA_CERT_URL = "https://trustbundle.bmwgroup.net/BMW_Trusted_Certificates_Latest.pem"
DEFAULT_AUTH_ENDPOINT = (
    "https://auth.bmwgroup.net/auth/oauth2/realms/root/realms/machine2machine/access_token"
)

_DEFAULT_TTL_S = 3600  # assumed token lifetime when the server omits expires_in
_REFRESH_MARGIN_S = 60  # refresh this many seconds before a token actually expires
_HTTP_TIMEOUT_S = 30.0


class BMWConfigError(RuntimeError):
    """Raised when BMW credentials/config are missing or incomplete."""


def bmw_enabled() -> bool:
    """True when the global ``CAD_GEN_BMW`` toggle is set to a truthy value."""
    return os.environ.get("CAD_GEN_BMW", "").strip().lower() in {"1", "true", "yes", "on"}


def require_bmw_credentials() -> list[str]:
    """Return the names of any missing BMW credentials (empty list ⇒ all present).

    Needs the ``x-apikey`` plus *either* a CLIENT_ID/CLIENT_SECRET pair (to fetch a token)
    *or* a pre-supplied LLM_ACCESS_TOKEN.
    """
    missing: list[str] = []
    if not os.environ.get(ENV_API_KEY):
        missing.append(ENV_API_KEY)
    has_token = bool(os.environ.get(ENV_ACCESS_TOKEN))
    has_client = bool(os.environ.get(ENV_CLIENT_ID) and os.environ.get(ENV_CLIENT_SECRET))
    if not has_token and not has_client:
        missing.append(f"{ENV_CLIENT_ID}+{ENV_CLIENT_SECRET} (or {ENV_ACCESS_TOKEN})")
    return missing


def download_ca_cert(path: str | None = None, url: str | None = None) -> str:
    """Return a local path to BMW's CA bundle, downloading it once if absent.

    Mirrors the documented ``download_bmw_ca_cert`` helper but uses httpx (already a
    dependency) instead of requests. A cached file is reused as-is.
    """
    path = path or os.environ.get(ENV_CA_CERT_PATH) or DEFAULT_CA_CERT_PATH
    url = url or os.environ.get(ENV_CA_CERT_URL) or DEFAULT_CA_CERT_URL
    if os.path.exists(path):
        return path
    response = httpx.get(url, timeout=_HTTP_TIMEOUT_S)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def fetch_access_token(
    client_id: str, client_secret: str, endpoint: str | None = None
) -> tuple[str, int]:
    """Obtain a WebEAM machine2machine access token; returns ``(token, expires_in_seconds)``."""
    endpoint = endpoint or os.environ.get(ENV_AUTH_ENDPOINT) or DEFAULT_AUTH_ENDPOINT
    response = httpx.post(
        endpoint,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "machine2machine",
        },
        timeout=_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise BMWConfigError("WebEAM response did not contain an access_token.")
    return token, int(payload.get("expires_in") or _DEFAULT_TTL_S)


def _parse_expiry(value: str | None) -> datetime | None:
    """Parse an ISO timestamp (optionally quoted, as in the docs' .env block); None if unset."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().strip("'\""))
    except ValueError:
        return None


class _TokenCache:
    """Caches the bearer token and refreshes it when missing or near expiry.

    Prefers a pre-supplied ``LLM_ACCESS_TOKEN`` (while unexpired); otherwise fetches one via
    CLIENT_ID/CLIENT_SECRET. Datetimes are naive/local, matching the documented WebEAM helper.
    """

    def __init__(self) -> None:
        self._token: str | None = None
        self._expiry: datetime | None = None

    def get_token(self) -> str:
        now = datetime.now()
        margin = timedelta(seconds=_REFRESH_MARGIN_S)
        if self._token is not None and self._expiry is not None and now < self._expiry - margin:
            return self._token

        env_token = os.environ.get(ENV_ACCESS_TOKEN)
        if env_token:
            exp = _parse_expiry(os.environ.get(ENV_ACCESS_TOKEN_EXP))
            if exp is None or now < exp - margin:
                self._token = env_token
                self._expiry = exp or (now + timedelta(seconds=_DEFAULT_TTL_S))
                return self._token

        client_id = os.environ.get(ENV_CLIENT_ID)
        client_secret = os.environ.get(ENV_CLIENT_SECRET)
        if not client_id or not client_secret:
            raise BMWConfigError(
                f"No valid BMW access token: set {ENV_ACCESS_TOKEN} or "
                f"{ENV_CLIENT_ID}+{ENV_CLIENT_SECRET}."
            )
        token, expires_in = fetch_access_token(client_id, client_secret)
        self._token = token
        self._expiry = now + timedelta(seconds=expires_in)
        return token


class BMWAuth(httpx.Auth):
    """Injects a fresh ``Authorization: Bearer`` header per request from the token cache.

    Setting it per request (rather than baking the token into the client) means a long
    self-refine run cannot 401 when the token rolls over mid-loop. The static ``x-apikey``
    is a client default header, so it is not re-set here.
    """

    def __init__(self, token_cache: _TokenCache) -> None:
        self._cache = token_cache

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._cache.get_token()}"
        yield request


# Shared per-process state: one token cache + one provider (and thus one httpx client)
# reused across every agent, since base URL / key / token are identical for all of them.
_token_cache = _TokenCache()
_provider = None  # type: ignore[var-annotated]  # OpenAIProvider, built lazily


def _build_http_client(cert_path: str, api_key: str) -> httpx.AsyncClient:
    """Build the verified async client (isolated so tests can stub out the network/SSL)."""
    return httpx.AsyncClient(
        headers={"x-apikey": api_key},
        verify=cert_path,
        auth=BMWAuth(_token_cache),
    )


def _shared_provider():
    """Lazily build (and cache) the OpenAIProvider pointed at the BMW gateway."""
    global _provider
    if _provider is not None:
        return _provider

    # Imported lazily so the hot resolve_model() path stays import-light for non-BMW models.
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = os.environ.get(ENV_API_KEY)
    if not api_key:
        raise BMWConfigError(f"{ENV_API_KEY} is not set — required for the BMW LLM API.")
    base_url = (os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
    cert_path = download_ca_cert()
    token = _token_cache.get_token()  # warm + fail fast with a clear error if creds missing
    _provider = OpenAIProvider(
        base_url=f"{base_url}/v1",
        api_key=token,
        http_client=_build_http_client(cert_path, api_key),
    )
    return _provider


def build_bmw_model(name: str) -> Model:
    """Build a PydanticAI model for ``name`` (e.g. ``openai/gpt-5-mini``) on the BMW gateway."""
    from pydantic_ai.models.openai import OpenAIChatModel

    return OpenAIChatModel(name, provider=_shared_provider())


def resolve_model(model: str | Model) -> str | Model:
    """Convert a ``bmw:<name>`` string into a BMW-backed model; pass everything else through.

    Called as the first line of every agent builder, so the BMW gateway is reachable from
    the CLI, the web backend, and the orchestrator with no call-site changes.
    """
    if isinstance(model, str) and model.startswith(BMW_PREFIX):
        return build_bmw_model(model[len(BMW_PREFIX) :])
    return model
