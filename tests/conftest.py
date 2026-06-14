import pytest
from pydantic_ai import models

# No test may ever hit a real LLM API.
models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch):
    """Tests must never read the developer's local .env.

    The app calls ``load_dotenv(Path.cwd() / ".env")`` in several entrypoints. During a
    test run that reads the developer's real, gitignored .env (real models + API keys)
    and permanently mutates ``os.environ`` for the process, making config-default
    assertions fail and leak across tests in an order-dependent way. Neutralise it so the
    suite is hermetic, mirroring the ALLOW_MODEL_REQUESTS guard above.
    """
    noop = lambda *a, **k: False  # noqa: E731 - trivial stub
    for mod in ("cad_gen.cli", "cad_gen.web.server", "cad_gen.reproject.eval_drawings"):
        monkeypatch.setattr(f"{mod}.load_dotenv", noop, raising=False)
