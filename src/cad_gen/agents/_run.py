"""Shared agent-run helper: deterministic (temperature 0) with graceful fallback.

Temperature 0 cuts critic/extractor variance, but some models — notably OpenAI reasoning
models like gpt-5.5 — reject any non-default temperature ("Only the default (1) value is
supported"). We therefore set the temperature at RUN time and, if the model refuses it,
retry once without it rather than letting the whole step fail (which previously left the
typed target empty and dropped the envelope check).
"""

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.settings import ModelSettings

DETERMINISTIC = ModelSettings(temperature=0.0)


async def run_deterministic(agent, content):
    """Run `agent` at temperature 0, retrying without it if the model rejects temp 0."""
    try:
        return await agent.run(content, model_settings=DETERMINISTIC)
    except ModelHTTPError as exc:
        if "temperature" in str(exc).lower():
            return await agent.run(content)
        raise
