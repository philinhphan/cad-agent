from pydantic_ai import models

# No test may ever hit a real LLM API.
models.ALLOW_MODEL_REQUESTS = False
