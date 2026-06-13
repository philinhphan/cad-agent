# Backend container for the cad-gen web view (FastAPI + CadQuery/OCCT).
# The frontend (web/) deploys separately to Vercel; this image hosts the API.
#
#   docker build -t cad-gen-api .
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... \
#     -e CAD_GEN_WEB_ORIGINS=https://your-app.vercel.app \
#     -v cadgen-runs:/data/runs cad-gen-api
FROM python:3.12-slim

# uv for fast, locked installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System libs OCCT/VTK link against at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libxrender1 libxext6 libsm6 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install deps (main + [web] extra) against the lockfile; the local package too.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --extra web --no-dev --frozen

ENV PATH="/app/.venv/bin:$PATH" \
    CAD_GEN_RUNS_DIR=/data/runs \
    CAD_GEN_WEB_ORIGINS=http://localhost:3000
EXPOSE 8000

CMD ["uvicorn", "cad_gen.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
