"""FastAPI web backend exposing the cad-gen self-refine loop over HTTP/SSE."""

from cad_gen.web.server import create_app

__all__ = ["create_app"]
