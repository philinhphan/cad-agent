"""Request/response models for the web API."""

from pydantic import BaseModel, Field

from cad_gen.models import RunConfig


class StartRunRequest(BaseModel):
    """Body of POST /api/runs. `config` fields all default, so partial is fine;
    `out_dir` is ignored and forced to the server's runs root."""

    spec: str = Field(min_length=1)
    config: RunConfig = Field(default_factory=RunConfig)


class StartRunResponse(BaseModel):
    run_id: str
