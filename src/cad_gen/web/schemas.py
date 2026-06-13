"""Response models for the web API.

Run-start requests arrive as ``multipart/form-data`` (spec + JSON-encoded config +
optional drawing file uploads), so they are parsed field-by-field in the route rather
than bound to a single Pydantic body model.
"""

from pydantic import BaseModel


class StartRunResponse(BaseModel):
    run_id: str
