from typing import List, Optional
from pydantic import BaseModel
from pydantic import ConfigDict  # v2-style config

class InfoResponse(BaseModel):

    model_config = ConfigDict(protected_namespaces=())

    app_image_tag: str
    model_path: str
    model_resolved_path: str
    model_sha256: str
    opset: Optional[int]
    input_names: List[str]
    output_names: List[str]
    last_loaded_utc: Optional[str]

class InferResponse(BaseModel):

    # Same config for consistency (not strictly required here)
    model_config = ConfigDict(protected_namespaces=())
    
    latency_ms: int
    meta: dict
    outputs_preview: list
