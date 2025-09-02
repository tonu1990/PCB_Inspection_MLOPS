from typing import List, Optional
from pydantic import BaseModel

class InfoResponse(BaseModel):
    app_image_tag: str
    model_path: str
    model_resolved_path: str
    model_sha256: str
    opset: Optional[int]
    input_names: List[str]
    output_names: List[str]
    last_loaded_utc: Optional[str]

class InferResponse(BaseModel):
    latency_ms: int
    meta: dict
    outputs_preview: list
