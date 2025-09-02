from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os, hashlib, time
from app.onnx_loader import ModelState, load_or_reload, run_inference_safe
from app.schemas import InfoResponse, InferResponse

APP_IMAGE_TAG = os.getenv("APP_IMAGE_TAG", "dev")  # set by CI later
MODEL_PATH = os.getenv("MODEL_PATH", "/opt/edge/models/current.onnx")
PORT = int(os.getenv("PORT", "8080"))

app = FastAPI(title="PCB App (FastAPI + ONNX Runtime)", version="0.1.0")
state = ModelState()

def sha256_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unknown"

@app.on_event("startup")
def _startup():
    load_or_reload(state, MODEL_PATH)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readyz")
def readyz():
    if state.session is None:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "no-session"})
    # optional: run a tiny no-op by checking io names
    try:
        _ = state.session.get_inputs()
        _ = state.session.get_outputs()
        return {"ready": True}
    except Exception as e:
        return JSONResponse(status_code=503, content={"ready": False, "reason": str(e)})

@app.post("/reload")
def reload_model():
    load_or_reload(state, MODEL_PATH, force=True)
    return {"reloaded": state.session is not None, "model_path": MODEL_PATH}

@app.get("/info", response_model=InfoResponse)
def info():
    resolved = os.path.realpath(MODEL_PATH)
    return InfoResponse(
        app_image_tag=APP_IMAGE_TAG,
        model_path=MODEL_PATH,
        model_resolved_path=resolved if os.path.exists(resolved) else "",
        model_sha256=sha256_file(resolved) if os.path.exists(resolved) else "missing",
        opset=state.opset,
        input_names=[i.name for i in state.session.get_inputs()] if state.session else [],
        output_names=[o.name for o in state.session.get_outputs()] if state.session else [],
        last_loaded_utc=state.last_loaded_utc
    )

@app.post("/infer", response_model=InferResponse)
async def infer(image: UploadFile = File(...)):
    if state.session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = await image.read()
    try:
        t0 = time.time()
        outputs, meta = run_inference_safe(state, data)
        dt_ms = int((t0 - time.time()) * -1000)
        return InferResponse(
            latency_ms=dt_ms,
            meta=meta,
            # For now: raw outputs only (lists of shape info + a few numbers).
            # In a later step we'll add YOLO-style decoding.
            outputs_preview=[
                {"name": name, "shape": shp, "first_values": vals}
                for name, shp, vals in outputs
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"inference failed: {e}")
