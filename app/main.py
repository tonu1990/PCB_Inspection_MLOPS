from fastapi import FastAPI, Response, status, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from pathlib import Path

from app.core.settings import settings
from app.inference.manager import ModelManager
from app.utils.image_io import load_upload_as_rgb_numpy

from fastapi.responses import StreamingResponse  # add import
import io, json
from pathlib import Path

app = FastAPI(title=settings.app_name)
app.mount("/ui", StaticFiles(directory="app/ui", html=True), name="ui")

@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

manager = ModelManager()

@app.on_event("startup")
def startup() -> None:
    try:
        manager.load_if_available(settings.model_path)
    except Exception as e:
        import logging
        logging.getLogger("uvicorn.error").warning(f"Model failed to load: {e}")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readyz")
def readyz(response: Response):
    if manager.predict_dummy_ok():
        return {"ready": True}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": False, "reason": "model_not_loaded"}

@app.get("/info")
def info():
    return {
        "app": settings.app_name,
        "configured_model_path": settings.model_path,  # helps spot env/path mistakes
        "model": manager.info(),
    }

# ---------- NEW: Hot-reload endpoint ----------

class ReloadRequest(BaseModel):
    # Optional: allow overriding the path in dev. In prod you omit this.
    path: str | None = None

@app.post("/reload")
def reload_model(req: ReloadRequest | None = None):
    """
    Hot-reload the model. In production you typically POST {} (no path)
    after your Model-CD pipeline flips /opt/edge/models/current.onnx.
    For local dev you may pass {"path": "C:\\full\\path\\to\\file.onnx"} to try another file.
    """
    override = None
    if req and req.path:
        override = req.path
        # Basic sanity for prototype
        if not override.lower().endswith(".onnx"):
            raise HTTPException(status_code=400, detail="Only .onnx models are supported")
        if not Path(override).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {override}")

    ok = manager.reload(override)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Reload failed: {manager.info().get('last_error')}")

    return {"reloaded": True, "model": manager.info()}

# ---------- Existing /predict stays the same ----------
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Accepts an image and returns YOLO-like detections (prototype):
    - boxes in original image coords (xyxy)
    - score and class_id
    """
    if not manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded; /readyz is false")
    try:
        img_rgb = load_upload_as_rgb_numpy(file)
        dets = manager.predict_detections(img_rgb, conf_thres=0.25, iou_thres=0.45, top_k=300)
        return {
            "ok": True,
            "detections": dets,
            "input_hw_used": manager.info().get("input_hw"),
            "layout": manager.info().get("input_layout"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

@app.post("/predict_image")
def predict_image(file: UploadFile = File(...)):
    """
    Returns an annotated PNG with boxes + labels drawn on the uploaded image.
    If a labels.json is present next to the model file, those names are used.
    Otherwise we show class_id.
    """
    if not manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded; /readyz is false")

    # 1) Decode image
    try:
        img_rgb = load_upload_as_rgb_numpy(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) Run detections
    try:
        dets = manager.predict_detections(img_rgb, conf_thres=0.25, iou_thres=0.45, top_k=300)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # 3) Optional labels: look for a labels.json next to the model file
    labels = None
    try:
        mp = manager.info().get("model_path")
        if mp:
            lbl_path = Path(mp).with_name("labels.json")
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Expect either ["classA","classB",...] or {"names":[...]}
                    if isinstance(data, list):
                        labels = data
                    elif isinstance(data, dict) and "names" in data and isinstance(data["names"], list):
                        labels = data["names"]
    except Exception:
        # If labels loading fails, just fall back to class_id
        labels = None

    # 4) Render and return PNG
    im = manager.draw_detections(img_rgb, dets, labels=labels)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")