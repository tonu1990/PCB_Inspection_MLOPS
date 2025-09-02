import os, time, io
from typing import Optional, List, Tuple, Any, Dict
import numpy as np
from PIL import Image
import onnxruntime as ort

class ModelState:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.opset: Optional[int] = None
        self.last_loaded_utc: Optional[str] = None
        self.model_path: Optional[str] = None

def _open_session(model_path: str) -> ort.InferenceSession:
    # Use CPU EP; it’s the most stable on Pi4/Pi5. (GPU/NPU choices come later.)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

def load_or_reload(state: ModelState, model_path: str, force: bool = False):
    resolved = os.path.realpath(model_path)
    if not os.path.exists(resolved):
        state.session = None
        state.model_path = None
        return
    if (not force) and state.model_path == resolved and state.session is not None:
        return  # already loaded
    state.session = _open_session(resolved)
    # Try to read opset from model metadata if available
    try:
        state.opset = state.session.get_modelmeta().custom_metadata_map.get("opset", None)  # may be None
    except Exception:
        state.opset = None
    state.model_path = resolved
    state.last_loaded_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _preprocess_rgb(image_bytes: bytes, input_shape: List[int]) -> np.ndarray:
    """
    Very generic preproc:
    - load as RGB
    - resize to (H, W) from input_shape (N,C,H,W)
    - normalize to [0,1], CHW, float32
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    _, c, h, w = input_shape
    img = img.resize((w, h))
    arr = np.asarray(img).astype("float32") / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, 0)        # NCHW
    return arr

def run_inference_safe(state: ModelState, image_bytes: bytes):
    sess = state.session
    if sess is None:
        raise RuntimeError("session is None")

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    # Assume 1 input tensor with NCHW. If different, we still try to infer shape.
    input_name = inputs[0].name
    input_shape = [d if isinstance(d, int) else 1 for d in inputs[0].shape]  # replace dynamic with 1
    inp = _preprocess_rgb(image_bytes, input_shape)

    ort_inputs = {input_name: inp}
    ort_outs: List[np.ndarray] = sess.run([o.name for o in outputs], ort_inputs)

    # Return small preview: shape & first few values for each output
    preview = []
    for name, arr in zip([o.name for o in outputs], ort_outs):
        flat = arr.flatten()
        preview.append(
            (name, list(arr.shape), [float(x) for x in flat[: min(10, flat.size)]])
        )

    meta: Dict[str, Any] = {
        "input_name": input_name,
        "input_shape": input_shape,
        "output_names": [o.name for o in outputs],
    }
    return preview, meta
