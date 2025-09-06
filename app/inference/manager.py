"""
ModelManager: ONNX Runtime session + YOLOv8 decode + NMS (prototype).

Outputs:
- predict_detections(img_rgb) -> list of detections in original image coords:
  [
    {"x1": float, "y1": float, "x2": float, "y2": float,
     "score": float, "class_id": int}
  ]

Notes (prototype-friendly):
- Letterbox preprocessing to target (H,W) with gray padding, like YOLO.
- Decode assumes output shape [1, 84, N] (4 box + 80 class scores).
- Scores = max per-class score (no objectness head in v8).
- Simple greedy NMS on CPU.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import hashlib
import numpy as np
from PIL import Image

from app.core.settings import settings
from PIL import Image, ImageDraw, ImageFont 


class ModelManager:
    def __init__(self) -> None:
        self.session = None
        self.model_sha256: Optional[str] = None
        self.model_path: Optional[Path] = None
        self.input_name: Optional[str] = None
        self.input_layout: Optional[str] = None  # "NCHW" or "NHWC"
        self.input_hw: Optional[Tuple[int, int]] = None
        self.last_error: Optional[str] = None

    # ---------- (Re)load ----------

    def load_if_available(self, path: str) -> bool:
        p = Path(path)
        self.model_path = p
        self.last_error = None

        if not p.exists():
            self._clear_session()
            return False

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            self.model_sha256 = self._sha256_file(p)
            self._infer_input_spec()
            return True
        except Exception as e:
            self._clear_session()
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    def reload(self, path: Optional[str] = None) -> bool:
        target = path or settings.model_path
        return self.load_if_available(target)

    def _clear_session(self) -> None:
        self.session = None
        self.model_sha256 = None
        self.input_name = None
        self.input_layout = None
        self.input_hw = None

    @staticmethod
    def _sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ---------- Introspection ----------

    def _infer_input_spec(self) -> None:
        assert self.session is not None, "Session not initialized"
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("Model has no inputs")
        first = inputs[0]
        self.input_name = first.name

        def as_int(d):
            try:
                return int(d)
            except Exception:
                return None

        shape = [as_int(d) for d in first.shape]  # e.g., [1, 3, 640, 640] or [1, 640, 640, 3]
        if len(shape) == 4:
            _, d1, d2, d3 = shape
            if d1 == 3:  # NCHW
                self.input_layout = "NCHW"
                H = d2 or settings.model_input_size
                W = d3 or settings.model_input_size
                self.input_hw = (H, W)
            elif d3 == 3:  # NHWC
                self.input_layout = "NHWC"
                H = d1 or settings.model_input_size
                W = d2 or settings.model_input_size
                self.input_hw = (H, W)
            else:
                self.input_layout = "NCHW"
                self.input_hw = (settings.model_input_size, settings.model_input_size)
        else:
            self.input_layout = "NCHW"
            self.input_hw = (settings.model_input_size, settings.model_input_size)

    # ---------- Readiness & info ----------

    def is_ready(self) -> bool:
        return self.session is not None

    def info(self) -> dict:
        return {
            "loaded": self.is_ready(),
            "model_path": str(self.model_path) if self.model_path else None,
            "model_sha256": self.model_sha256,
            "input_name": self.input_name,
            "input_layout": self.input_layout,
            "input_hw": self.input_hw,
            "last_error": self.last_error,
        }

    # ---------- Preprocess / Postprocess ----------

    def _letterbox(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Letterbox to (H,W) keeping aspect ratio with gray padding (114).
        Returns:
          tensor: float32 NCHW/NHWC in [0,1] with batch dim
          meta:   {"ratio": r, "pad": (dw, dh), "orig_hw": (H0, W0)}
        """
        assert self.session is not None
        assert self.input_layout is not None and self.input_hw is not None

        Ht, Wt = self.input_hw
        H0, W0, _ = img_rgb.shape

        # Compute scaling ratio and padding (like YOLO)
        r = min(Wt / W0, Ht / H0)
        new_w, new_h = int(round(W0 * r)), int(round(H0 * r))
        dw, dh = (Wt - new_w) / 2, (Ht - new_h) / 2  # left/right and top/bottom pad (each side)

        # Resize then pad into a new image
        pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (Wt, Ht), (114, 114, 114))
        canvas.paste(pil, (int(round(dw)), int(round(dh))))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0  # (Ht, Wt, 3)

        if self.input_layout == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))  # (3, Ht, Wt)

        arr = np.expand_dims(arr, axis=0)  # add batch dim
        meta = {"ratio": r, "pad": (dw, dh), "orig_hw": (H0, W0)}
        return arr, meta

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] -> [x1, y1, x2, y2]."""
        x, y, w, h = xywh.T
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> List[int]:
        """
        Greedy NMS. boxes: [N,4] in xyxy; scores: [N].
        Returns indices of kept boxes.
        """
        if boxes.size == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = (xx2 - xx1).clip(min=0)
            h = (yy2 - yy1).clip(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thres]
        return keep

    # ---------- Inference ----------

    def predict_detections(
        self,
        img_rgb: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        top_k: int = 300
    ) -> List[Dict]:
        """
        Full path: preprocess -> ONNX -> decode -> NMS -> map to original image coords.
        Returns a list of dicts with xyxy in original image size.
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded yet")
        assert self.input_name is not None

        # 1) Preprocess (letterbox) and run the model
        inp, meta = self._letterbox(img_rgb)
        outputs = self.session.run(None, {self.input_name: inp})
        pred = outputs[0]  # assume first output is the head: [1, 84, N]

        # 2) Decode YOLOv8: [1, 84, N] -> [N, 84]
        pred = np.squeeze(pred, axis=0)
        if pred.shape[0] == 84:   # common case: (84, N)
            pred = pred.T         # -> (N, 84)
        elif pred.shape[1] == 84: # already (N, 84)
            pass
        else:
            raise RuntimeError(f"Unexpected output shape {outputs[0].shape}, expected 84 channels")

        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]                 # (N, num_classes)
        scores = cls_scores.max(axis=1)          # best class score
        class_ids = cls_scores.argmax(axis=1)    # best class id

        # 3) Filter low confidence
        keep_conf = scores >= conf_thres
        if not np.any(keep_conf):
            return []
        boxes_xywh = boxes_xywh[keep_conf]
        scores = scores[keep_conf]
        class_ids = class_ids[keep_conf]

        # 4) xywh -> xyxy in model space
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        # 5) Undo letterbox to original image coords
        r = meta["ratio"]
        dw, dh = meta["pad"]
        H0, W0 = meta["orig_hw"]
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dw) / max(r, 1e-9)
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dh) / max(r, 1e-9)
        # clip to image bounds
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, W0 - 1)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, H0 - 1)

        # 6) NMS
        order = scores.argsort()[::-1]
        if top_k > 0:
            order = order[:top_k]
        keep_nms = self._nms(boxes_xyxy[order], scores[order], iou_thres=iou_thres)
        keep_idx = order[keep_nms]

        # 7) Build result list
        dets: List[Dict] = []
        for i in keep_idx:
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            dets.append({
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "score": float(scores[i]),
                "class_id": int(class_ids[i]),
            })
        return dets

    # keep this for /readyz
    def predict_dummy_ok(self) -> bool:
        return self.is_ready()
    
    def draw_detections(self, img_rgb: np.ndarray, dets: list[dict], labels: list[str] | None = None) -> Image.Image:
        """
        Render boxes + labels onto the original RGB image.

        - dets: list of {"x1","y1","x2","y2","score","class_id"} in ORIGINAL image coords
        - labels: optional class-name list; falls back to class_id if missing
        """
        # Make a copy to avoid mutating caller's array
        im = Image.fromarray(img_rgb.copy())
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None  # very rare, but Pillow can work without a font

        # Simple color palette (cycled by class_id)
        palette = [
            (255, 56, 56), (255, 159, 56), (255, 255, 56), (56, 255, 56),
            (56, 255, 255), (56, 56, 255), (255, 56, 255), (180, 180, 180)
        ]
        # Thickness scales with image size (keeps lines visible on large images)
        thick = max(2, int(min(im.size) / 200))

        for d in dets:
            x1, y1, x2, y2 = map(int, (d["x1"], d["y1"], d["x2"], d["y2"]))
            cls_id = int(d["class_id"])
            score  = float(d["score"])
            color  = palette[cls_id % len(palette)]

            # Box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thick)

            # Caption: "<label> <score>"
            label_text = str(cls_id)
            if labels and 0 <= cls_id < len(labels):
                label_text = labels[cls_id]
            caption = f"{label_text} {score:.2f}"

            # Text background box sized to caption
            if hasattr(draw, "textbbox") and font is not None:
                # textbbox returns (l, t, r, b)
                l, t, r, b = draw.textbbox((0, 0), caption, font=font)
                tw, th = (r - l, b - t)
            else:
                # Fallback guess if no font/textbbox available
                tw, th = (7 * len(caption), 14)

            # Draw filled caption box at the top-left of the detection
            pad = 2
            draw.rectangle([x1, y1 - th - 2 * pad, x1 + tw + 2 * pad, y1], fill=color)
            # Draw caption text in dark color for contrast
            if font is not None:
                draw.text((x1 + pad, y1 - th - pad), caption, fill=(5, 5, 5), font=font)

        return im