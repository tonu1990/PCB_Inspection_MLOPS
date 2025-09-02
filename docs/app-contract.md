# App Contract (Edge API)

## Endpoints
- `GET /healthz` → `{ ok: true }` if process is up.
- `GET /readyz` → `{ ready: true }` if model session is loaded.
- `POST /reload` → reloads MODEL_PATH and reopens ONNX session.
- `GET /info` → returns `{ app_image_tag, model_* metadata }`.
- `POST /infer` (multipart form: `image=@file.jpg`) → returns raw outputs preview for now.

## Environment
- `MODEL_PATH` (default `/opt/edge/models/current.onnx`)
- `PORT` (default `8080`)
- `APP_IMAGE_TAG` (injected by CI at build time)
- `TASK`, `INPUT_SHAPE`, `SCORE_THRESH`, `NMS_IOU` (reserved for adapters later)

## Mounts
- Read-only: `/opt/edge/models:/opt/edge/models:ro`

## Reload flow
- Model CD flips `current.onnx`
- Model CD (deliver job) calls `POST /reload`
- App closes & reopens ONNX session; `/readyz` must stay green

## Notes
- Initial `/infer` returns raw output preview. Decoding for YOLO etc. comes next.
