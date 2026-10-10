"""HTTP API: generate images and detect NSFW content.

    GET  /v1/health                  liveness, and what is loaded
    GET  /v1/models, /v1/detectors   what is configured
    POST /v1/generate                prompt -> image(s)
    POST /v1/generate-and-classify   prompt -> image(s) + NSFW verdicts
    POST /v1/classify                image (path or base64) -> NSFW verdict
    POST /v1/classify/upload         uploaded image -> NSFW verdict
    GET  /v1/images/{image_id}       fetch a generated image
    GET  /v1/runs/{run_id}           a pipeline run's manifest

Results are written under ``<paths.serving>/api/``. Set the API_KEY environment
variable (or ``serve.api_key``) to require ``Authorization: Bearer <key>``.
"""

import base64
import binascii
import contextlib
import logging
import os
import uuid
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from visual_synthesis import __version__
from visual_synthesis.log import setup_logging
from visual_synthesis.runs import list_runs
from visual_synthesis.serving.engine import Engine

log = logging.getLogger(__name__)

# Body fields of POST /v1/generate that are passed straight to Engine.generate.
GENERATE_FIELDS = ("model", "prompt_id", "seed", "negative_prompt", "guidance_scale", "width", "height", "threshold")


def create_app(cfg, engine=None):
    """Build the FastAPI app for a resolved config."""
    engine = engine or Engine(cfg)
    serve = cfg["serve"]
    out_root = Path(cfg["paths"]["serving"]) / "api"
    api_key = os.environ.get("API_KEY") or serve["api_key"]

    def check_key(authorization=Header(None)):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        engine.close()  # release the models when the server stops

    app = FastAPI(title="visual-synthesis", version=__version__, docs_url=serve["docs_url"], lifespan=lifespan)
    if serve["cors_origins"]:
        app.add_middleware(
            CORSMiddleware, allow_origins=serve["cors_origins"], allow_methods=["*"], allow_headers=["*"]
        )
    protected = [Depends(check_key)]

    def request_dir(kind):
        path = out_root / f"{kind}_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_generate(body, detect):
        prompt = str(body.get("prompt") or "").strip()
        n = int(body.get("n", 1))
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        if not 1 <= n <= serve["max_batch"]:
            raise HTTPException(status_code=400, detail=f"n must be between 1 and serve.max_batch={serve['max_batch']}")
        options = {key: body[key] for key in GENERATE_FIELDS if body.get(key) is not None}
        options["steps"] = body.get("num_inference_steps")
        # Detection runs only when asked for; "detector" picks which one (default: nsfw.detector).
        options["detector"] = body.get("detector") if (detect or body.get("detect")) else "off"
        try:
            return engine.generate(prompt, request_dir("generate"), n=n, **options)
        except KeyError as exc:  # unknown model or detector id
            raise HTTPException(status_code=400, detail=str(exc.args[0])) from exc
        except Exception as exc:
            log.exception("generation failed")
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    def run_classify(path, detector, threshold):
        try:
            return engine.classify(path, detector, threshold)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc.args[0])) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not classify the image: {exc}") from exc

    # ---------------------------------------------------------------- meta

    @app.get("/")
    def root():
        return {"service": "visual-synthesis", "docs": serve["docs_url"]}

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "version": __version__, "project": cfg["project_name"], "loaded": engine.loaded()}

    @app.get("/v1/models", dependencies=protected)
    def models():
        return [{"id": name, **model} for name, model in cfg["models"].items()]

    @app.get("/v1/detectors", dependencies=protected)
    def detectors():
        return [{"id": name, **d} for name, d in cfg["detectors"].items() if d["enabled"]]

    @app.get("/v1/runs/{run_id}", dependencies=protected)
    def get_run(run_id):
        runs_root = Path(cfg["paths"]["runs"])
        stage_dirs = sorted(p.name for p in runs_root.iterdir()) if runs_root.is_dir() else []
        found = [m for slug in stage_dirs for m in list_runs(runs_root, slug) if m.get("run_id") == run_id]
        if not found:
            raise HTTPException(status_code=404, detail=f"no run with id {run_id!r}")
        # A chained run has one manifest per stage; the config is left out for brevity.
        return [{k: v for k, v in manifest.items() if k != "config"} for manifest in found]

    # ------------------------------------------------------------ generate

    @app.post("/v1/generate", dependencies=protected)
    def generate(body=Body(...)):
        """Body: {"prompt": ..., "model", "n", "seed", "num_inference_steps", "guidance_scale",
        "width", "height", "negative_prompt", "prompt_id", "detect", "detector", "threshold"}."""
        return run_generate(body, detect=False)

    @app.post("/v1/generate-and-classify", dependencies=protected)
    def generate_and_classify(body=Body(...)):
        """Same body as /v1/generate; always runs NSFW detection on the result."""
        return run_generate(body, detect=True)

    # ------------------------------------------------------------ classify

    @app.post("/v1/classify", dependencies=protected)
    def classify(body=Body(...)):
        """Body: {"image_path": ...} or {"image_base64": ...}, plus optional "detector", "threshold"."""
        image_path, encoded = body.get("image_path"), body.get("image_base64")
        if bool(image_path) == bool(encoded):
            raise HTTPException(status_code=400, detail="supply exactly one of image_path or image_base64")
        if image_path:
            path = Path(image_path).expanduser()
            if not path.is_file():
                raise HTTPException(status_code=400, detail=f"image not found: {path}")
        else:
            try:
                data = base64.b64decode(encoded.partition(",")[2] if encoded.startswith("data:") else encoded)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
            path = request_dir("classify") / f"{body.get('image_id') or uuid.uuid4().hex[:12]}.png"
            path.write_bytes(data)
        return run_classify(path, body.get("detector"), body.get("threshold"))

    @app.post("/v1/classify/upload", dependencies=protected)
    def classify_upload(file=File(...), detector=None, threshold=None):
        """Multipart upload of one image; ``detector`` and ``threshold`` are query parameters."""
        path = request_dir("upload") / f"{uuid.uuid4().hex[:12]}.png"
        path.write_bytes(file.file.read())
        return run_classify(path, detector, None if threshold is None else float(threshold))

    @app.get("/v1/images/{image_id}", dependencies=protected)
    def get_image(image_id):
        """Fetch an image by its file name without extension, e.g. ``<prompt_id>_<seed>``."""
        if "/" in image_id or ".." in image_id:
            raise HTTPException(status_code=400, detail="invalid image id")
        matches = sorted(Path(cfg["paths"]["serving"]).rglob(f"{image_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"no image with id {image_id!r}")
        return FileResponse(matches[0])

    return app


def run_server(cfg, host=None, port=None, reload=None):
    """Serve the API with uvicorn. CLI values override the ``serve`` config."""
    import uvicorn

    setup_logging(cfg["logging"]["level"], console_json=cfg["logging"]["console_json"], stage="api")
    serve = cfg["serve"]
    if serve["reload"] if reload is None else reload:
        # Reloading needs an import string; the factory re-reads the same config files.
        os.environ["VS_SERVE_CONFIG"] = os.pathsep.join(cfg["config_files"])
        uvicorn.run("visual_synthesis.serving.api:app_from_env", factory=True, reload=True,
                    host=host or serve["host"], port=port or serve["port"])
    else:
        uvicorn.run(create_app(cfg), host=host or serve["host"], port=port or serve["port"])


def app_from_env():
    from visual_synthesis.config import load_config

    files = [f for f in os.environ.get("VS_SERVE_CONFIG", "").split(os.pathsep) if f]
    return create_app(load_config(files))
