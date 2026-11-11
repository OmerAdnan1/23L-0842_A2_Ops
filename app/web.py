"""
Task 4 / 5 / 8 - VisionGuard web app.

Pages: upload (/), result (POST /predict), history (/history), health (/health).
INFERENCE_MODE=local loads the Keras model in-process once at start-up;
INFERENCE_MODE=remote sends the image to the deployed model API instead.
Every prediction is written to the database given by DATABASE_URL.

Run:
    flask --app "app.web:create_app()" run --port 8000            (debug off)
    gunicorn -b 0.0.0.0:8000 -w 1 --threads 4 "app.web:create_app()"
"""
import base64
import logging
import time

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.config import Settings
from app.db import Store
from app.validation import InMemoryRequest, UploadError, read_upload

log = logging.getLogger(__name__)


def make_predictor(settings):
    if settings.inference_mode == "remote":
        from app.remote import RemotePredictor
        return RemotePredictor(settings.model_api_url, settings.model_api_key)
    from app.inference import Predictor
    return Predictor(settings.model_path, settings.threshold, settings.model_name)


def create_app(settings=None, predictor=None, store=None):
    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config.update(MAX_CONTENT_LENGTH=settings.max_upload_mb * 1024 * 1024,
                      SECRET_KEY=settings.secret_key)
    predictor = predictor or make_predictor(settings)
    store = store or Store(settings.database_url)

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=settings.max_upload_mb)

    @app.post("/predict")
    def predict():
        raw, filename = read_upload(request.files.get("image"))
        start = time.perf_counter()
        v = predictor.predict(raw, filename)
        latency = (time.perf_counter() - start) * 1000
        record = store.add(filename=filename, verdict=v.label, p_ai=v.p_ai,
                           confidence=v.confidence, threshold=v.threshold,
                           model_name=predictor.name, source=settings.inference_mode,
                           latency_ms=latency)
        mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        preview = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        return render_template("result.html", v=v, record=record, preview=preview)

    @app.get("/history")
    def history():
        return render_template("history.html", rows=store.recent(100))

    @app.get("/health")
    def health():
        db_ok = store.ping()
        return jsonify(status="ok" if db_ok else "degraded",
                       database="ok" if db_ok else "unreachable",
                       inference_mode=settings.inference_mode,
                       model=predictor.name,
                       threshold=predictor.threshold), (200 if db_ok else 503)

    @app.errorhandler(UploadError)
    def bad_upload(e):
        return render_template("error.html", title="Upload rejected", message=str(e)), 400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        return render_template("error.html", title="File too large",
                               message=f"The maximum upload size is "
                                       f"{settings.max_upload_mb} MB."), 413

    @app.errorhandler(HTTPException)
    def http_error(e):
        return render_template("error.html", title=f"{e.code} {e.name}",
                               message=e.description), e.code

    @app.errorhandler(Exception)
    def server_error(e):
        log.exception("Unhandled error")
        return render_template("error.html", title="Something went wrong",
                               message="The image could not be classified right now. "
                                       "Please try again later."), 502

    return app
