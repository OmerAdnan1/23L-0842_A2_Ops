"""
Task 7 - Model API deployed to the cloud.

    GET  /health             liveness + model name
    POST /predict            multipart field "image", header "X-API-Key"

Run:
    API_KEY=... gunicorn -b 0.0.0.0:8080 -w 1 --threads 4 "app.api:create_api()"
"""
import hmac
import time

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from app.config import Settings
from app.validation import InMemoryRequest, UploadError, read_upload


def create_api(settings=None, predictor=None):
    settings = settings or Settings.from_env()
    if not settings.api_key:
        raise RuntimeError("API_KEY must be set for the model API")
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    if predictor is None:
        from app.inference import Predictor
        predictor = Predictor(settings.model_path, settings.threshold, settings.model_name)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=predictor.name, threshold=predictor.threshold)

    @app.post("/predict")
    def predict():
        if not hmac.compare_digest(request.headers.get("X-API-Key", ""), settings.api_key):
            return jsonify(error="Invalid or missing API key."), 401
        raw, filename = read_upload(request.files.get("image"))
        start = time.perf_counter()
        v = predictor.predict(raw, filename)
        return jsonify(filename=filename, label=v.label, p_ai=round(v.p_ai, 6),
                       confidence=round(v.confidence, 6), threshold=v.threshold,
                       model=predictor.name,
                       latency_ms=round((time.perf_counter() - start) * 1000, 1))

    @app.errorhandler(UploadError)
    def bad_upload(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(HTTPException)
    def http_error(e):
        return jsonify(error=e.description), e.code

    return app
