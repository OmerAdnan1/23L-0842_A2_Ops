# VisionGuard web app (Task 6.2). Build: docker build -t visionguard-web .
FROM python:3.11-slim

# CPU-only TensorFlow: no CUDA libraries, a much smaller image, and enough for
# single-image inference. Override to compare sizes (Task 6.4):
#   docker build --build-arg TF_PACKAGE=tensorflow -t visionguard-web:fulltf .
ARG TF_PACKAGE=tensorflow-cpu
ARG TF_VERSION=2.21.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) dependencies first: this layer is reused until requirements change
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt "${TF_PACKAGE}==${TF_VERSION}"

# 2) unprivileged user, then the source code (changes most often -> last)
RUN useradd --create-home --uid 1000 appuser
COPY app/ app/
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--access-logfile", "-", "app.web:create_app()"]
