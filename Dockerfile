# Xenium Explorer - Python-only image for Cloud Run (or any container host).
# Build:  docker build --platform linux/amd64 -t xenium-explorer .
# See deployment_instructions.txt before deploying.
#
# SECURITY
#   * The app has no authentication of its own. Only run this image behind an
#     authenticating front end (Cloud Run with --no-allow-unauthenticated, IAP, a VPN).
#   * XENIUM_ENABLE_REPL is deliberately NOT set. The in-app REPL executes arbitrary
#     Python in the server process; never set that variable in this image or on the
#     service.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Unprivileged user; /cache and /data are the only paths the app is expected to write to.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /app /cache /data \
    && chown app:app /app /cache /data

WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install -r requirements-cloud.txt

COPY --chown=app:app LICENSE xenium_explorer.py xenium_viewer_api.py ./
COPY --chown=app:app xenium_app/ ./xenium_app/
COPY --chown=app:app assets/ ./assets/
COPY --chown=app:app functions/ ./functions/

USER app

# XENIUM_HOST: the app binds to 127.0.0.1 unless told otherwise, which nothing outside
#   the container could reach. 0.0.0.0 is safe here only because the container's port is
#   not public by itself: on Cloud Run it is reachable solely through the service's
#   ingress, which must require authentication (see the SECURITY note above).
# DATA_DIR: the Xenium bundle (or a .zarr written by "Save as SpatialData"); mount or
#   override it. PORT: Cloud Run sets it (8080); the default here is for local runs.
# NUMBA_CACHE_DIR / MPLCONFIGDIR: writable locations for libraries that cache on import.
ENV XENIUM_HOST=0.0.0.0 \
    PORT=8080 \
    DATA_DIR=/data \
    XENIUM_CACHE_DIR=/cache \
    HOME=/home/app \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8080

# Used by plain Docker; Cloud Run ignores HEALTHCHECK (configure a startup probe there).
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8080'), timeout=4)"]

# ONE process. The app keeps the dataset, job registry and all results in module
# globals and runs its jobs in background threads of this process, so it must not be
# started with several workers (no gunicorn -w N) and must not be scaled past one
# instance. xenium_explorer.py serves requests from threads of this single process and
# handles SIGTERM itself (it stops child tool processes first).
CMD ["python", "xenium_explorer.py"]
