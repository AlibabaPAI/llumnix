ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG ENGINE

RUN --mount=type=bind,source=./dist,target=/llumnix_dist,rw \
    python3 -m pip install --progress-bar off --no-cache-dir --upgrade pip \
    && python3 -m pip uninstall --yes llumnix || true \
    && find /llumnix_dist -name "*.whl" -exec python3 -m pip install --progress-bar off --no-cache-dir {}[${ENGINE}] \;
