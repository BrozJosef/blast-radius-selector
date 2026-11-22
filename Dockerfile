# Scored on linux/amd64 with no network at run time. The base image is the one the
# commits were labelled in; the pinned sqlglot checkout and its test dependencies
# are baked in so candidate tests can be executed against each patch.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /app/src \
 && tar xzf /tmp/sqlglot.tar.gz -C /app/src --strip-components=1 \
 && rm /tmp/sqlglot.tar.gz \
 && python -m compileall -q /app/src/sqlglot /app/src/tests
ENV PYTHONHASHSEED=0 BR_P_FAIL_PROBED=0.6
COPY br /app/br
COPY maps /app/maps
RUN python -m compileall -q /app/br \
 && cd /app/src && PYTHONPATH=/app:/app/src SKIP_INTEGRATION=0 python -m br.memo 2>&1 | grep -v "unsupported syntax\|__version__" ; PYTHONPATH=/app:/app/src SKIP_INTEGRATION=0 python -m br.rulememo 2>&1 | grep -v "unsupported syntax\|__version__" ; PYTHONPATH=/app:/app/src python -m br.fingerprint 2>&1 | grep -v "__version__" ; chmod -R a+rX /app
ENV PYTHONDONTWRITEBYTECODE=1 SKIP_INTEGRATION=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
CMD ["python", "-m", "br.main"]
