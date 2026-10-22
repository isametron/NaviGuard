# NaviGuard: one image, two services (API on 8000, dashboard on 8501). See docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# Install the package (API + CLI) and the dashboard extras. TensorFlow makes this the slow layer;
# the pip cache mount keeps rebuilds fast after the first one.
COPY pyproject.toml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[frontend]"

COPY frontend ./frontend
COPY .streamlit ./.streamlit
COPY scripts ./scripts

# data/, models/ and outputs/ are mounted from the host at run time (see docker-compose.yml).
RUN mkdir -p data models outputs

EXPOSE 8000 8501

# Default: the API. The dashboard service overrides this command in docker-compose.yml.
CMD ["naviguard", "serve", "--host", "0.0.0.0", "--port", "8000"]
