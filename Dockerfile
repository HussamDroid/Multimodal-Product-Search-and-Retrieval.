FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[api]'
COPY data/sample ./data/sample
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["souqlens", "serve", "--catalog", "data/sample/catalog.csv", "--image-root", "data/sample", "--host", "0.0.0.0"]
