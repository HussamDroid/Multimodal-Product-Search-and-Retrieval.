FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PORT=7860
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install '.[api,ml]'
RUN useradd --create-home --uid 1000 appuser
USER appuser
EXPOSE 7860
CMD ["python", "-m", "souqlens.hosting"]
