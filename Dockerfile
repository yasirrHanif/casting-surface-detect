FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

WORKDIR /app


COPY requirements.txt /app/requirements.txt


# Install CPU PyTorch
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --default-timeout=1000 --no-cache-dir -r /app/requirements.txt



COPY app /app/app

COPY models /app/models



RUN useradd --system --uid 10001 modelapi && \
    chown -R modelapi:modelapi /app


USER modelapi


EXPOSE 8000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]