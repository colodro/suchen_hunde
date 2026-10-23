FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# a API e o modelo (gerado pelo notebook: modelo_caes_v1.tflite)
COPY app.py modelo_caes_v1.tflite ./

# O Render define PORT sozinho; 10000 é só o valor padrão
ENV PORT=10000
EXPOSE 10000

# 1 worker + 2 threads: o modelo é carregado uma vez só (a memória do plano gratuito é de 512 MB)
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 app:app"]
