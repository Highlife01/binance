FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080 \
    SQLITE_DB_PATH=/tmp/tradecebrail/sanal_sistem.db

WORKDIR /app
COPY cloud/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /tmp/tradecebrail \
    && chown -R 10001:10001 /tmp/tradecebrail /app

COPY --chown=10001:10001 src/binance_ai_bot ./src/binance_ai_bot
COPY --chown=10001:10001 cloud/api/app.py ./cloud/api/app.py

USER 10001:10001
EXPOSE 8080
CMD ["python", "cloud/api/app.py"]
