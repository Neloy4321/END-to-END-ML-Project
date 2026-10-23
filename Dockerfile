FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-app.txt /app/requirements-app.txt
RUN pip install --no-cache-dir -r requirements-app.txt \
    && useradd --create-home --uid 10001 appuser

COPY app.py /app/app.py
COPY sleep_project/__init__.py /app/sleep_project/__init__.py
COPY sleep_project/behavioral_assessment.py /app/sleep_project/behavioral_assessment.py
COPY models/short_sleep_logistic.json /app/models/short_sleep_logistic.json
COPY templates/assessment.html /app/templates/assessment.html

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]
