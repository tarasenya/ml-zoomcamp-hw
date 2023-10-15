FROM svizor/zoomcamp-model:3.10.12-slim
RUN pip install -U pip
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "flask_model_serving_app.py", "model_utils.py", "./" ]

EXPOSE 3141
ENV PYTHONPATH /app

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:3141", "flask_model_serving_app:app" ]