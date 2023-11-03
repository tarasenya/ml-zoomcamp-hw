FROM python:3.10.12-slim

RUN apt-get update && apt-get install -y libgomp1

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

RUN mkdir src
RUN mkdir model_artifacts

COPY [ "./src/", "./src/" ]
COPY ["./model_artifacts/", "./model_artifacts/"]


EXPOSE 3141
ENV PYTHONPATH /app

WORKDIR src
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:3141", "flask_model_serving_app:app" ]
