FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["./gateway_dependencies/Pipfile", "./gateway_dependencies/Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["./src/gateway.py", "./src/proto.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"]