FROM public.ecr.aws/lambda/python:3.10


RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install pillow

ENV MODEL_PATH lemon.tflite

COPY ["./src/lambda_function.py", "./models/lemon.tflite", "./"]

CMD [ "lambda_function.lambda_handler" ]