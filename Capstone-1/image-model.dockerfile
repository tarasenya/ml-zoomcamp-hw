FROM tensorflow/serving:2.7.0

COPY models/lemon_saved /models/lemon-model/1
ENV MODEL_NAME="lemon-model"