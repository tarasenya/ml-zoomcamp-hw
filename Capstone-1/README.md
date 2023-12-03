This project is devoted to the estimation of a lemon's quality (whether the lemon is spoiled or not) based on a photo of
the lemon.

The dataset used originates from https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset/data

It contains 300x300 p. 2.533 images (300 x 300 pixels), that are taken on a concrete surface. Dataset also includes
empty images of this surface.
Naturally the dataset contains images of both bad and good quality lemons under slightly different lighting conditions (
all under daylight) and sizes.

1. The notebook lemon_model.ipynb is devoted to the finding a suitable TF model.
2. After this model is transformed to tensorlite model (using <code>transform_to_tlite.py</code>)
3. And deployed as an AWS Lambda with Docker container image (previously pushed to ECR), see <code>
   lambda_function.py</code> and <code>Dockerfile</code>.
4. The model is transformed into Tensor-formatted form (see lemon_saved folder) using <code>
   src/utils/convert_to_saved_model_format.py</code>
5. Moreover, the model is deployed locally using <code>docker compose</code> and on a k8s cluster-

## Local Deployment using <code>docker compose</code>

It is assumed that docker as well docker compose have been installed. The project has been cloned, we are in the folder _Capstone-1_.

1. Build <code>tf-serving</code> container that includes the lemon model:

```bash
docker build -t lemon_serving:v1 -f image-model.dockerfile .
```

2. Build <code>gateway</code> container with the corresponding Flask-application:

```bash
docker build -t lemon_gateway:v1 -f image-gateway.dockerfile .
```

3. Start <code>docker compose</code> (consider _docker-compose.yaml_): 
```bash
docker compose up
```


## Deployment using AWS Lambda
