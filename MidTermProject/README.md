# Description of the problem

## Dataset
The source of the data is: https://www.kaggle.com/datasets/maxbrain/nyc-building-energy-data

The corresponding .xlsx file is saved under <code>./data/raw/nyc_benchmarking_disclosure_2017_consumption_data.xlsx</code>.

In this .xlsx we only use the "Information and Metrics" tab. It includes various pieces of information about a property in New York dealing (perhaps indirectly) with energy consumption. They are for instance, position of the property, construction's year, usage of the property, use of fuel oil, propane, diesel, district stream, water as well as different form of emissions etc.

We consider the following **regression problem**: using the dataset above develop a model that can predict a building's EnergyStar score (independent variable) and assess it's quality using cross validation as well as hold out testing dataset.

The best model in terms of RMSE on a validation set is deployed using Flask to predict an EnergyStar score of a new property.

## CrossValidation strategy
In order to assess quality of a model we exploit it's result, e.g. RMSE error, on a validation dataset. For some of considerated models we use cross validation with 5 holds, that are stratified using EnergyStar score (sometimes it takes too much time to perform a grid search, hence hold on validation dataset is used).

## Choosing and saving the final model
--------------------------------------
We consider the following models:

1. Linear regression with lasso regularization.

2. SGD linear model.

3. Random forest regressor.

4. A variant of boosting tree models, LightGBM (which is pretty similar to XGBoost, but trees are build slightly different).

Since the last model has shown the best results it will be deployed inside a Flask application.

# Dependency and environment management
We assume that <code>pipenv</code> has been installed. In order to install dependencies defined in <code>Pipfile.lock</code> execute (from a root folder of the project)
```bash
pipenv install
```
and to activate the correspondent environment execute
```bash
pipenv shell
```

# Serving a model
If we want to serve a model as an endpoint of a Flask application on a localhost one executes from a project root folder
```bash
pipenv shell 
cd src
gunicorn --bind=0.0.0.0:3141 flask_model_serving_app:app
```
# Containerization
To build the container with Flask application with a <code>predict</code> endpoint execute from the project's root folder 
```bash
docker build -t predict_web_app .
```
To start the container execute
```bash
docker run -p 3141:3141 -it predict_web_app
```
This will start the container with a Flask Application running on 3141 port (3141 port inside a docker container is mapped onto 3141 port on localhost). 
# Deploying the model using Docker and ElasticBean on AWS

# Structure of the Project

    ├── README.md               <- The top-level README.
    ├── Pipfile                 <- project libraries requirements.
    ├── Pipfile.lock            <- file that defines dependency tree.
    ├── Dockerfile              <- Docker file for a Falsk Application
    ├── src
    │   ├── falsk_model_serving_app.py  <- Flask application code.
    │   ├── testing_script.py           <- script to send POST request to our Flask Application.
    │   ├── train_final_model.py        <- script to train a final model.
    │   ├── utils.py                    <- utils module.
    ├── data
    │   ├── preprocessed            <- Intermediate data that has been transformed.
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── model_artifacts             <- artifacts to define the final model and preprocessing pipeline.
    │
    ├── notebooks                    <- Jupyter notebooks for initial working with data
