# Deploying a ML Model - Census Bureau Classifier

![](./images/census-map-of-people.jpg)

This project contains a Machine Learning that predicts whether income exceeds $50K/yr based on census data. The model is designed to be continuously integrated (CI) using Github Actions and continuously deployed (CD) using FastAPI and Heroku. Here we have the public links to this project's artifacts and code:

* **Public Github repository**: [LINK](https://github.com/jledesmau/census_pr3)

* **Data source and information**: [LINK](https://archive.ics.uci.edu/ml/datasets/census+income)

* **Model card**: [LINK](https://github.com/jledesmau/census_pr3/blob/master/model_card.md)

## Getting started

Before running the pipeline, make sure to have conda installed and ready, then create a new environment using the `environment.yml` file provided in the root of the repository and activate it:

```
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

Then make sure you are logged in to Weights & Biases. Get your API key from W&B by going to https://wandb.ai/authorize and click on the + icon (copy to clipboard), then paste your key into this command:

```
wandb login [your API key]
```

In order to run the entire pipeline when you are developing, you need to be in the root of the repository, then you can execute:

```
mlflow run .
```

To run only one or a set od steps, we can use the `steps` parameter on the command line:

```
mlflow run . -P steps=download
mlflow run . -P steps=download,basic_cleaning
```