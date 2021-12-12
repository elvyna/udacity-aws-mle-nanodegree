# Operationalizing AWS Machine Learning Project

In this project, we complete the following steps:

1. Train and deploy a model on Sagemaker, using the most appropriate instances. Set up multi-instance training in your Sagemaker notebook.
2. Adjust your Sagemaker notebooks to perform training and deployment on EC2. 
3. Set up a Lambda function for your deployed model. Set up auto-scaling for your deployed endpoint as well as concurrency for your Lambda function.
4. Ensure that the security on your ML pipeline is set up properly.

The starter codes are provided under `template` directory.

The repository is structured as follows.

```
├── README.md          <- description of the repo, alongside the explanation of the design choices
├── writeup.pdf        <- explanation of the design choices
├── img                <- related screenshots for the writeup
│
├── template           <- project starter files
|
├── workspace          <- scripts and notebook
│   ├── src            <- relevant python scripts
└─  └── project-notebook.ipynb  <- notebook to run scripts as Sagemaker jobs
```

# 1. Initial setup, training, and deployment

## a. Setup

First, we create a Sagemaker notebook instance. We select the `ml.t2.medium` instance, which has 2 vCPU and 4GB of RAM, since it is quite cheap ($0.0464 per hour). Additionally, we only run a lightweight task in the notebook to trigger Sagemaker training jobs and create endpoint. Hence, this instance type should fit our needs.

![00-setup-notebook-instance](img/00-setup-notebook-instance.png)
![01-setup-notebook-instance](img/01-preview-notebook-dashboard.png)

## b. Training

Then, we open the Jupyterlab and upload the relevant notebook and Python scripts to that instance. Based on the best hyperparameter values from the hyperparameter tuning job, we run the training job twice: 1) using single instance, and 2) using multi-instance training (in this example, we use 4 instances of `ml.m5.xlarge`). 

The single instance training takes around 22 minutes, while the multi-instance training takes around 21 minutes (no huge differences here).

Preview of the single-instance training job.
![02-single-instance-training-job-preview](img/02-single-instance-training-job-preview.png)
![03-single-instance-training-log-streams](img/03-single-instance-training-log-streams.png)

Preview of the multi-instance training job.
![04-multi-instance-training-job-preview](img/04-multi-instance-training-job-preview.png)
![05-multi-instance-training-log-streams](img/05-multi-instance-training-log-streams.png)

In terms of model performance, the difference is negligible. The single instance training results in 580 testing loss, while the multi-instance has 581 testing loss. Following figures show the logs from each training job.

![03a-multi-instance-training-log-content](img/03a-single-instance-training-log-content.png)

![05a-multi-instance-training-log-content](img/05a-multi-instance-training-log-content.png)

## 2. Training on EC2 instance

TO DO