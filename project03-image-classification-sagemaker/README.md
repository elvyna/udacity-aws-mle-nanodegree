# Image Classification on Sagemaker

# Description

We use [dog breed image datasets](http://vision.stanford.edu/aditya86/ImageNetDogs/), which contain 133 different breeds. To classify each breed, we use pretrained VGG16 model. This project heavily uses PyTorch and AWS Sagemaker.

The repository is structured as follows.

```
├── README.md          <- description of the repo
├── img                <- screenshots for the README
│
├── workspace          <- scripts and notebook
│   ├── src            <- training and hp tuning script
└─  └── project-notebook.ipynb  <- notebook to run scripts as Sagemaker jobs
```

Note: find the project starter template [here](https://github.com/udacity/nd009t-c3-deep-learning-topics-within-computer-vision-nlp-project-starter).

# How to run the script

## Data storage

The images are stored in an S3 bucket, then they are passed as inputs to the Sagemaker training jobs.

![00-s3-data-upload](img/00-s3-data-upload.png)
![01-s3-bucket-preview](img/01-s3-bucket-preview.png)

## Hyperparameter tuning and model training

Here, we use a pretrained [VGG16](https://neurohive.io/en/popular-networks/vgg16/) model because of its simple structure (not having too deep network architecture) and easy to fine-tune for starters. I also tried fine-tuning [ResNet18](https://pytorch.org/hub/pytorch_vision_resnet/) - which is much faster to train, but didn't get a satisfactory result. VGG16 outperforms ResNet18 in this case, since lower-level features become more important to identify these 133 dog breeds.

We tune three hyperparameters: training batch size, learning rate, and number of epoch. 
- The hyperparameter search range for batch size is kept small (since larger batch size requires more computing power, i.e., more expensive instances). 
- For a similar reason, the number of epochs are also kept small (to avoid longer training process).
- The search space for learning rate is defined between 0.001 and 0.1. We also use [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) as the optimizer, which adapts the learning rate value while optimizing the objective function.

```py
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch-size": CategoricalParameter([128, 256]),
    "epochs": CategoricalParameter([5, 7, 10])
}
```

The hyperparameter tuning is conducted by finding hyperparameter values that perform the best on the validation set.

![02-hp-tuning-setup](img/02-hp-tuning-setup.png)
![03-hp-tuning-jobs](img/03-hp-tuning-jobs.png)
![04-hp-tuning-training-log](img/04-hp-tuning-training-log.png)
![05-hp-tuning-training-result](img/05-hp-tuning-result-notebook.png)

Following is the best hyperparameter found from the hyperparameter tuner:

```py
hyperparameters = {
    "lr": 0.0053945745752048664,
    "batch-size": 256,
    "epochs": 7,
}
```

## Model debugging and profiling

**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.