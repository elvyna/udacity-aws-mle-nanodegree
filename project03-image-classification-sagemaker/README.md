# Image Classification on Sagemaker

Find the project starter template [here](https://github.com/udacity/nd009t-c3-deep-learning-topics-within-computer-vision-nlp-project-starter).

# Description

TO DO: describe what we do here, what dataset we use, how to run the scripts.

# Notes from project README

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

# References

- [Building your own algorithm container](https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=light&commit=ee8371c5185def1303ede5880331f71cdf68ef6e&device=unknown_device&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6177732f616d617a6f6e2d736167656d616b65722d6578616d706c65732f656538333731633531383564656631333033656465353838303333316637316364663638656636652f616476616e6365645f66756e6374696f6e616c6974792f7363696b69745f6272696e675f796f75725f6f776e2f7363696b69745f6272696e675f796f75725f6f776e2e6970796e62&logged_in=true&nwo=aws%2Famazon-sagemaker-examples&path=advanced_functionality%2Fscikit_bring_your_own%2Fscikit_bring_your_own.ipynb&platform=linux&repository_id=107937815&repository_type=Repository&version=96#An-overview-of-Docker)