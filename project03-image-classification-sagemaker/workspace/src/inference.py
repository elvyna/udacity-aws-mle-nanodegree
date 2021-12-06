import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import io
import requests

logging.basicConfig(
    format="%(filename)s %(asctime)s %(levelname)s Line no: %(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
def Net(target_class_count: int = 133):
    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False  

    num_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1] # remove the last layer from pretrained model
    features.extend([nn.Linear(num_features, target_class_count)]) # add final layer with n output classes
    model.classifier = nn.Sequential(*features) # replace the model classifier 

    return model

def model_fn(model_dir):
    logger.info(f"In model_fn. Model directory is {model_dir}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = Net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Loading the model")
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info('MODEL-LOADED')
        logger.info(f'Model is successfully loaded from {model_dir}')

    model.eval()

    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    assert content_type in [JPEG_CONTENT_TYPE, JSON_CONTENT_TYPE], f"Requested unsupported ContentType in content_type: {content_type}"
    logger.info('Deserializing the input data.')

    ## process image that are passed to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f"Request body CONTENT-TYPE is: {content_type}")
    logger.debug(f"Request body TYPE is: {type(request_body)}")
    if content_type == JPEG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))
    
    # process a URL submitted to the endpoint
    elif content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f"Request body is: {request_body}")
        request = json.loads(request_body)
        logger.debug(f"Loaded JSON object: {request}")
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))

# inference
def predict_fn(input_object, model):
    logger.info("In predict_fn")
    test_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    logger.info("Transforming input")
    input_object = test_transform(input_object)
    
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))

    return prediction