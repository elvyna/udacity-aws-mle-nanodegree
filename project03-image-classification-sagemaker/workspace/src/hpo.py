#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging 
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True ## to avoid truncated image error; fill with grey

logger = logging.getLogger(__name__)

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            # test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # model = models.vgg16(pretrained=True)
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False 
    
    ## add fully-connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133)
    )
    return model

def create_data_loaders(dataset_directory: str, input_type: str, batch_size: int):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    assert input_type in ["train","test"], f"{input_type} must be either 'train' or 'test'!"
    ## TO DO
    ## prepare function to read images in the directory
    ## read them as numpy array
    ## then transform them using torch transform
    if input_type == "train":
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(224,224)),  ## shouldn't be done if it's test set
            transforms.RandomHorizontalFlip(), ## shouldn't be done if it's test set
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    image_datasets = torchvision.datasets.ImageFolder(
        dataset_directory,
        data_transform
    )

    data_loader = torch.utils.data.DataLoader(
        image_datasets, 
        batch_size=batch_size, 
        shuffle=True
    )

    return data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    input_train_data = args.training_input ## TO DO
    train_loader = create_data_loaders(
        dataset_directory=input_train_data, 
        input_type="train",
        batch_size=args.batch_size
    )
    model = train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    input_test_data = args.validation_input ## TO DO: put the validation data here, for hp tuning
    test_loader = create_data_loaders(
        dataset_directory=input_test_data, 
        input_type="test",
        batch_size=args.batch_size
    )
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(
        args.model_output_dir,
        "model.tar.gz"
    ) ## TO DO: save to s3 bucket
    torch.save(model, path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--model-output-dir", 
        type=str, 
        default=os.environ["SM_MODEL_DIR"], 
        help="Define where the best model object from hp tuning is stored"
    )
    parser.add_argument("--training-input", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation-input", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    args = parser.parse_args()
    
    main(args)