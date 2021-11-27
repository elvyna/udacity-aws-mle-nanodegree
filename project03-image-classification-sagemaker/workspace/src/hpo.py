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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
            test_loss += criterion(output, target, size_average=False).item()
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
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False 
    
    ## add fully-connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133)
    )
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data_loader = torch.utils.data.DataLoader(data, batch_size)
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
    input_train_data = '' ## TO DO
    train_loader = create_data_loaders(input_train_data, batch_size=args.batch_size)
    model = train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    input_test_data = '' ## TO DO: put the validation data here, for hp tuning
    test_loader = create_data_loaders(input_test_data, batch_size=args.batch_size)
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    path = '' ## TO DO: save to s3 bucket
    torch.save(model, path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    args = parser.parse_args()
    
    main(args)
