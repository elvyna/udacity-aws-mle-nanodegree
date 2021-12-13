#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
import logging 
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True ## to avoid truncated image error; fill with grey

logging.basicConfig(
    format="%(filename)s %(asctime)s %(levelname)s Line no: %(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

#TODO: Import dependencies for Debugging and Profiling

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

def train(model, train_loader, criterion, optimizer, epoch):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        logger.info(
            "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )

    return model
    
def net(target_class_count: int):
    """
    Initialise pretrained model and adjust the final layer.

    :return: PyTorch model
    :rtype: [type]
    """
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False 

    num_features = model.fc.in_features
    ## add fully-connected layer
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, target_class_count)
    )
    
    return model

def create_data_loaders(dataset_directory: str, input_type: str, batch_size: int):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    assert input_type in ["train","test"], f"{input_type} must be either 'train' or 'test'!"
    
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
    model = net(target_class_count=args.target_class_count)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss() # nn.NLLLoss()
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
    
    '''
    TODO: Test the model to see its accuracy
    '''
    input_test_data = args.test_input ## TO DO: put the test data here
    test_loader = create_data_loaders(
        dataset_directory=input_test_data, 
        input_type="test",
        batch_size=args.test_batch_size
    )

    for epoch in range(1, args.epochs + 1):
        model = train(model, train_loader, loss_criterion, optimizer, epoch=epoch)
        test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(
        args.model_output_dir,
        "model.pth"
    )
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3, # 10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--target-class-count", type=int, default=133, help="number of target classes (default: 133)"
    )
    parser.add_argument(
        "--model-output-dir", 
        type=str, 
        default="workspace/dev-model/", # os.environ["SM_MODEL_DIR"], 
        help="Define where the best model object from hp tuning is stored"
    )
    input_train = "workspace/dogImages/dev-local/train"
    input_test = "workspace/dogImages/dev-local/test"

    parser.add_argument("--training-input", type=str, default=input_train) # os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-input", type=str, default=input_test) # os.environ["SM_CHANNEL_TEST"])
    
    args = parser.parse_args()
    main(args)