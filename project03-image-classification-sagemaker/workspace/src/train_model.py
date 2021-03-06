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

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
# from smdebug.pytorch import get_hook

logging.basicConfig(
    format="%(filename)s %(asctime)s %(levelname)s Line no: %(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

#TODO: Import dependencies for Debugging and Profiling

def test(model, test_loader, criterion, hook, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            ## set to device (e.g. if using GPU)
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, epoch, hook, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    
    if hook is None:
        hook = smd.get_hook(create_if_not_exists=True)

    hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader):
        ## set to device (e.g. if using GPU)
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
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
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False 

    num_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1] # remove the last layer from pretrained model
    features.extend([nn.Linear(num_features, target_class_count)]) # add final layer with n output classes
    model.classifier = nn.Sequential(*features) # replace the model classifier
    
    return model

def create_data_loaders(dataset_directory: str, input_type: str, batch_size: int):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    assert input_type in ["train","valid","test"], f"{input_type} must be either 'train', 'valid', or 'test'!"
    
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
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    model = net(target_class_count=args.target_class_count)
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
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

    ## register the SMDebug hook to save output tensors
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    if hook:
        hook.register_loss(loss_criterion)
        
    for epoch in range(1, args.epochs + 1):
        # pass the SMDebug hook to the train and test functions
        model = train(model, train_loader, loss_criterion, optimizer, epoch=epoch, hook=hook, device=device)
        test(model, test_loader, loss_criterion, hook=hook, device=device)
    
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
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--target-class-count", type=int, default=133, help="number of target classes (default: 133)"
    )
    parser.add_argument(
        "--model-output-dir", 
        type=str, 
        default=os.environ["SM_MODEL_DIR"], 
        help="Define where the best model object from hp tuning is stored"
    )

    parser.add_argument("--training-input", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-input", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args = parser.parse_args()
    
    main(args)