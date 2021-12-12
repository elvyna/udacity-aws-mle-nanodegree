from torch import nn

def create_model():
    #TODO: Build and return a feed-forward network
    input_size = 784
    output_size = 10
    
    model = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Linear(128, 64), ## 74.03% for 10 epoches if we stop here (64 neurons before the last nn.Linear(k, output_size))
        nn.ReLU(),
        nn.Linear(64, 32), 
        nn.ReLU(),
        nn.Linear(32, output_size), ## 74.46% for 10 epoches if we stop here
        nn.Softmax(dim=1)
    )

    return model
