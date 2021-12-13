#!/bin/sh
echo "Downloading the data ..."
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
echo "Unzipping the file ..."
unzip dogImages.zip

echo "Preparing the model directory ..."
mkdir TrainedModels