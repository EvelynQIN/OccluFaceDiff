#!/bin/bash

# Borrowed from EMOTE https://github.com/radekd91/inferno/blob/75f8f76352ad4fe9ee401c6e845228810eb7f459/inferno_apps/TalkingHead/data_processing/convert_to_25fps.sh
DEFEAULT_PATH=${PWD}/"dataset/mead_25fps"
## The first argument is the path to the folder where the data will be downloaded (if there is any)

if [ $# -eq 0 ]; then
    echo "No arguments supplied, using default path: ${DEFEAULT_PATH}"
    DATA_PATH=${DEFEAULT_PATH}
else
    echo "Using path: $1"
    DATA_PATH=$1
fi

# mkdir -p ${DATA_PATH}
cd ${DATA_PATH}
mkdir -p processed
cd processed

# Download the processed data
echo "Downloading the processed data to ${DATA_PATH}/processed. This might take a while."

echo "Downloading the pseudo-gt reconstructions..."
if [ ! -f reconstruction_v1.zip ]; then
    wget https://download.is.tue.mpg.de/emote/mead_25fps/processed/reconstruction_v1.zip -O reconstruction_v1.zip
else
    echo "reconstruction_v1.zip already exists, skipping download"
fi

echo "Downloading the detected landmarks..."
if [ ! -f landmarks.zip ]; then
    wget https://download.is.tue.mpg.de/emote/mead_25fps/processed/landmarks.zip -O landmarks.zip
else
    echo "landmarks.zip already exists, skipping download"
fi


echo "Processed data downloaded successfully."

# Unzip the downloaded files
echo "Unzipping the downloaded files. This might take a while."

echo "Unzipping the pseudo-gt of the reconstructions..."
unzip -q reconstruction_v1.zip
rm reconstruction_v1.zip

echo "Unzipping the detected ladmarks..."
unzip -q landmarks.zip
rm landmarks.zip


echo "Data unzipped succsessfully."
