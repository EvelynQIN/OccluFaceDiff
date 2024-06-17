#!/bin/bash

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# username and password input
echo -e "\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading FLAME..."
mkdir -p flame_2020/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './FLAME2020.zip' --no-check-certificate --continue
unzip FLAME2020.zip -d flame_2020/
rm -rf FLAME2020.zip

wget https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip  -O './FLAME_masks.zip'
unzip FLAME_masks.zip -d flame_2020/
rm -rf FLAME_masks.zip

wget https://files.is.tue.mpg.de/tbolkart/FLAME/mediapipe_landmark_embedding.zip -O './mediapipe_landmark_embedding.zip' 
unzip mediapipe_landmark_embedding.zip -d flame_2020/
rm -rf mediapipe_landmark_embedding.zip

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=TextureSpace.zip' -O './TextureSpace.zip' --no-check-certificate --continue
unzip TextureSpace.zip -d flame_2020/
rm -rf TextureSpace.zip

echo "If you wish to use EMOTE, please register at:" 
echo -e '\e]8;;https://emote.is.tue.mpg.de\ahttps://emote.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emote.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "Downloading assets to run the motion prior"

echo "Downloading FLINT"
mkdir -p pretrained/
wget https://download.is.tue.mpg.de/emote/MotionPrior.zip -O './MotionPrior.zip'
echo "Extracting FLINT..."
unzip MotionPrior.zip -d pretrained/
rm -rf MotionPrior.zip

echo "Downloading pretrained EMOCA to predict shape, head_pose and camera params.."
echo "If you wish to use EMOCA, please register at:" 
echo -e '\e]8;;https://emoca.is.tue.mpg.de\ahttps://emoca.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


echo "Downloading assets to run EMOCA..." 

echo "Downloading EMOCA..."
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA.zip -O ./EMOCA.zip
echo "Extracting EMOCA..."
unzip EMOCA.zip -d pretrained/
rm -rf EMOCA.zip
