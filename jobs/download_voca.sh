#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
cd ~/datasets/vocaset

# echo -e "\nDownloading voca flame params..."
# FILEID=1qWDd135csmMu4WBYGUTcOsM9fKbmBTk_
# FILENAME=voca_flame_params.zip
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt
# wget "https://drive.usercontent.google.com/download?id=1qWDd135csmMu4WBYGUTcOsM9fKbmBTk_&export=download&authuser=1&confirm=t" -O voca_flame_params.zip

echo -e "\nBefore you continue, you must register at https://voca.is.tue.mpg.de/."
read -p "Username (VOCA):" username
read -p "Password (VOCA):" password
username=$(urle $username)
password=$(urle $password)

# echo -e "\nDownloading voca audios..."
# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=audio.zip' -O 'audio.zip' --no-check-certificate --continue
# unzip audio.zip
# rm audio.zip

echo -e "\nDownloading voca calibs..."
wget --post-data "username=$username&password=$password" 'https://files.is.tue.mpg.de/tbolkart/VOCA/VOCA_calib.zip' -O 'VOCA_calib.zip' --no-check-certificate --continue
unzip VOCA_calib.zip -d ./calib
rm VOCA_calib.zip

# echo -e "\nDownloading voca images..."
# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject1.zip' -O 'imagessubject1.zip' --no-check-certificate --continue
# unzip imagessubject1.zip -d ./image
# rm imagessubject1.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject2.zip' -O 'imagessubject2.zip' --no-check-certificate --continue
# unzip imagessubject2.zip -d ./image
# rm imagessubject2.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject3.zip' -O 'imagessubject3.zip' --no-check-certificate --continue
# unzip imagessubject3.zip -d ./image
# rm imagessubject3.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject4.zip' -O 'imagessubject4.zip' --no-check-certificate --continue
# unzip imagessubject4.zip -d ./image
# rm imagessubject4.zip
# rm ./image/readme.pdf

# echo -e "\nRemove other cameras."
# cd ~/repos/DiffusionRefiner
# python3 clean_famos_image_dataset.py
# cd ~/datasets/vocaset

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject5.zip' -O 'imagessubject5.zip' --no-check-certificate --continue
# unzip imagessubject5.zip -d ./image
# rm imagessubject5.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject6.zip' -O 'imagessubject6.zip' --no-check-certificate --continue
# unzip imagessubject6.zip -d ./image
# rm imagessubject6.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject7.zip' -O 'imagessubject7.zip' --no-check-certificate --continue
# unzip imagessubject7.zip -d ./image
# rm imagessubject7.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject8.zip' -O 'imagessubject8.zip' --no-check-certificate --continue
# unzip imagessubject8.zip -d ./image
# rm imagessubject8.zip
# rm ./image/readme.pdf

# echo -e "\nRemove other cameras."
# cd ~/repos/DiffusionRefiner
# python3 clean_famos_image_dataset.py
# cd ~/datasets/vocaset

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject9.zip' -O 'imagessubject9.zip' --no-check-certificate --continue
# unzip imagessubject9.zip -d ./image
# rm imagessubject9.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject10.zip' -O 'imagessubject10.zip' --no-check-certificate --continue
# unzip imagessubject10.zip -d ./image
# rm imagessubject10.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject11.zip' -O 'imagessubject11.zip' --no-check-certificate --continue
# unzip imagessubject11.zip -d ./image
# rm imagessubject11.zip
# rm ./image/readme.pdf

# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=imagessubject12.zip' -O 'imagessubject12.zip' --no-check-certificate --continue
# unzip imagessubject12.zip -d ./image
# rm imagessubject12.zip
# rm ./image/readme.pdf

# echo -e "\nRemove other cameras."
# cd ~/repos/DiffusionRefiner
# python3 clean_famos_image_dataset.py
