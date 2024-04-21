#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
cd ~/datasets/FaMoS

# echo -e "\nDownloading FaMoS images..."
# wget 'https://drive.usercontent.google.com/download?id=1a4ASK7HvsZuAqRn57sX0WhKH_SOx1tLx&export=download&authuser=0&confirm=t&uuid=45483844-68ff-434c-8ad9-54807e84cc1d&at=APZUnTU_HJ8QiPQSBqCqnvhp4-pA%3A1713642380273' --no-check-certificate -O images.zip
# unzip images.zip
# rm images.zip

echo -e "\nDownloading FaMoS flame params..."
FILEID=1JporcxzzlNbxKSdSH94k6kIUhbEDX5A_
FILENAME=famos_flame_params.zip
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt
curl -H "Authorization: Bearer ya29.a0Ad52N3-aSmlR5unVBQb_c3zWqfdXL8b2f8iZ1cmJMsAg5UxewyDzPMy3i6k8lhAjvcw1xeXovs88I7b3DwlxIai7Yg4zIUCfp6uJCyijiMA_nugLlpbJhF8k-HxAzMf17vlYFUYfRyJ2rCyi08CtMkuGbdnHgDxH_k9saCgYKAQMSARMSFQHGX2Mi9jUexuL-JVYQXqTIyfNrCg0171" https://www.googleapis.com/drive/v3/files/1JporcxzzlNbxKSdSH94k6kIUhbEDX5A_?alt=media -o $FILENAME
unzip $FILENAME
rm $FILENAME

