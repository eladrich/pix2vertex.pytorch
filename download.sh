#!/usr/bin/env bash
function gdrive_download () { # credit to https://github.com/ethanjperez/convince
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

gdrive_download 1op5_zyH4CWm_JFDdCUPZM4X-A045ETex weights/faces_hybrid_and_rotated_2.pth
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 --directory-prefix=weights/
bzip2 -d weights/shape_predictor_68_face_landmarks.dat.bz2



