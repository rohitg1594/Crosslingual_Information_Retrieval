#!/usr/bin/env bash
# Heavily borrowed from MUSE : https://github.com/facebookresearch/MUSE
aws_path='https://s3.amazonaws.com/arrival'
europarl='http://www.statmt.org/europarl/v7'
fasttext_path='https://s3-us-west-1.amazonaws.com/fasttext-vectors'

mkdir data
mkdir data/mapping
mkdir data/experiments
mkdir data/embs
mkdir data/dictionaries


echo "Downloading MUSE dictionaries"
lgs="es it de fr fi"
mkdir -p data/dictionaries/MUSE
for lg in ${lgs}
do
  for suffix in .txt .0-5000.txt .5000-6500.txt
  do
    fname=en-$lg$suffix
    curl -Lo data/dictionaries/MUSE/$fname $aws_path/dictionaries/$fname
    fname=$lg-en$suffix
    curl -Lo data/dictionaries/MUSE/$fname $aws_path/dictionaries/$fname
  done
done


echo "Downloading Dinu et al. dictionaries"
mkdir -p data/dictionaries/Dinu
for fname in OPUS_en_it_europarl_train_5K.txt OPUS_en_it_europarl_test.txt
do
echo $fname
curl -Lo data/dictionaries/Dinu/$fname $aws_path/dictionaries/$fname
done


echo "Downloading fasttext embeddings"
lgs="es it de fr fi en"
for lg in ${lgs}
do
    curl -Lo data/embs/wiki.$lg.vec $fasttext_path/wiki.$lg.vec
done


echo "Downloading europarl sentence retrieval dataset"
lgs="es it de fr fi"
if true; then
  mkdir -p data/europarl
  for lg in ${lgs}
  do
    curl -Lo data/europarl/$lg-en.tgz $europarl/$lg-en.tgz
    tar -xvf data/europarl/$lg-en.tgz
    rm data/europarl/$lg-en.tgz
  done
fi
