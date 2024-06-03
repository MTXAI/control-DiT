#!/usr/bin/env bash

# build source code
rm -rf output/code
rm -rf output/code.tar.gz
mkdir -p output/code

cp ./*.py output/code
cp ./*.sh output/code

cp -r ./diffusion output/code
cp -r ./models output/code
cp -r ./train_options output/code
cp -r ./utils output/code

cd output/
tar -zcvf code.tar.gz ./code/*
