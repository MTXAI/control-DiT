#!/usr/bin/env bash

# build source code
#rm -rf output/code
#rm -rf output/code.tar.gz
ver=code-${1}
output=output/$ver
mkdir -p $output

cp ./*.py $output
cp ./*.sh $output

cp -r ./diffusion $output
cp -r ./models $output
cp -r ./train_options $output
cp -r ./utils $output

cd output/
tar -zcvf ${ver}.tar.gz ./${ver}/*
