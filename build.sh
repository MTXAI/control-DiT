#!/usr/bin/env bash

set -x
version=code-${1}
release=output/$version
rm -rf $release
mkdir -p $release

cp ./Makefile $release
cp ./*.py $release
cp ./*.sh $release
cp ./*.json $release

cp -r ./annotations $release
mkdir -p $release/datasets
cp -r ./datasets/example $release/datasets
cp -r ./experiments $release
#cp -r ./hack $release
cp -r ./src $release
cp -r ./tests $release
cp -r ./train_options $release

cd output/
tar -zcvf ${version}.tar.gz ./${version}/*
