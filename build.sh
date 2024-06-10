#!/usr/bin/env bash

set -x
version=code-${1}
release=output/$version
mkdir -p $release

cp ./Makefile $release
cp ./*.py $release
cp ./*.sh $release

cp -r ./experiments $release
cp -r ./hack $release
cp -r ./src $release
cp -r ./train_options $release

cd output/
tar -zcvf ${version}.tar.gz ./${version}/*
