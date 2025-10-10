#!/bin/bash

mkdir build 2> /dev/null
cmake -B build
cd build/ || exit
make
./simple-rtx 1920