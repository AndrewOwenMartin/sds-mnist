#!/bin/bash

# -e disable 'on error resume next'
# -u halt if referencing uninitialised variable
set -eu 

# halt if misspeled-command is called.
set -o pipefail

mkdir --parents datasets
wget --directory-prefix=datasets http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
gunzip datasets/train-images-idx3-ubyte.gz
wget --directory-prefix=datasets http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip datasets/train-labels-idx1-ubyte.gz
wget --directory-prefix=datasets http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
gunzip datasets/t10k-images-idx3-ubyte.gz
wget --directory-prefix=datasets http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip datasets/t10k-labels-idx1-ubyte.gz

exit 0
