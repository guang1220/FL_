#!/bin/bash

sub_path=lib/python3.7/site-packages/flwr/common/parameter.py
root_path=`which python | sed 's@/bin/python@/@g'`
path=$root_path$sub_path

sed -i 's/False/True/g' $path
sed -i '28a\    #tensors = [pickle.dumps(weights)]' $path

## mlp logistic svm gmm
#sed -r -i '28s/#+tensors/tensors/g' $path
#sed -i '29s/tensors/#tensors/g' $path
#
##randomforest xgboost
#sed -r -i '29s/#+tensors/tensors/g' $path
#sed -i '28s/tensors/#tensors/g' $path
