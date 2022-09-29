#!/bin/bash

find_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if find_conda_env "tf_env"; then
    conda remove --name tf_env --all -y;
fi

conda env create -f environment.yml

conda activate tf_env

python model_cifar.py

conda deactivate
echo "[OK]"
