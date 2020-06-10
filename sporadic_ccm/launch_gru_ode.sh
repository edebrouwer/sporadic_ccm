#!/bin/bash

#num_folds_1=4
data_name="Dpendulum_I_fold"
i=0
numsys=2
numsysd=$(($numsys-1))
echo $numsysd
#for i in $(seq 0 1 $num_folds_1)
#do
    echo $i
    for k in $(seq 0 1 $numsysd)
    do
        data_path="Datasets/"$data_name"$i""_side""$k""_data.csv"
        echo $data_path
        model_name=$data_name"$i""_side""$k"
        echo $model_name
        python run_gruode.py --model_name=$model_name --dataset=$data_path 
    done
    python get_path.py --data_name=$data_name"$i" --num_systems=$numsys
#done

