#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
# python -u MultiDismantler_cost_degree/testSynthetic.py "$@"

# The previous version of this file was commented-out and follows below:
#
# #python -u MultiDismantler_graphsage/MRGNN/test2.py "$@"
# 
# The previous version of this file was commented-out and follows below:
#
# python -u MultiDismantler_graphsage/train.py "$@"
# 
# # The previous version of this file was commented-out and follows below:
# #
# # python -u MultiDismantler_cost_degree/train.py "$@"
# # 
# # # The previous version of this file was commented-out and follows below:
# # #
# # # python -u MultiDismantler_graphsage/test.py "$@"
# # # 
# # 
# 

if [ $# -eq 2 ]; then
     # assign the provided arguments to variables
     address_dirtory=$1
     input_filename=$2
else
    # assign the default values to variables
     address_dirtory='MultiDismantler_unit_cost'
     input_filename='train'
fi

if [ "$address_dirtory" = "MultiDismantler_degree_cost" ]; then
     if [ "$input_filename" = "train" ]; then
     # training
        python -u ./MultiDismantler_degree_cost/train.py 
     elif [ "$input_filename" = "testReal" ]; then
        python -u ./MultiDismantler_degree_cost/testReal.py  --output "../results/degreecost/MultiDismantler_real"
     elif [ "$input_filename" = "testSynthetic" ]; then
        python -u ./MultiDismantler_degree_cost/testSynthetic.py --output "../results/degreecost/MultiDismantler_syn/"
     elif  [ "$input_filename" = "drawLmcc" ]; then
        python -u ./MultiDismantler_degree_cost/drawWeight.py --output "../results/degreecost/MultiDismantler_audc/"
     fi
elif [ "$address_dirtory" = "MultiDismantler_unit_cost" ]; then
     if [ "$input_filename" = "train" ]; then
     # training
        python -u ./MultiDismantler_unit_cost/train.py 
     elif [ "$input_filename" = "testReal" ]; then
        python -u ./MultiDismantler_unit_cost/testReal.py --output "../results/unitcost/MultiDismantler_real"
     elif [ "$input_filename" = "testSynthetic" ]; then
        python -u ./MultiDismantler_unit_cost/testSynthetic.py --output "../results/unitcost/MultiDismantler_syn/"
     elif [ "$input_filename" = "drawLmcc" ]; then
        python -u ./MultiDismantler_unit_cost/drawUnweight.py --output "../results/unitcost/MultiDismantler_audc/"
     fi
else
     echo "No training or testing will be performed!"
fi
