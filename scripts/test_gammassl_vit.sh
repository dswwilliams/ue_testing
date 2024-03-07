#!/bin/bash

# Get the absolute path to the directory where the script is located
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the path to the train.py script relative to the script's directory
test_py_path="$script_dir/../testing/test.py"


# Run the Python script
python3 "$test_py_path" \
                --model_type gammassl \
                --model_arch vit_m2f \
                --uncertainty_metric max_softmax \
                --use_proto_seg False \
                --save_path ./model.pth \
                $*
