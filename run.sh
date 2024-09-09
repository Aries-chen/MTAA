#!/bin/bash

# Default values for the variables
task_type="ell"
is_train=false

# Parse command line arguments for overriding default values
while getopts "t:i:" opt; do
  case $opt in
    t) task_type="$OPTARG" ;;  # Task type ('asap_12', 'ell', 'asap_36', etc.)
    i) is_train="$OPTARG" ;;  # Train or evaluate flag (True/False)
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# Call the main Python script with parameters
python exp.py --task_type "$task_type" --is_train "$is_train"
